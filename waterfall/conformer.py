import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
import torchaudio
from waterfall import graph
from waterfall.manual_ctc import ctc, prior, softmax_ctc, eta_scheduler
import k2
from waterfall.utils.datapipe import Lang
import time

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

class Conv2dSubsampling2(nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-6:2]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling3(nn.Module):
    """Convolutional 2D subsampling (to 1/3 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling3 object."""
        super(Conv2dSubsampling3, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 2 - 1) // 3) + 1 - 2), odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-8:3]

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]
class Encoder(torch.nn.Module):
    """Conformer encoder module.

    Args:
        idim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        pos_enc_layer_type (str): Encoder positional encoding layer type.
        selfattention_layer_type (str): Encoder attention layer type.
        activation_type (str): Encoder activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
        cnn_module_kernel (int): Kernerl size of convolution module.
        padding_idx (int): Padding idx for input_layer=embed.
        stochastic_depth_rate (float): Maximum probability to skip the encoder layer.
        intermediate_layers (Union[List[int], None]): indices of intermediate CTC layer.
            indices start from 1.
            if not None, intermediate outputs are returned (which changes return type
            signature.)

    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer="conv2d",
        normalize_before=True,
        concat_after=False,
        positionwise_layer_type="linear",
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        pos_enc_layer_type="abs_pos",
        selfattention_layer_type="selfattn",
        activation_type="swish",
        use_cnn_module=False,
        zero_triu=False,
        cnn_module_kernel=31,
        padding_idx=-1,
        stochastic_depth_rate=0.0,
        intermediate_layers=None,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()

        activation = get_activation(activation_type)
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            pos_enc_class = LegacyRelPositionalEncoding
            assert selfattention_layer_type == "legacy_rel_selfattn"
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        self.conv_subsampling_factor = 1
        if input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(idim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "conv2d":
            self.embed = Conv2dSubsampling(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 4
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 2
        elif input_layer == "conv2d3":
            self.embed = Conv2dSubsampling3(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 3
        elif input_layer == "vgg2l":
            self.embed = VGG2L(idim, attention_dim)
            self.conv_subsampling_factor = 4
        elif input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(idim, attention_dim, padding_idx=padding_idx),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("encoder self-attention layer type = self-attention")
            encoder_selfattn_layer = MultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "legacy_rel_selfattn":
            assert pos_enc_layer_type == "legacy_rel_pos"
            encoder_selfattn_layer = LegacyRelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            logging.info("encoder self-attention layer type = relative self-attention")
            assert pos_enc_layer_type == "rel_pos"
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
            encoder_selfattn_layer_args = (
                attention_heads,
                attention_dim,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)

        # feed-forward module definition
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                dropout_rate,
                activation,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        # convolution module definition
        convolution_layer = ConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel, activation)

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                attention_dim,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                stochastic_depth_rate * float(1 + lnum) / num_blocks,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)

        self.intermediate_layers = intermediate_layers

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(self.embed, (Conv2dSubsampling, VGG2L, Conv2dSubsampling2,  Conv2dSubsampling3)):
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.intermediate_layers is None:
            xs, masks = self.encoders(xs, masks)
        else:
            intermediate_outputs = []
            for layer_idx, encoder_layer in enumerate(self.encoders):
                xs, masks = encoder_layer(xs, masks)

                if (
                    self.intermediate_layers is not None
                    and layer_idx + 1 in self.intermediate_layers
                ):
                    # intermediate branches also require normalization.
                    encoder_output = xs
                    if isinstance(encoder_output, tuple):
                        encoder_output = encoder_output[0]
                        if self.normalize_before:
                            encoder_output = self.after_norm(encoder_output)
                    intermediate_outputs.append(encoder_output)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks


class ConformerModel(pl.LightningModule):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lang_dir=None,
                 cfg=None):

        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.save_hyperparameters()



        self.encoder = Encoder(
            idim=input_dim,
            attention_dim=cfg['adim'],
            attention_heads=cfg['aheads'],
            linear_units=cfg['eunits'],
            num_blocks=cfg['elayers'],
            input_layer=cfg['transformer-input-layer'],
            dropout_rate=cfg['dropout-rate'],
            positional_dropout_rate=cfg['dropout-rate'],
            attention_dropout_rate=cfg['transformer-attn-dropout-rate'],
            pos_enc_layer_type=cfg['transformer-encoder-pos-enc-layer-type'],
            selfattention_layer_type=cfg['transformer-encoder-selfattn-layer-type'],
            activation_type=cfg['transformer-encoder-activation-type'],
            macaron_style=cfg['macaron-style'],
            use_cnn_module=cfg['use-cnn-module'],
            zero_triu=False if 'zero-triu' not in cfg.keys(
            ) else cfg['zero-triu'],
            cnn_module_kernel=cfg['cnn-module-kernel'],
            stochastic_depth_rate=0.0 if 'stochastic-depth-rate' not in cfg.keys() else cfg['stochastic-depth-rate'])

        # self.freese_and_init() # This is legacy of wav2vec 2.0

        self.output_layer = nn.Linear(cfg['adim'], self.output_dim)

        self.lang = Lang(lang_dir, load_topo=True,
                         load_lexicon=True, load_den_graph=True)

    def freese_and_init(self):
        # TODO
        pass

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):

        if self.cfg['loss'] in ['k2']:
            wavs = batch['feats']
            lengths = batch['feats_lens']
            word_ids = batch['word_ids']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=torch.cat([torch.zeros_like(log_probs[:, :, :1], dtype=log_probs.dtype), log_probs], dim=-1),
                                           supervision_segments=supervision_segments)

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device)

            assert decoding_graph.requires_grad == False

            if 'mask_inf' not in self.cfg.keys() or not self.cfg['mask_inf']:
                numerator = graph.graphloss(decoding_graph=decoding_graph,
                                            dense_fsa_vec=dense_fsa_vec,
                                            output_beam=self.cfg['output_beam'],
                                            reduction='sum')
            else:
                numerator = graph.graphloss(decoding_graph=decoding_graph,
                                            dense_fsa_vec=dense_fsa_vec,
                                            output_beam=self.cfg['output_beam'],
                                            reduction='none')
                inf_mask = torch.logical_not(torch.isinf(numerator))
                if False in inf_mask:
                    logging.warn(
                        'There are utterances whose inputs are shorter than labels..')
                numerator = torch.masked_select(numerator, inf_mask).sum()
            if 'no_den' in self.cfg.keys() and self.cfg['no_den']:
                loss = numerator
            else:
                if 'den_with_lexicon' in self.cfg.keys() and self.cfg['den_with_lexicon']:
                    den_decoding_graph = self.lang.den_graph.to(
                        log_probs.device)
                else:
                    den_decoding_graph = self.lang.topo.to(log_probs.device)

                assert den_decoding_graph.requires_grad == False

                if 'mask_inf' not in self.cfg.keys() or not self.cfg['mask_inf']:
                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  output_beam=self.cfg['output_beam'] if 'output_beam_den' not in self.cfg.keys(
                                                  ) else self.cfg['output_beam_den'],
                                                  reduction='sum')
                else:
                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  output_beam=self.cfg['output_beam'] if 'output_beam_den' not in self.cfg.keys(
                                                  ) else self.cfg['output_beam_den'],
                                                  reduction='none')
                    denominator = torch.masked_select(
                        denominator, inf_mask).sum()
                loss = numerator - denominator

        return loss/batch_num

    def forward(self, x, xlens):
        src_mask = make_non_pad_mask(xlens.tolist()).to(
            x.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(x, src_mask)
        x = self.output_layer(hs_pad)
        x = F.log_softmax(x, dim=-1)
        return x, torch.sum(hs_mask, dim=-1, dtype=torch.long)

    def debugging(self):
        norm_sum_4 = 0.
        norm_sum_3 = 0.
        norm_sum_2 = 0.
        norm_sum_1 = 0.
        norm_sum_lin = 0.
        for para in self.wav2vec.encoder.transformer.layers[-4].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_4 += torch.norm(para.grad).item()
        for para in self.wav2vec.encoder.transformer.layers[-3].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_3 += torch.norm(para.grad).item()
        for para in self.wav2vec.encoder.transformer.layers[-2].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_2 += torch.norm(para.grad).item()
        for para in self.wav2vec.encoder.transformer.layers[-1].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_1 += torch.norm(para.grad).item()
        for para in self.output_layer.parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_lin += torch.norm(para.grad).item()
        print('norm_sum_4', norm_sum_4)
        print('norm_sum_3', norm_sum_3)
        print('norm_sum_2', norm_sum_2)
        print('norm_sum_1', norm_sum_1)
        print('norm_sum_lin', norm_sum_lin)

        # print('torch.max(para.grad)', torch.max(para.grad))
        # print('torch.min(para.grad)', torch.min(para.grad))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self.compute_loss(batch, batch_idx, optimizer_idx)
        self.log('loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('valid_loss', loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        feats = batch['feats']
        feats_lens = batch['feats_lens']
        targets = batch['targets']
        names = batch['names']
        spks = batch['spks']
        texts = batch['texts']
        log_probs, xlens = self(feats, feats_lens)
        return log_probs, xlens, names, spks, texts

    def configure_optimizers(self):
        # optimiser = get_std_opt(
        # self.parameters(),
        # self.cfg['adim'],
        # self.cfg['transformer-warmup-steps'],
        # self.cfg['transformer-lr'],
        # )

        optimiser = torch.optim.Adam(
            self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        # return [optimiser], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, verbose=True, min_lr=1e-8),
                              # 'monitor': 'valid_loss'}]
        return optimiser

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)

        for pg in optimizer.param_groups:
            lr = pg["lr"]
            break
        self.log('lr', lr, sync_dist=True)

        # if (self.trainer.global_step + 1) <= self.cfg['transformer-warmup-steps']:
        # lr = (
            # self.cfg['transformer-lr']
            # * self.cfg['adim'] ** (-0.5)
            # * min((self.trainer.global_step+1) ** (-0.5), (self.trainer.global_step+1) * self.cfg['transformer-warmup-steps'] ** (-1.5))
        # )

        lr = (
            self.cfg['transformer-lr']
            * self.cfg['adim'] ** (-0.5)
            * min(self.cfg['transformer-warmup-steps'] ** (-0.5), (self.trainer.global_step+1) * self.cfg['transformer-warmup-steps'] ** (-1.5))
        )

        for pg in optimizer.param_groups:
            pg["lr"] = lr

def get_model(input_dim, output_dim):
    model = ConformerModel(input_dim, output_dim)
    return model


def proto_model(input_dim, output_dim):
    model = ConformerModel(input_dim, output_dim)
    return model
