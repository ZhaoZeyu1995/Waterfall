import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
import torchaudio
from waterfall import graph
import k2
from waterfall.utils.datapipe import Lang

# from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask

import logging

import torch

from espnet.nets.pytorch_backend.conformer.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.conformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet.nets.pytorch_backend.transducer.vgg2l import VGG2L
from espnet.nets.pytorch_backend.transformer.attention import (
    LegacyRelPositionMultiHeadedAttention,
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from espnet.nets.pytorch_backend.transformer.embedding import (
    LegacyRelPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    ScaledPositionalEncoding,
)
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import (
    Conv1dLinear,
    MultiLayeredConv1d,
)
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6, Conv2dSubsampling8


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
        ctc_softmax=None,
        conditioning_layer_dim=None,
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
        self.batch_norm = None
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
            self.batch_norm = torch.nn.BatchNorm1d(attention_dim)
        elif input_layer == "conv2d2":
            self.embed = Conv2dSubsampling2(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 2
            self.batch_norm = torch.nn.BatchNorm1d(attention_dim)
        elif input_layer == "conv2d6":
            self.embed = Conv2dSubsampling6(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 6
            self.batch_norm = torch.nn.BatchNorm1d(attention_dim)
        elif input_layer == "conv2d8":
            self.embed = Conv2dSubsampling8(
                idim,
                attention_dim,
                dropout_rate,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
            self.conv_subsampling_factor = 8
            self.batch_norm = torch.nn.BatchNorm1d(attention_dim)
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
        self.use_conditioning = True if ctc_softmax is not None else False
        if self.use_conditioning:
            self.ctc_softmax = ctc_softmax
            self.conditioning_layer = torch.nn.Linear(
                conditioning_layer_dim, attention_dim
            )

    def forward(self, xs, masks):
        """Encode input sequence.

        Args:
            xs (torch.Tensor): Input tensor (#batch, time, idim).
            masks (torch.Tensor): Mask tensor (#batch, time).

        Returns:
            torch.Tensor: Output tensor (#batch, time, attention_dim).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(self.embed, (Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6, Conv2dSubsampling8, VGG2L)):
            print('self.embed', self.embed)
            xs, masks = self.embed(xs, masks)
        else:
            xs = self.embed(xs)

        if self.batch_norm is not None:
            xs = (self.batch_norm(xs[0].permute(0, 2, 1)).permute(0, 2, 1), xs[1])

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

                    if self.use_conditioning:
                        intermediate_result = self.ctc_softmax(encoder_output)

                        if isinstance(xs, tuple):
                            x, pos_emb = xs[0], xs[1]
                            x = x + self.conditioning_layer(intermediate_result)
                            xs = (x, pos_emb)
                        else:
                            xs = xs + self.conditioning_layer(intermediate_result)

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.normalize_before:
            xs = self.after_norm(xs)

        if self.intermediate_layers is not None:
            return xs, masks, intermediate_outputs
        return xs, masks

class ConformerModelNoWarmup(pl.LightningModule):
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

        self.batch_norm = nn.BatchNorm1d(cfg['adim'])
        self.output_layer = nn.Linear(cfg['adim'], self.output_dim)

        if self.cfg['loss'] == 'builtin_ctc':
            self.lang = Lang(lang_dir)
        elif self.cfg['loss'] == 'k2':
            self.lang = Lang(lang_dir, load_topo=True, load_lexicon=True)

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):

        if self.cfg['loss'] in ['k2']:
            wavs = batch['feats']
            lengths = batch['feats_lens']
            word_ids = batch['word_ids']
            target_lengths = batch['target_lengths']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=log_probs,
                                           supervision_segments=supervision_segments)

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device)

            assert decoding_graph.requires_grad == False

            numerator = graph.graphloss(decoding_graph=decoding_graph,
                                        dense_fsa_vec=dense_fsa_vec,
                                        target_lengths=target_lengths,
                                        reduction='mean')

            if 'no_den' in self.cfg.keys() and self.cfg['no_den']:
                loss = numerator
            elif 'no_den_grad' in self.cfg.keys() and self.cfg['no_den_grad']:
                with torch.no_grad():
                    den_decoding_graph = k2.create_fsa_vec(
                        [self.lang.topo.to(log_probs.device) for _ in range(batch_num)])

                    assert den_decoding_graph.requires_grad == False

                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  target_lengths=target_lengths,
                                                  reduction='mean')
                loss = numerator - denominator
            else:

                den_decoding_graph = k2.create_fsa_vec(
                    [self.lang.topo.to(log_probs.device) for _ in range(batch_num)])

                assert den_decoding_graph.requires_grad == False

                denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                              dense_fsa_vec=dense_fsa_vec,
                                              target_lengths=target_lengths,
                                              reduction='mean')
                loss = numerator - denominator
        elif self.cfg['loss'] == 'builtin_ctc':
            wavs = batch['feats']
            lengths = batch['feats_lens']
            target_lengths = batch['target_lengths']
            targets_ctc = batch['targets_ctc']
            log_probs, xlens = self(wavs, lengths)
            loss = F.ctc_loss(log_probs.permute(1, 0, 2), targets_ctc, xlens,
                              target_lengths, reduction='mean')

        else:
            raise Exception('Unrecognised Loss Function %s' %
                            (self.cfg['loss']))

        return loss

    def forward(self, x, xlens):
        src_mask = make_non_pad_mask(xlens.tolist()).to(
            x.device).unsqueeze(-2)
        x, x_mask = self.encoder(x, src_mask)
        x = self.batch_norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, torch.sum(x_mask, dim=-1, dtype=torch.long).unsqueeze(-1)

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
        optimiser = torch.optim.Adam(
            self.parameters(), lr=float(self.cfg['lr']))
        return [optimiser], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                                                      'min',
                                                                                      patience=self.cfg['lr_patience'],
                                                                                      verbose=True,
                                                                                      factor=self.cfg['factor'],
                                                                                      min_lr=float(self.cfg['min_lr'])),
                              'monitor': 'valid_loss'}]


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

        
        self.batch_norm = nn.BatchNorm1d(cfg['adim'])
        self.output_layer = nn.Linear(cfg['adim'], self.output_dim)

        if self.cfg['loss'] == 'builtin_ctc':
            self.lang = Lang(lang_dir)
        elif self.cfg['loss'] == 'k2':
            self.lang = Lang(lang_dir, load_topo=True, load_lexicon=True)

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):

        if self.cfg['loss'] in ['k2']:
            wavs = batch['feats']
            lengths = batch['feats_lens']
            word_ids = batch['word_ids']
            target_lengths = batch['target_lengths']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=log_probs,
                                           supervision_segments=supervision_segments)

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device)

            assert decoding_graph.requires_grad == False

            numerator = graph.graphloss(decoding_graph=decoding_graph,
                                        dense_fsa_vec=dense_fsa_vec,
                                        target_lengths=target_lengths,
                                        reduction='mean')

            if 'no_den' in self.cfg.keys() and self.cfg['no_den']:
                loss = numerator
            elif 'no_den_grad' in self.cfg.keys() and self.cfg['no_den_grad']:
                with torch.no_grad():
                    den_decoding_graph = k2.create_fsa_vec(
                        [self.lang.topo.to(log_probs.device) for _ in range(batch_num)])

                    assert den_decoding_graph.requires_grad == False

                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  target_lengths=target_lengths,
                                                  reduction='mean')
                loss = numerator - denominator
            else:

                den_decoding_graph = k2.create_fsa_vec(
                    [self.lang.topo.to(log_probs.device) for _ in range(batch_num)])

                assert den_decoding_graph.requires_grad == False

                denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                              dense_fsa_vec=dense_fsa_vec,
                                              target_lengths=target_lengths,
                                              reduction='mean')
                loss = numerator - denominator
        elif self.cfg['loss'] == 'builtin_ctc':
            wavs = batch['feats']
            lengths = batch['feats_lens']
            target_lengths = batch['target_lengths']
            targets_ctc = batch['targets_ctc']
            log_probs, xlens = self(wavs, lengths)
            loss = F.ctc_loss(log_probs.permute(1, 0, 2), targets_ctc, xlens,
                              target_lengths, reduction='mean')

        else:
            raise Exception('Unrecognised Loss Function %s' %
                            (self.cfg['loss']))

        return loss

    def forward(self, x, xlens):
        src_mask = make_non_pad_mask(xlens.tolist()).to(
            x.device).unsqueeze(-2)
        x, x_mask = self.encoder(x, src_mask)
        x = self.batch_norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, torch.sum(x_mask, dim=-1, dtype=torch.long).unsqueeze(-1)

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
        optimiser = torch.optim.Adam(
            self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
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
        lr = (
            float(self.cfg['final_lr'])
            * min((self.trainer.global_step+1) ** (-0.5) * self.cfg['transformer-warmup-steps'] ** (0.5),
                  (self.trainer.global_step+1) * self.cfg['transformer-warmup-steps'] ** (-1))
        )

        for pg in optimizer.param_groups:
            pg["lr"] = lr
