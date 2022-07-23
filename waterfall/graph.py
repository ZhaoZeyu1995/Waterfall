import torch
from torch import nn
import k2
import os

# inspired by icefall, modified by Zeyu Zhao (University of Edinburgh 2022)


class GraphLoss(nn.Module):
    '''Graph Loss computation in k2. 
    '''

    def __init__(self,
                 output_beam: float,
                 reduction: str = 'sum',
                 use_double_scores: bool = False):
        '''
        Args:
          output_beam:
             Beam to prune output, similar to lattice-beam in Kaldi.  Relative
             to best path of output.
          reduction:
            Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'.
            'none': no reduction will be applied, 'mean': the output losses
            will be **divided** by the target lengths and then the **mean** over
            the batch is taken. 'sum': sum the output losses over batches.
          use_double_scores:
            True to use double precision floating point in computing
            the total scores. False to use single precision.
        '''
        super().__init__()
        assert reduction in ('none', 'mean', 'sum')
        self.output_beam = output_beam
        self.reduction = reduction
        self.use_double_scores = use_double_scores

    def forward(self,
                decoding_graph: k2.Fsa,
                dense_fsa_vec: k2.DenseFsaVec,
                target_lengths: torch.Tensor = None,
                max_states: int = 15000000) -> torch.Tensor:
        '''Compute the Graph loss given a decoding graph and a dense fsa vector.

        Args:
          decoding_graph:
            An FsaVec. It can be the composition result of a Graph topology
            and a transcript.
          dense_fsa_vec:
            It represents the neural network output. Refer to the help
            information in :class:`k2.DenseFsaVec`.
          target_lengths:
            Used only when `reduction` is `mean`. It is a 1-D tensor of batch
            size representing lengths of the targets, e.g., number of phones or
            number of word pieces in a sentence.
        Returns:
          If `reduction` is `none`, return a 1-D tensor with size equal to batch
          size. If `reduction` is `mean` or `sum`, return a scalar.
        '''
        lattice = k2.intersect_dense(decoding_graph, dense_fsa_vec,
                                     self.output_beam)

        tot_scores = lattice.get_tot_scores(
            log_semiring=True, use_double_scores=self.use_double_scores)
        loss = -1 * tot_scores
        loss = loss.to(torch.float32)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            assert self.reduction == 'mean'
            loss /= target_lengths
            return loss.mean()


def graphloss(decoding_graph: k2.Fsa,
              dense_fsa_vec: k2.DenseFsaVec,
              output_beam: float = 10,
              reduction: str = 'sum',
              use_double_scores: bool = False,
              max_states: int = 15000000,
              target_lengths: torch.Tensor = None) -> torch.Tensor:
    '''Compute the Graph loss given a decoding graph and a dense fsa vector.

    Args:
      decoding_graph:
        An FsaVec. It can be the composition result of a token topology
        and a transcript.
      dense_fsa_vec:
        It represents the neural network output. Refer to the help information
        in :class:`k2.DenseFsaVec`.
      output_beam:
         Beam to prune output, similar to lattice-beam in Kaldi.  Relative
         to best path of output.
      reduction:
        Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        'none': no reduction will be applied, 'mean': the output losses will be
        divided by the target lengths and then the mean over the batch is taken.
        'sum': sum the output losses over batches.
      use_double_scores:
        True to use double precision floating point in computing
        the total scores. False to use single precision.
      target_lengths:
        Used only when `reduction` is `mean`. It is a 1-D tensor of batch
        size representing lengths of the targets, e.g., number of phones or
        number of word pieces in a sentence.
    Returns:
      If `reduction` is `none`, return a 1-D tensor with size equal to batch
      size. If `reduction` is `mean` or `sum`, return a scalar.
    '''
    m = GraphLoss(output_beam=output_beam,
                reduction=reduction,
                use_double_scores=use_double_scores)

    return m(decoding_graph, dense_fsa_vec, target_lengths, max_states=max_states)

def process_openfst(file):
    '''
    There is a small bug about k2.Fsa.from_openfst(openfst_str), 
    as I found when openfst_str contains a line without the weight at the end, 
    an error will occur. 

    This programme also bridge the gap between openfst and k2.

    args:
    file: str, the directory of an openfst binary file 
    return:
    a str that can be read by k2.Fsa.from_openfst()
    '''
    lines = os.popen('fstprint %s' % (file)).read().split('\n')
    for idx, line in enumerate(lines):
        lc = line.split('\t')
        if len(lc) == 4:
            lines[idx] = '\t'.join(lc+['0.0\n'])
    return '\n'.join(lines)


