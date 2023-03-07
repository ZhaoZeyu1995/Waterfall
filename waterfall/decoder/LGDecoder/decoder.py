#!/usr/bin/env python3
import openfst_python as fst
import numpy as np
import kaldiio
import argparse
import logging


'''
This is a prototype of the T+LG decoder.
There are some assumptions in this decoder:
1. The input is a 2D numpy array, which is the log probability of each token in each frame.
2. This decoder allows there are epsilons as the input labels in the fst LG.
3. The epsilon is still numbered as 0 in the fst LG for both input and output SymbolTables.
4. However, the epsilon is not numbered as 0 in the fst T and there is no epsilon in the SymbolTable of the fst T.
5. For each LG state, there is at most one token associated with it, and all the tokens are stored in a dictionary.
'''


class TopoToken(object):
    '''
    The class to store the topology token information

    Attributes:
    ac_cost: the acoustic cost of the token
    prev_token: the previous token
    arc: the arc of the token
    '''

    def __init__(self, ac_cost: float, prev_token, arc: fst.Arc):
        self.ac_cost = ac_cost
        self.prev_token = prev_token
        self.arc = arc

    @property
    def cost(self):
        if self.prev_token is None:
            return self.ac_cost
        else:
            return self.prev_token.cost + self.ac_cost


class Arc(object):
    '''
    The class to store the arc information

    Attributes:
    ilabel: the input label of the arc
    olabel: the output label of the arc
    weight: the weight of the arc
    nextstate: the next state of the arc
    '''

    def __init__(self,
                 ilabel: int,
                 olabel: int,
                 weight: float,
                 nextstate: int):
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight
        self.nextstate = nextstate


def is_final_state(input_fst, state):
    # Check is a state is a final state of a fst
    if input_fst.final(state) == fst.Weight.Zero(input_fst.weight_type()):
        return False
    else:
        return True


class Token(object):
    '''
    The class to store the token information

    Attributes:
    prev_token: the previous token
    arc: the arc of the token
    topo_tokens: the list of topology tokens
    topo_fst: the topology fst T
    '''

    def __init__(self, prev_token, arc: fst.Arc, topo_tokens: list, topo_fst: fst.Fst):
        self.prev_token = prev_token
        self.arc = arc
        self.topo_tokens = topo_tokens
        self.topo_fst = topo_fst

    @property
    def ac_cost(self):
        '''
        The method to calculate the acoustic cost of the token
        which is accumulated and recorded by topo_tokens
        '''
        if len(self.topo_tokens) == 0:
            return float('inf')
        ac_cost = None
        for topo_token in self.topo_tokens:
            if ac_cost is None:
                ac_cost = topo_token.cost
            else:
                # The ac_cost of the token is the logsumexp of the ac_cost of the topo_tokens
                ac_cost = -np.logaddexp(-ac_cost, -topo_token.cost)
        return ac_cost

    @property
    def finished_ac_cost(self):
        '''
        The method to calculate the finished acoustic cost of the token
        which is accumulated and recorded by topo_tokens but only takes the final topo_states into account

        Note this method is not applied in the current version of the decoder as I feel a partial path in the token fst will not hurt that much.
        '''
        finished_ac_cost = None
        for topo_token in self.topo_tokens:
            if finished_ac_cost is None:
                finished_ac_cost = topo_token.cost
            else:
                if is_final_state(self.topo_fst, topo_token.arc.nextstate):
                    # The finished_ac_cost of the token is the logsumexp of the finished_ac_cost of the topo_tokens
                    finished_ac_cost = -np.logaddexp(
                        -finished_ac_cost, -topo_token.cost)
        return finished_ac_cost

    @property
    def graph_cost(self):
        '''
        The method to calculate the graph cost of the token
        which is accumulated and recorded by tokens but not topo_tokens,
        while the latter is designed to record the acoustic cost only
        '''
        if self.prev_token is None:
            if self.arc is None:
                return 0.
            else:
                return float(self.arc.weight)
        else:
            return self.prev_token.graph_cost + float(self.arc.weight)

    @property
    def cost(self):
        '''
        The method to calculate the cost of the token
        which is the sum of the graph cost and the acoustic cost
        '''
        return self.graph_cost + self.ac_cost

    @property
    def finished_cost(self):
        '''
        The method to calculate the finished cost of the token
        which is the sum of the graph cost and the finished acoustic cost

        Note this method is not applied in the current version of the decoder as I feel a partial path in the token fst will not hurt that much.
        '''
        return self.graph_cost + self.finished_ac_cost


# The definition of the decoder Class
class Decoder(object):
    '''
    The class to decode the input

    Attributes:
    fst_path: the path of the fst LG
    topo_fst_path: the path of the topology fst T
    word_table: the path of the word table
    acoustic_scale: the acoustic scale
    beam: the beam of the decoder in LG
    max_active: the max_active of the decoder in LG
    topo_beam: the beam of the topology decoder in T for each LG token
    topo_max_active: the max_active of the topology decoder in T for each LG token

    Methods:
    decode: the method to decode the input
    '''

    # The constructor of the class
    def __init__(self,
                 fst_path: str,
                 topo_fst_path: str,
                 acoustic_scale: float,
                 beam: float,
                 max_active: int,
                 topo_beam: float,
                 topo_max_active: int,
                 word_table=None,
                 ):
        self.fst_path = fst_path
        self.topo_fst_path = topo_fst_path
        logging.info('Reading the fst from {}'.format(self.fst_path))
        self.fst = fst.Fst.read(self.fst_path)
        logging.info('Removing epsilon transitions from the fst')
        self.fst.rmepsilon()
        logging.info('Reading the topology fst from {}'.format(self.topo_fst_path))
        self.topo_fst = fst.Fst.read(self.topo_fst_path)
        if word_table is not None:
            logging.info('Reading the word table from {}'.format(word_table))
            self.word_table = fst.SymbolTable.read_text(word_table)

        self.acoustic_scale = acoustic_scale
        self.beam = beam
        self.max_active = max_active
        self.topo_beam = topo_beam
        self.topo_max_active = topo_max_active

        # Add epsilon self-loop to each state in self.fst
        logging.info('Adding epsilon self-loop to each state in the fst')
        for state in self.fst.states():
            self.fst.add_arc(state, fst.Arc(
                0, 0, fst.Weight(self.fst.weight_type(), 0.), state))

    # The method to decode the input

    def decode(self, log_probs: np.ndarray):
        self.init_decoder()
        self.log_probs = - self.acoustic_scale * log_probs
        self.num_frames = self.log_probs.shape[0]
        for frame in range(self.num_frames):
            logging.info('Processing frame {}'.format(frame))
            logging.info('The number of tokens in the current frame is {}'.format(
                len(self.tokens)))
            for idx, (state, token) in enumerate(self.tokens.items()):
                logging.info('The number of topo_tokens in the %d-th token at state %d is %d' %
                      (idx, int(state), len(token.topo_tokens)))
            self.best_cost = float('inf')
            self.process_frame(frame)
            self.process_non_emitting()
            logging.info('The number of tokens in the new frame is {}'.format(
                len(self.new_tokens)))
            self.prune()
            self.tokens = self.new_tokens
            self.new_tokens = dict()
        self.finalize_decoding()
        # return the best word sequence
        return self.get_best_word_seq()

    def init_decoder(self):
        self.tokens = dict()
        topo_tokens = [TopoToken(0., prev_token=None, arc=Arc(
            0, 0, 0.0, self.topo_fst.start()))]
        # The list to record the tokens in the current frame
        self.tokens[self.fst.start()] = Token(
            None, Arc(0, 0, 0.0, self.fst.start()), topo_tokens, self.topo_fst)
        # The set to record the new tokens in the current frame
        self.new_tokens = self.tokens

        self.best_cost = float('inf')
        self.process_non_emitting()

        self.tokens = self.new_tokens
        self.new_tokens = dict()

    def process_frame(self, frame: int):
        for state in self.tokens.keys():
            token = self.tokens[state]
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0 and arc.olabel != 0:  # non-emitting transition
                    continue
                self.forward(token, arc, frame)

    def prune(self):
        # Prune the tokens in the current frame
        new_tokens = []
        for state, token in self.new_tokens.items():
            if token.cost < self.best_cost + self.beam:
                new_tokens.append(token)
            else:
                delete_token(token)
        if len(new_tokens) > self.max_active:
            new_tokens.sort(key=lambda x: x.cost)
            new_tokens = new_tokens[:self.max_active]
        self.new_tokens = {token.arc.nextstate: token for token in new_tokens}

    def forward(self, token: Token, arc: fst.Arc, frame: int):
        '''
        The method to forward the token from a state in LG
        '''
        new_topo_tokens = self.forward_topo_tokens(
            arc.ilabel, token.topo_tokens, frame)

        if len(new_topo_tokens) == 0:
            return
        new_token = Token(token,
                          arc,
                          new_topo_tokens,
                          self.topo_fst)

        if new_token.cost < self.best_cost:
            self.best_cost = new_token.cost

        if new_token.cost < self.best_cost + self.beam:
            if arc.nextstate not in self.new_tokens:
                self.new_tokens[arc.nextstate] = new_token
            else:
                if new_token.cost < self.new_tokens[arc.nextstate].cost:
                    delete_token(self.new_tokens[arc.nextstate])
                    self.new_tokens[arc.nextstate] = new_token
                else:
                    delete_token(new_token)
        else:
            delete_token(new_token)


    def process_non_emitting(self):
        '''
        The method to forward the non-emitting tokens

        Args:
        best_cost: the best cost of the tokens in the current frame

        '''
        states = list(self.new_tokens.keys())

        while len(states) > 0:
            state = states.pop(0)
            token = self.new_tokens[state]
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0 and arc.olabel != 0:
                    '''
                    non-emitting transition but emitting output, which is different from epsilon self-loop. 
                    As we remove the epsilon transitions, whose input and output labels are both epsilon, at the initial stage, 
                    now the epsilon transitions are the ones we added manually to deal with epsilon output by the token fst.
                    '''
                    new_token = Token(
                        token, arc, token.topo_tokens, self.topo_fst)
                    if new_token.cost < self.best_cost + self.beam:
                        if arc.nextstate not in self.new_tokens:
                            self.new_tokens[arc.nextstate] = new_token
                        else:
                            if new_token.cost < self.new_tokens[arc.nextstate].cost:
                                delete_token(self.new_tokens[arc.nextstate])
                                self.new_tokens[arc.nextstate] = new_token
                                states.append(arc.nextstate)
                            else:
                                delete_token(new_token)
                    else:
                        delete_token(new_token)

    def forward_topo_tokens(self, phone_id: int, topo_tokens: list, frame: int):
        '''
        The method to forward the topology tokens
        '''
        new_topo_tokens = []
        best_cost = float('inf')
        for topo_token in topo_tokens:
            for arc in self.topo_fst.arcs(topo_token.arc.nextstate):
                if arc.olabel == phone_id:
                    ac_cost = self.log_probs[frame, arc.ilabel]
                    new_topo_token = TopoToken(ac_cost, topo_token, arc)
                    if new_topo_token.cost < best_cost:
                        best_cost = new_topo_token.cost
                    if new_topo_token.cost < best_cost + self.topo_beam:
                        new_topo_tokens.append(new_topo_token)
                    else:
                        delete_token(new_topo_token)
        return self.prune_topo_tokens(new_topo_tokens, best_cost)

    def prune_topo_tokens(self, topo_tokens: list, best_cost: float):
        '''
        The method to prune the topology tokens
        '''
        new_topo_tokens = []
        for topo_token in topo_tokens:
            if topo_token.cost < best_cost + self.topo_beam:
                new_topo_tokens.append(topo_token)
            else:
                delete_token(topo_token)

        if len(new_topo_tokens) > self.topo_max_active:
            new_topo_tokens.sort(key=lambda token: token.cost)
            new_topo_tokens = new_topo_tokens[:self.topo_max_active]
        return new_topo_tokens

    def finalize_decoding(self):
        self.best_token = None
        self.best_final_cost = None
        for state, token in self.tokens.items():
            # continue if the state is not final
            if not is_final_state(self.fst, state):
                continue

            final_weight = float(self.fst.final(state))
            final_cost = token.cost + final_weight
            if self.best_token is None:
                self.best_token = token
                self.best_final_cost = final_cost
            else:
                if final_cost < self.best_final_cost:
                    self.best_token = token
                    self.best_final_cost = final_cost

    # The method to get the best path and the corresponding words
    def get_best_path(self):
        path = []
        token = self.best_token
        while token is not None:
            if token.arc is not None:
                path.append(token.arc)
            token = token.prev_token
        path.reverse()
        return path

    # Get the words from the best path
    def get_best_word_seq(self):
        best_path = self.get_best_path()
        words = []
        for arc in best_path:
            if arc.olabel != 0:
                words.append(self.word_table.find(arc.olabel))
        return words

    # Get the word-ids from the best path
    def get_best_word_id_seq(self):
        best_path = self.get_best_path()
        word_ids = []
        for arc in best_path:
            if arc.olabel != 0:
                word_ids.append(arc.olabel)
        return word_ids

# Delete a token and all its previous tokens


def delete_token(token: Token):
    if token.prev_token is not None:
        delete_token(token.prev_token)
    del token

# read a scp file with kaldiio and use the decoder to decode each input feature


def decode_scp(scp_path: str, decoder: Decoder):
    with kaldiio.ReadHelper(f'scp:{scp_path}') as reader:
        for key, value in reader:
            log_probs = value
            path = decoder.decode(log_probs)
            logging.info(key, path)
            logging.info("Length of the path: ", len(path))
            break


# The main function
if __name__ == '__main__':
    # Arguments parser for testing with a scp file
    parser = argparse.ArgumentParser()
    parser.add_argument('--fst_path', type=str, help='Path to the decoding graph',
                        default='../../wsj/asr_bpe/data/lang_bpe_100_eval_test_bd_fg_ctc/LG.fst')
    parser.add_argument('--topo_fst_path', type=str, help='Path to the topology graph',
                        default='../../wsj/asr_bpe/data/lang_bpe_100_eval_test_bd_fg_ctc/k2/T.fst')
    parser.add_argument('--word_table', type=str, help='Path to the word table',
                        default='../../wsj/asr_bpe/data/lang_bpe_100_eval_test_bd_fg_ctc/words.txt')
    parser.add_argument('--scp_path', type=str, help='Path to the scp file',
                        default='../../wsj/asr_bpe/exp/ctc-transformer-100/predict_test_eval92/output.1.scp')
    parser.add_argument('--acoustic_scale', type=float,
                        default=0.8, help='Acoustic scale')
    parser.add_argument('--beam', type=float, default=16.0,
                        help='Beam in LG.fst')
    parser.add_argument('--max_active', type=int, default=20,
                        help='Max active states in LG.fst')
    parser.add_argument('--topo_beam', type=float, default=16.0,
                        help='Beam in T.fst in each LG token')
    parser.add_argument('--topo_max_active', type=int, default=100,
                        help='Max active states in T.fst in each LG token')

    args = parser.parse_args()

    # Create the decoder object with the given arguments
    decoder = Decoder(args.fst_path,
                      args.topo_fst_path,
                      args.word_table,
                      args.acoustic_scale,
                      args.beam,
                      args.max_active,
                      args.topo_beam,
                      args.topo_max_active
                      )

    # Decode the scp file
    decode_scp(args.scp_path, decoder)
