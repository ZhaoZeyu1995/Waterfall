#!/usr/bin/env python3
import openfst_python as fst
import numpy as np
import kaldiio
import argparse


'''
This is a prototype of the T+LG decoder.
There are some assumptions in this decoder:
1. The input is a 2D numpy array, which is the log probability of each token in each frame.
2. There is no transition with epsilon input label in the fst LG or the fst T.
'''


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


class Token(object):
    '''
    The class to store the token information

    Attributes:
    cost: the cost of the token
    prev_token: the previous token
    arc: the arc of the token
    topo_states: the topology states of the token
    '''

    def __init__(self, cost: float, prev_token: Token, arc: Arc, topo_states: list):
        self.cost = cost
        self.prev_token = prev_token
        self.arc = arc
        self.topo_states = topo_states


# The definition of the decoder Class
class Decoder(object):
    '''
    The class to decode the input

    Attributes:
    fst_path: the path of the fst LG
    topo_fst_path: the path of the topology fst T
    word_table: the path of the word table
    acoustic_scale: the acoustic scale

    Methods:
    decode: the method to decode the input
    '''

    # The constructor of the class
    def __init__(self,
                 fst_path: str,
                 topo_fst_path: str,
                 word_table: str,
                 acoustic_scale: float):

        self.fst_path = fst_path
        self.topo_fst_path = topo_fst_path
        self.fst = fst.Fst.read(self.fst_path)
        self.topo_fst = fst.Fst.read(self.topo_fst_path)
        self.word_table = fst.SymbolTable.read_text(word_table)
        self.acoustic_scale = acoustic_scale

        self.check_fst()

        # Add epsilon self-loop to each state in self.fst
        for state in self.fst.states():
            state.add_arc(fst.Arc(0, 0, 0.0, state.state_id))

    def check_fst(self):
        # Check the self.fst has no transition with epsilon input label
        for state in self.fst.states():
            for arc in state.arcs():
                if arc.ilabel == 0:
                    raise ValueError(
                        "The input label of the transition is epsilon in the fst")

        # Check the self.topo_fst has no transition with epsilon input label
        for state in self.topo_fst.states():
            for arc in state.arcs():
                if arc.ilabel == 0:
                    raise ValueError(
                        "The input label of the transition is epsilon in the topo fst")

    # The method to decode the input
    def decode(self, log_probs: np.ndarray):
        self.init_decoder()
        self.log_probs = - self.acoustic_scale * log_probs
        self.num_frames = self.log_probs.shape[0]
        self.best_cost = float('inf')
        for frame in range(self.num_frames):
            self.process_frame(frame)
        self.finalize_decoding()
        # return the best word sequence
        return self.get_best_word_seq()

    def init_decoder(self):
        self.tokens = dict()
        topo_states = []
        topo_states.append(self.topo_fst.start())
        self.tokens[self.fst.start()] = Token(0, None, None, topo_states)

    def process_frame(self, frame: int):
        new_tokens = dict()
        for state in self.tokens.keys():
            for arc in state.arcs():
                if arc.ilabel != 0:
                    self.process_emission(state, frame, new_tokens, arc)
                else:
                    # This is a non-emission transition self loop
                    self.process_non_emission(state, frame, new_tokens, arc)
        self.tokens = new_tokens

    def process_emission(self, state: int, frame: int, new_tokens: dict, arc: fst.Arc):
        ac_cost, topo_states = self.calculate_acoustic_cost(
            arc.ilabel, self.tokens[state].topo_states, frame)
        cost = self.tokens[state].cost + float(arc.weight) + ac_cost

        if cost < self.best_cost:
            self.best_cost = cost

        simple_arc = Arc(arc.ilabel, arc.olabel, arc.weight, arc.nextstate)
        if arc.nextstate not in new_tokens.keys():
            new_tokens[arc.nextstate] = Token(
                cost, self.tokens[state], simple_arc, topo_states)
        else:
            if cost < new_tokens[arc.nextstate].cost:
                new_tokens[arc.nextstate] = Token(
                    cost, self.tokens[state], simple_arc, topo_states)

    def process_non_emission(self, state: int, frame: int, new_tokens: dict, arc: fst.Arc):
        ac_cost, topo_states = self.calculate_acoustic_cost(
            arc.ilabel, self.tokens[state].topo_states, frame)
        cost = self.tokens[state].cost + ac_cost

        if cost < self.best_cost:
            self.best_cost = cost

        simple_arc = Arc(arc.ilabel, arc.olabel, arc.weight, arc.nextstate)
        if arc.nextstate not in new_tokens.keys():
            new_tokens[arc.nextstate] = Token(
                cost, self.tokens[state], simple_arc, topo_states)
        else:
            if cost < new_tokens[arc.nextstate].cost:
                new_tokens[arc.nextstate] = Token(
                    cost, self.tokens[state], simple_arc, topo_states)

    def calculate_acoustic_cost(self, phone_id: int, topo_states: list, frame: int):
        ac_cost = 0
        new_topo_states = []
        for topo_state in topo_states:
            for arc in self.topo_fst.arc(topo_state):
                if arc.olabel == phone_id:
                    # update ac_cost in log probability domain
                    ac_cost = np.logaddexp(
                        ac_cost, self.log_probs[frame][arc.ilabel])
                    new_topo_states.append(arc.nextstate)
        return ac_cost, new_topo_states

    def finalize_decoding(self):
        self.best_token = None
        for state in self.tokens.keys():
            # check if state is a final state of the self.fst
            final_weight = self.fst.final(state)
            self.tokens[state].cost += float(final_weight)
            if not final_weight == fst.Weight.zero(self.fst.weight_type()):
                continue

            if self.best_token is None:
                self.best_token = self.tokens[state]
            else:
                if self.tokens[state].cost < self.best_token.cost:
                    self.best_token = self.tokens[state]

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
            print(key, path)


# The main function
if __name__ == '__main__':
    # Arguments parser for testing with a scp file
    parser = argparse.ArgumentParser()
    parser.add_argument('--fst_path', type=str, default='LG.fst')
    parser.add_argument('--topo_fst_path', type=str, default='T.fst')
    parser.add_argument('--word_table', type=str, default='words.txt')
    parser.add_argument('--acoustic_scale', type=float, default=1.0)
    parser.add_argument('--scp_path', type=str, default='test.scp')
    args = parser.parse_args()

    # Create the decoder object with the given arguments
    decoder = Decoder(args.fst_path, args.topo_fst_path, args.word_table, args.acoustic_scale)

    # Decode the scp file
    decode_scp(args.scp_path, decoder)



