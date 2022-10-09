"""WFST decoder for Seq2Seq model

Python version decoder which treat Seq2Seq models(attention, transformer, CTC)
as acoustic model. Record inner states of Seq2Seq models in token and execute
token passing algorithm in WFST decoding graph.
"""
import sys
import numpy as np
import openfst_python as fst


class LatticeArc:
    """Arc used in Token
    acoustic cost and graph cost are stored separately.
    ilabel, int
    olabel, int
    weight, graph_cost, float, in -log domain, tropical semiring only, for simplicity reason, we only implement in tropical semiring
    nextstate, fst_state
    """

    def __init__(self, ilabel, olabel, weight, nextstate):
        self.ilabel = ilabel
        self.olabel = olabel
        self.weight = weight
        self.nextstate = nextstate


class Token:
    """Token used in token passing decode algorithm.
    The token can be linked by another token or None.
    The token record the linked arc, cost and inner packed states
    used in model
    """

    def __init__(self, arc, acoustic_cost, prev_tok=None):
        self.prev_tok = prev_tok
        self.arc = LatticeArc(arc.ilabel, arc.olabel, arc.weight, arc.nextstate)
        if prev_tok is not None:
            self.cost = prev_tok.cost + float(arc.weight) + acoustic_cost
        else:
            self.cost = float(arc.weight) + acoustic_cost
        self.rescaled_cost = -1.0


def delete_token(token: Token):
    '''
    Delete a token and its history recursively
    May not be very useful.
    '''
    prev = token.prev
    while token:
        del token
        token = prev


class WFSTDecoder:
    """A Viterbi decoder, which resembles the decode-faster in Kaldi
    It takes a decoding graph, a log-likelihood matrix (T, N) as inputs and outputs the best word sequence.
    Actually, it takes the word sequence, one of whose alignments is the best compared to other possible alignments.
    As this is only a prototype, it does not support generating lattice, but the best word sequences only.
    """

    def __init__(self,
                 fst_path,
                 acoustic_scale=0.1,
                 max_active=2000,
                 min_active=20,
                 beam=16.0,
                 beam_delta=0.5,
                 max_seq_len=100000):
        """Init decoder set some inner variance.

        Args:
            fst_path: path of decoding graph
            acoustic_scale: float, default 0.1
            max_active: int, default 2000
            min_active: int, default 20
            beam: float,  default 16.0
            beam_delta: float, default 0.5
            max_seq_len: int, default 100000
        """
        # self.cur_toks = {}
        # self.prev_toks = {}
        self.beam_delta = beam_delta
        self.fst = fst.Fst.read(fst_path)
        self.acoustic_scale = acoustic_scale
        self.max_active = max_active
        self.min_active = min_active
        self.beam = beam
        self.max_seq_len = max_seq_len

    def decode(self, log_likelihood):
        """using log-likelihood and decoding graph to decode 

        Args:
            log_likelihood: np.array, (T, N)
            
        """
        self.init_decoding()
        while not self.end_detect():
            self.prev_toks = self.cur_toks
            self.cur_toks = {}
            weight_cutoff = self.process_emitting(
                encoder_outputs, inference_one_step_fn)
            self.process_nonemitting(weight_cutoff)

    def end_detect(self):
        """determine whether to stop propagating"""
        if self.cur_toks and self.num_steps_decoded < self.max_seq_len:
            return False
        else:
            return True

    def init_decoding(self):
        """Init decoding states for every input utterance

        """
        self.cur_toks = {}
        self.prev_toks = {}
        start_state = self.fst.start()
        assert start_state != -1
        dummy_arc = LatticeArc(0, 0, 0.0, start_state)
        self.cur_toks[start_state] = Token(dummy_arc, 0.0, None)
        self.num_steps_decoded = 0
        self.process_nonemitting(float('inf'))

    def deal_completed_token(self, state, eos_score):
        """deal completed token and rescale scores

        Args:
            state: state of the completed token
            eos_score: acoustic score of eos
        """
        tok = self.prev_toks[state]
        if self.fst.final(state).to_string() != fst.Weight.Zero('tropical').to_string():
            tok.rescaled_cost = ((tok.cost + (-eos_score) +
                                  float(self.fst.final(state)))/self.num_steps_decoded)
            self.completed_token_pool.append(tok)
        queue = []
        tmp_toks = {}
        queue.append(state)
        tmp_toks[state] = tok
        while queue:
            state = queue.pop()
            tok = tmp_toks[state]
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0:
                    new_tok = Token(arc, 0.0, tok, tok.cur_label,
                                    tok.inner_packed_states)
                    if self.fst.final(arc.nextstate).to_string() != fst.Weight.Zero('tropical').to_string():
                        new_tok.rescaled_cost = ((new_tok.cost +
                                                  float(self.fst.final(arc.nextstate)))/self.num_steps_decoded)
                        self.completed_token_pool.append(new_tok)
                    else:
                        tmp_toks[arc.nextstate] = new_tok
                        queue.append(arc.nextstate)

    def process_emitting(self, encoder_outputs, inference_one_step_fn):
        """Process one step emitting states using callback function

        Args:
            encoder_outputs: encoder outputs
            inference_one_step_fn: callback function

        Returns:
            next_weight_cutoff: cutoff for next step
        """
        state2id = {}
        cand_seqs = []
        inner_packed_states_array = []
        for idx, state in enumerate(self.prev_toks):
            state2id[state] = idx
            cand_seqs.append(self.prev_toks[state].cur_label)
            inner_packed_states_array.append(
                self.prev_toks[state].inner_packed_states)
        all_log_scores, inner_packed_states_array = inference_one_step_fn(encoder_outputs,
                                                                          cand_seqs, inner_packed_states_array)
        weight_cutoff, adaptive_beam, best_token = self.get_cutoff()
        next_weight_cutoff = float('inf')
        if best_token is not None:
            for arc in self.fst.arcs(best_token.arc.nextstate):
                if arc.ilabel != 0:
                    ac_cost = (-all_log_scores[state2id[best_token.arc.nextstate]][arc.ilabel-1] *
                               self.acoustic_scale)
                    new_weight = float(arc.weight) + best_token.cost + ac_cost
                    if new_weight + adaptive_beam < next_weight_cutoff:
                        next_weight_cutoff = new_weight + adaptive_beam
        for state, tok in self.prev_toks.items():
            seq_id = state2id[state]
            if tok.cost <= weight_cutoff:
                if self.eos == np.argmax(all_log_scores[seq_id]):
                    self.deal_completed_token(
                        state, all_log_scores[seq_id][self.eos])
                    continue
                for arc in self.fst.arcs(state):
                    if arc.ilabel != 0:
                        ac_cost = - \
                            all_log_scores[seq_id][arc.ilabel -
                                                   1] * self.acoustic_scale
                        new_weight = float(arc.weight) + tok.cost + ac_cost
                        if new_weight < next_weight_cutoff:
                            new_tok = Token(arc, ac_cost, tok, tok.cur_label+[arc.ilabel-1],
                                            inner_packed_states_array[seq_id])
                            if new_weight + adaptive_beam < next_weight_cutoff:
                                next_weight_cutoff = new_weight + adaptive_beam
                            if arc.nextstate in self.cur_toks:
                                if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                                    self.cur_toks[arc.nextstate] = new_tok
                            else:
                                self.cur_toks[arc.nextstate] = new_tok
        self.num_steps_decoded += 1
        return next_weight_cutoff

    def process_nonemitting(self, cutoff):
        """Process one step non-emitting states
        TODO: Delete tokens when necessary

        Args:
            cutoff: float, the cutoff cost, token 

        """
        queue = list(self.cur_toks.keys())
        while queue:
            state = queue.pop()
            tok = self.cur_toks[state]
            if tok.cost > cutoff:
                continue
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0:
                    new_tok = Token(arc, 0.0, tok)
                    if new_tok.cost < cutoff:
                        if arc.nextstate in self.cur_toks.keys():
                            if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                                delete_token(self.cur_toks[arc.nextstate])
                                self.cur_toks[arc.nextstate] = new_tok
                                queue.append(arc.nextstate)
                            else:
                                delete_token(new_tok)
                        else:
                            self.cur_toks[arc.nextstate] = new_tok
                            queue.append(arc.nextstate)
                    else:
                        delete_token(new_tok)



    def get_cutoff(self):
        """get cutoff used in current and next step

        Returns:
            beam_cutoff: beam cutoff
            adaptive_beam: adaptive beam
            best_token: best token this step
        """
        best_cost = float('inf')
        best_token = None
        if(self.max_active == sys.maxsize
                and self.min_active == 0):
            for _, tok in self.prev_toks.items():
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
                adaptive_beam = self.beam
                beam_cutoff = best_cost + self.beam
                return beam_cutoff, adaptive_beam, best_token
        else:
            tmp_array = []
            for _, tok in self.prev_toks.items():
                tmp_array.append(tok.cost)
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
            beam_cutoff = best_cost + self.beam
            min_active_cutoff = float('inf')
            max_active_cutoff = float('inf')
            if len(tmp_array) > self.max_active:
                np_tmp_array = np.array(tmp_array)
                k = self.max_active
                max_active_cutoff = np_tmp_array[np.argpartition(
                    np_tmp_array, k-1)[k-1]]
            if max_active_cutoff < beam_cutoff:
                adaptive_beam = max_active_cutoff - best_cost + self.beam_delta
                return max_active_cutoff, adaptive_beam, best_token
            if len(tmp_array) > self.min_active:
                np_tmp_array = np.array(tmp_array)
                k = self.min_active
                if k == 0:
                    min_active_cutoff = best_cost
                else:
                    min_active_cutoff = np_tmp_array[np.argpartition(
                        np_tmp_array, k-1)[k-1]]
            if min_active_cutoff > beam_cutoff:
                adaptive_beam = min_active_cutoff - best_cost + self.beam_delta
                return min_active_cutoff, adaptive_beam, best_token
            else:
                adaptive_beam = self.beam
                return beam_cutoff, adaptive_beam, best_token

    def get_best_path(self):
        """get decoding result in best completed path

        Returns:
            ans: id array of decoding results
        """
        ans = []
        if not self.completed_token_pool:
            return ans
        best_completed_tok = self.completed_token_pool[0]
        for tok in self.completed_token_pool:
            if best_completed_tok.rescaled_cost > tok.rescaled_cost:
                best_completed_tok = tok
        arcs_reverse = []
        tok = best_completed_tok
        while tok:
            arcs_reverse.append(tok.arc)
            tok = tok.prev_tok
        assert arcs_reverse[-1].nextstate == self.fst.start()
        arcs_reverse.pop()
        for arc in arcs_reverse[::-1]:
            if arc.olabel != 0:
                ans.append(arc.olabel)
        return ans
