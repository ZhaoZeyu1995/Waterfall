"""
WFST decoder -- A Viterbi Token Passing Algorithm Implementation 
Note that, this is a prefix based decoder
We only keep one token for each state and prefix pair.
Note that, this is different from a conventional token passing, where there is at most one token for each state in the decoding graph.
Clearly, there can be different path, which means different prefixes, corresponding to one single state.
"""
import sys
import numpy as np
import openfst_python as fst
import logging

logging.basicConfig(filemode='decode.log', level=logging.INFO)


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
        self.arc = LatticeArc(arc.ilabel, arc.olabel,
                              arc.weight, arc.nextstate)
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
    # print('Deleting tokens')
    cur = token
    while cur:
        prev = cur.prev_tok
        del cur
        cur = prev
    # print('Finished!')


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
                 allow_partial=False,
                 beam_delta=0.5):
        """Init decoder set some inner variance.

        Args:
            fst_path: path of decoding graph
            acoustic_scale: float, default 0.1
            max_active: int, default 2000
            min_active: int, default 20
            beam: float,  default 16.0
            beam_delta: float, default 0.5
        """
        self.beam_delta = beam_delta
        logging.info('Loading the decoding graph...')
        self.fst = fst.Fst.read(fst_path)
        logging.info('Done!')
        self.acoustic_scale = acoustic_scale
        self.max_active = max_active
        self.min_active = min_active
        self.beam = beam
        self.allow_partial = allow_partial

    def decode(self, log_likelihood: np.array):
        """using log-likelihood and decoding graph to decode 

        Args:
            log_likelihood: np.array, (T, N)

        """
        self.init_decoding()
        self.log_likelihood_scaled = - self.acoustic_scale * log_likelihood
        self.target_frames_decoded = self.log_likelihood_scaled.shape[0]

        while self.num_frames_decoded < self.target_frames_decoded:
            # print('self.num_frames_decoded', self.num_frames_decoded)
            self.prev_toks = self.cur_toks
            self.cur_toks = {}
            # print('process_emitting...')
            weight_cutoff = self.process_emitting()
            # print('Number of prefix after emitting %d' % (len(self.cur_toks)))
            # print('Number of token after emitting %d' % (len(self.get_cur_tokens())))
            # self.report_cur_toks()
            # print('process_nonemitting...')
            # msg = 'At frame %d, after processing_emitting # prefixes %d' % (
                # self.num_frames_decoded, len(self.get_cur_prefixes()))
            # logging.info(msg)
            self.process_nonemitting(weight_cutoff)
            # msg = 'At frame %d, after processing_nonemitting # prefixes %d' % (
                # self.num_frames_decoded, len(self.get_cur_prefixes()))
            # logging.info(msg)

            # print('Number of prefix after nonemitting %d' % (len(self.cur_toks)))
            # print('Number of token after nonemitting %d' % (len(self.get_cur_tokens())))
            # self.report_cur_toks()

    def report_cur_toks(self):
        print('At frame', self.num_frames_decoded)
        for prefix, toks in self.cur_toks.items():
            print('prefix', prefix)
            print('has', len(toks), 'tokens')
            for tok in toks:
                print('tok.arc.ilabel', tok.arc.ilabel)
                print('tok.arc.olabel', tok.arc.olabel)
                print('tok.cost', tok.cost)
                print('tok.prev_tok', tok.prev_tok)

    def init_decoding(self):
        """Init decoding states for every input utterance

        """
        self.cur_toks = {
        }  # (prefix , state)-> token, where prefix is tuple of int
        self.prev_toks = {}  # (prefix, state)-> token
        start_state = self.fst.start()
        assert start_state != -1
        dummy_arc = LatticeArc(0, 0, 0.0, start_state)
        # The initial prefix is (), which is an empty tuple
        self.cur_toks[((), start_state)] = Token(dummy_arc, 0.0, None)

        self.num_frames_decoded = 0
        self.process_nonemitting(float('inf'))

    def get_prev_tokens(self):  # Get all prev tokens
        all_tokens = []
        for _, tokens in self.prev_toks.items():
            all_tokens.extend(tokens)
        return all_tokens

    def get_cur_tokens(self):  # Get all current tokens
        all_tokens = []
        for _, tokens in self.cur_toks.items():
            all_tokens.extend(tokens)
        return all_tokens

    def get_cur_tokens_states(self):  # Get all current active states
        all_states = []
        for _, tokens in self.cur_toks.items():
            all_states.extend([token.nextstate for token in tokens])
        return all_states

    def get_prefix_token_pair(self):  # Get all (prefix, token) pairs
        all_prefix_token_pair = []
        for prefix, tokens in self.cur_toks.items():
            for token in tokens:
                all_prefix_token_pair.append((prefix, token))
        return all_prefix_token_pair

    # Get all previous (prefix, token) pairs
    def get_prev_prefix_token_pair(self):
        all_prefix_token_pair = []
        for prefix, tokens in self.prev_toks.items():
            for token in tokens:
                all_prefix_token_pair.append((prefix, token))
        return all_prefix_token_pair

    def reached_final(self):
        '''
        Check if any one of the tokens in self.cur_toks has reached to a final state of self.fst
        '''
        for (prefix, state), tok in self.cur_toks.items():
            if (tok.cost != float('inf') and self.fst.final(state) != fst.Weight.Zero(self.fst.weight_type())):
                return True
        return False

    def extract_prefix_final_only(self):
        '''
        This is called usually at the end of the decoding to calculate the cost for each prefix in self.cur_toks
        The paths which end at a final state are taken into account only.
        '''

        prefix2cost = {}
        for (prefix, state), tok in self.cur_toks.items():
            if (tok.cost != float('inf') and self.fst.final(state) != fst.Weight.Zero(self.fst.weight_type())):
                if prefix in prefix2cost:
                    prefix2cost[prefix] = np.logaddexp(prefix2cost[prefix], np.float64(tok.cost)) 
                else:
                    prefix2cost[prefix] = np.float64(tok.cost)
        return prefix2cost

    def extract_prefix_all(self):
        '''
        This is called usually at the end of the decoding to calculate the cost for each prefix in self.cur_toks
        All the paths, whether end at a final state or not, are taken into account.
        '''

        prefix2cost = {}
        for (prefix, state), tok in self.cur_toks.items():
            if prefix in prefix2cost:
                prefix2cost[prefix] = np.logaddexp(prefix2cost[prefix], np.float64(tok.cost)) 
            else:
                prefix2cost[prefix] = np.float64(tok.cost)
        return prefix2cost

    def get_cur_prefixes(self):
        all_prefixes = set()
        for (prefix, _) in self.cur_toks:
            all_prefixes.add(prefix)
        return all_prefixes

    def process_emitting(self):
        """Process one step emitting states using callback function
        Returns:
            next_weight_cutoff: float, cutoff for next step
        """
        frame = self.num_frames_decoded

        # print('Calculating cutoff...')
        weight_cutoff, adaptive_beam, best_token, tok_count = self.get_cutoff()
        # print('Calculating cutoff finished')

        logging.info('For frame %d, there are %d tokens' %
                     (frame, tok_count))  # This is for debugging only

        # print('best_token', best_token)
        next_weight_cutoff = float('inf')
        if best_token is not None:  # Process the best token first, and hopefully find a proper next_weight_cutoff
            # At the beginning, best_token.arc.nextstate is the start state of the decoding graph
            for arc in self.fst.arcs(best_token.arc.nextstate):
                if arc.ilabel != 0:
                    ac_cost = self.log_likelihood_scaled[frame, int(
                        arc.ilabel)-1]
                    new_weight = float(arc.weight) + best_token.cost + ac_cost
                    if new_weight + adaptive_beam < next_weight_cutoff:  # make next_weight_cutoff tighter
                        next_weight_cutoff = new_weight + adaptive_beam

        # print('Got a hopefully proper next_weight_cutoff')

        # print('Begin to iterate through self.prev_toks.items() %d' % (len(self.prev_toks)))

        # for state, tok in self.prev_toks.items():
            # print('state', state)
            # print('tok.arc.ilabel', tok.arc.ilabel)
            # print('tok.arc.olabel', tok.arc.olabel)
            # print('tok.cost', tok.cost)
            # print('tok.prev_tok', tok.prev_tok)

        for (prefix, state), tok in self.prev_toks.items():
            if tok.cost < weight_cutoff:
                for arc in self.fst.arcs(state):
                    if arc.ilabel != 0:
                        # print('processing the arc', arc, 'for the state', state)
                        # print('arc.ilabel', arc.ilabel)
                        ac_cost = self.log_likelihood_scaled[frame, int(
                            arc.ilabel)-1]
                        # print('Got the log_likelihood for ', arc.ilabel)
                        new_weight = float(arc.weight) + tok.cost + ac_cost
                        if new_weight < next_weight_cutoff:
                            new_tok = Token(arc, ac_cost, tok)
                            # print('Created a new token for arc.ilabel', arc.ilabel)

                            if new_weight + adaptive_beam < next_weight_cutoff:  # make the next_weight_cutoff tighter
                                next_weight_cutoff = new_weight + adaptive_beam

                            if arc.olabel == 0:
                                if (prefix, arc.nextstate) in self.cur_toks:
                                    if self.cur_toks[(prefix, arc.nextstate)].cost > new_tok.cost:
                                        delete_token(
                                            self.cur_toks[(prefix, arc.nextstate)])
                                        self.cur_toks[(
                                            prefix, arc.nextstate)] = new_tok
                                    else:
                                        delete_token(new_tok)
                                else:
                                    self.cur_toks[(
                                        prefix, arc.nextstate)] = new_tok
                            else:
                                new_prefix = prefix + (arc.olabel,)
                                if (new_prefix, arc.nextstate) in self.cur_toks:
                                    if self.cur_toks[(new_prefix, arc.nextstate)].cost > new_tok.cost:
                                        delete_token(
                                            self.cur_toks[(new_prefix, arc.nextstate)])
                                        self.cur_toks[(
                                            new_prefix, arc.nextstate)] = new_tok
                                    else:
                                        delete_token(new_tok)
                                else:
                                    self.cur_toks[(
                                        new_prefix, arc.nextstate)] = new_tok
            delete_token(self.prev_toks[(prefix, state)])
        self.prev_toks = {}
        self.num_frames_decoded += 1
        return next_weight_cutoff

    def process_nonemitting(self, cutoff):
        """Process one step non-emitting states
        Delete tokens when possible

        Args:
            cutoff: float, the cutoff cost, token 

        """

        queue = list(self.cur_toks.keys())
        while queue:
            (prefix, state) = queue.pop()
            tok = self.cur_toks[(prefix, state)]
            if tok.cost > cutoff:
                continue
            for arc in self.fst.arcs(state):
                if arc.ilabel == 0:
                    new_tok = Token(arc, 0.0, tok)
                    if new_tok.cost > cutoff:
                        delete_token(new_tok)
                    else:
                        if arc.olabel == 0:
                            if (prefix, arc.nextstate) in self.cur_toks.keys():
                                # update the token for that (prefix, state)
                                if self.cur_toks[(prefix, arc.nextstate)].cost > new_tok.cost:
                                    delete_token(
                                        self.cur_toks[(prefix, arc.nextstate)])
                                    self.cur_toks[(
                                        prefix, arc.nextstate)] = new_tok
                                    queue.append((prefix, arc.nextstate))
                                else:
                                    delete_token(new_tok)
                            else:  # Add a new state in self.cur_toks
                                self.cur_toks[(
                                    prefix, arc.nextstate)] = new_tok
                                queue.append((prefix, arc.nextstate))
                        else:
                            new_prefix = prefix + (arc.olabel,)
                            if (new_prefix, arc.nextstate) in self.cur_toks.keys():
                                # update the token for that (prefix, state)
                                if self.cur_toks[(new_prefix, arc.nextstate)].cost > new_tok.cost:
                                    delete_token(
                                        self.cur_toks[(new_prefix, arc.nextstate)])
                                    self.cur_toks[(
                                        new_prefix, arc.nextstate)] = new_tok
                                    queue.append((new_prefix, arc.nextstate))
                                else:
                                    delete_token(new_tok)
                            else:  # Add a new state in self.cur_toks
                                self.cur_toks[(
                                    new_prefix, arc.nextstate)] = new_tok
                                queue.append((new_prefix, arc.nextstate))

                    # if new_tok.cost < cutoff:
                        # if arc.nextstate in self.cur_toks.keys():
                            # if self.cur_toks[arc.nextstate].cost > new_tok.cost:
                            # delete_token(self.cur_toks[arc.nextstate])
                            # self.cur_toks[arc.nextstate] = new_tok
                            # queue.append(arc.nextstate)
                            # else:
                            # delete_token(new_tok)
                        # else:
                            # self.cur_toks[arc.nextstate] = new_tok
                            # queue.append(arc.nextstate)
                    # else:
                        # delete_token(new_tok)

    def get_cutoff(self):
        """get cutoff used in current and next step

        Returns:
            beam_cutoff: float, beam cutoff
            adaptive_beam: float, adaptive beam
            best_token: float, best token this step
            tok_count: int, the number of tokens we currently keep
        """

        best_cost = float('inf')
        best_token = None
        tok_count = len(self.prev_toks)

        if (self.max_active == sys.maxsize
                and self.min_active == 0):
            for _, tok in self.prev_toks.items():
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
                adaptive_beam = self.beam
                beam_cutoff = best_cost + self.beam
            return beam_cutoff, adaptive_beam, best_token, tok_count
        else:
            tmp_array = []
            for _, tok in self.prev_toks.items():
                tmp_array.append(tok.cost)
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
            tok_count = len(self.prev_toks)
            beam_cutoff = best_cost + self.beam
            min_active_cutoff = float('inf')
            max_active_cutoff = float('inf')
            if len(tmp_array) > self.max_active:
                np_tmp_array = np.array(tmp_array)
                k = self.max_active
                np_tmp_array_partitioned = np_tmp_array[np.argpartition(
                    np_tmp_array, k-1)]
                max_active_cutoff = np_tmp_array_partitioned[k-1]
            if max_active_cutoff < beam_cutoff:  # tighter
                adaptive_beam = max_active_cutoff - best_cost + self.beam_delta
                # no need to check min_active
                return max_active_cutoff, adaptive_beam, best_token, tok_count
            # max_active_cutoff >= beam_cutoff looser, we need to set an adaptive_beam which keeps at least min_active
            if len(tmp_array) > self.min_active:
                np_tmp_array = np.array(tmp_array)
                k = self.min_active
                if k == 0:
                    min_active_cutoff = best_cost
                else:
                    if len(tmp_array) > self.max_active:
                        np_tmp_array_partitioned_part = np_tmp_array_partitioned[:self.max_active]
                        min_active_cutoff = np_tmp_array_partitioned_part[np.argpartition(
                            np_tmp_array_partitioned_part, k-1)][k-1]
                    else:
                        min_active_cutoff = np_tmp_array[np.argpartition(
                            np_tmp_array, k-1)[k-1]]
            if min_active_cutoff > beam_cutoff:  # min_active_cutoff if losser than beam_cutoff, we need to make adaptive_beam larger so that we can keep at least min_active tokens
                adaptive_beam = min_active_cutoff - best_cost + self.beam_delta
                return min_active_cutoff, adaptive_beam, best_token, tok_count
            else:
                adaptive_beam = self.beam
                return beam_cutoff, adaptive_beam, best_token, tok_count

    def get_best_path(self):
        """get decoding result in best completed prefix

        Returns:
            wordid_result: tuple of int, id array of decoding results
        """
        # print('Checking if reached_final')
        is_final = self.reached_final()
        # print('is_final or not', is_final)
        # print('Finished checking ')
        if self.allow_partial:
            logging.info('Allow partial!')
            if not is_final:
                logging.warn('Not reached the final states but allow partial!')
            prefix2cost = self.extract_prefix_all()
            best_cost = float('inf')
            best_prefix = None
            for prefix, cost in prefix2cost.items():
                if (cost < best_cost):
                    best_cost = cost
                    best_prefix = prefix
            return best_prefix
        else:
            logging.info('Not allow partial!')
            if not is_final:
                logging.error('Not reached the final but not allow partial.')
                return None
            prefix2cost = self.extract_prefix_final_only()
            best_cost = float('inf')
            best_token = None
            # print('Iterating over self.cur_toks.items(), ', len(self.cur_toks))
            for prefix, cost in prefix2cost.items():
                if (cost < best_cost):
                    best_cost = cost
                    best_prefix = prefix
            return best_prefix
