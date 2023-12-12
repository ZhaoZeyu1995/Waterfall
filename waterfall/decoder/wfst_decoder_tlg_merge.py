"""
WFST decoder -- A Viterbi Token Passing Algorithm Implementation

This decoder takes two FSTs as input, one T and one LG.

However, it still stores at most one token for each state pair (state_T, state_LG).
Here, we only store the token history for LG fst, because it is already enough for us to back trace and get the final result.

Note that, there is actually no cost in T fst, as it is just a limitation of the topology but does not have any
prior knowledge or information about which token has a higher probability to appear.

Besides, we have to make sure that there is no epsilon on the input sides of T fst.
LG should be arc_sorted by its input labels.
"""
import sys
import numpy as np
import openfst_python as fst
import logging

logging.basicConfig(filemode="decode.log", level=logging.INFO)


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

    def __init__(self, arc, acoustic_cost, state_t, prev_tok=None):
        self.prev_tok = prev_tok
        self.arc = LatticeArc(arc.ilabel, arc.olabel, arc.weight, arc.nextstate)
        self.state_t = state_t
        if prev_tok is not None:
            self.cost = prev_tok.cost + float(arc.weight) + acoustic_cost
        else:
            self.cost = float(arc.weight) + acoustic_cost


def delete_token(token: Token):
    """
    Delete a token and its history recursively
    May not be very useful.
    """
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

    def __init__(
        self,
        t_path,
        lg_path,
        acoustic_scale=0.1,
        max_active=2000,
        min_active=20,
        beam=16.0,
        beam_delta=0.5,
    ):
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
        logging.info("Loading the decoding graph...")
        self.t_fst = fst.Fst.read(t_path)
        self.lg_fst = fst.Fst.read(lg_path)
        logging.info("Done!")
        self.acoustic_scale = acoustic_scale
        self.max_active = max_active
        self.min_active = min_active
        self.beam = beam

        # print(self.words)
        # exit()

    def decode(self, log_likelihood: np.array):
        """using log-likelihood and decoding graph to decode

        Args:
            log_likelihood: np.array, (T, N)

        """
        self.init_decoding()
        self.log_likelihood_scaled = -self.acoustic_scale * log_likelihood
        self.target_frames_decoded = self.log_likelihood_scaled.shape[0]

        while self.num_frames_decoded < self.target_frames_decoded:
            # print('self.num_frames_decoded', self.num_frames_decoded)
            self.prev_toks = self.cur_toks
            self.cur_toks = {}
            # print('process_emitting...')
            weight_cutoff = self.process_emitting()
            # logging.info('After emitting For frame %d, there are %d tokens' %
            # (self.num_frames_decoded, len(self.cur_toks)))  # This is for debugging only
            # print('process_nonemitting...')
            self.process_nonemitting(weight_cutoff)
            # logging.info('After nonemitting For frame %d, there are %d tokens' %
            # (self.num_frames_decoded, len(self.cur_toks)))  # This is for debugging only

    def merge_and_prune_tokens(self, prev_or_cur, cutoff):
        """
        tokens: dict, (state_t, state_lg) -> {toks}
        """
        if prev_or_cur == "prev":
            tokens = self.prev_toks
        elif prev_or_cur == "cur":
            tokens = self.cur_toks

        new_tokens = dict()

        # Merge

        merged_tokens2cost = dict()
        for (state_t, state_lg), toks in tokens.items():
            for tok in toks:
                if (state_lg, toks.arc) not in merged_tokens2cost:
                    merged_tokens2cost[(state_lg, tok.arc)] = tok.cost
                else:
                    merged_tokens2cost[(state_lg, tok.arc)] = -np.logaddexp(
                        -merged_tokens2cost[(state_lg, tok.arc)], -tok.cost
                    )
        # Prune
        for (state_lg, arc), cost in merged_tokens2cost.items():
            if cost <= cutoff:
                for (state_t_tokens, state_lg_tokens), toks in tokens.items():
                    for tok in toks:
                        if (state_lg_tokens, tok.arc) == (state_lg, arc):
                            if (state_t_tokens, state_lg) in new_tokens:
                                new_tokens[(state_t, state_lg)].add(tok)
                            else:
                                new_tokens[(state_t, state_lg)] = {tok}
        return merged_tokens2cost, new_tokens

    def init_decoding(self):
        """Init decoding states for every input utterance"""
        self.cur_toks = {}
        self.cur_state_lg_arc_lg = {}
        self.state_lg_arc_lg2cost = {}  # (state_lg, arc_lg) -> cost
        self.prev_toks = {}
        start_state_t = self.t_fst.start()
        start_state_lg = self.lg_fst.start()
        assert start_state_t != -1
        assert start_state_lg != -1
        dummy_arc = LatticeArc(0, 0, 0.0, start_state_lg)
        self.cur_toks[(start_state_lg)] = {Token(dummy_arc, 0.0, None, None)}
        self.cur_state_lg_arc_lg[(start_state_lg, dummy_arc)] = {
            Token(dummy_arc, 0.0, start_state_t, None)
        }
        self.state_lg_arc_lg2cost[(start_state_lg, dummy_arc)] = 0.0
        self.num_frames_decoded = 0
        self.process_nonemitting(float("inf"))

    def reached_final(self):
        """
        Check if any one of the tokens in self.cur_toks has reached to a final state of self.fst
        """
        for state, tok in self.cur_toks.items():
            if (
                tok.cost != float("inf")
                and self.t_fst.final(state[0])
                != fst.Weight.Zero(self.t_fst.weight_type())
                and self.lg_fst.final(state[1])
                != fst.Weight.Zero(self.lg_fst.weight_type())
            ):
                return True
        return False

    def process_emitting(self):
        """Process one step emitting states using callback function
        Returns:
            next_weight_cutoff: float, cutoff for next step
        """
        frame = self.num_frames_decoded

        # print('Calculating cutoff...')
        (
            weight_cutoff,
            adaptive_beam,
            best_state,
            best_token,
            tok_count,
        ) = self.get_cutoff()
        # print('Calculating cutoff finished')

        # logging.info('For frame %d, there are %d tokens' %
        # (frame, tok_count))  # This is for debugging only

        # print('best_token', best_token)
        next_weight_cutoff = float("inf")
        if (
            best_token is not None
        ):  # Process the best token first, and hopefully find a proper next_weight_cutoff
            # At the beginning, best_token.arc.nextstate is the start state of the decoding graph
            for arc_t in self.t_fst.arcs(best_state[0]):
                # logging.info(str(self.log_likelihood_scaled.shape))
                ac_cost = self.log_likelihood_scaled[frame, int(arc_t.ilabel)]
                if arc_t.olabel != 0:
                    for arc_lg in self.lg_fst.arcs(best_state[1]):
                        # We have to make sure that the output label of T can be accepted by LG.
                        if arc_lg.ilabel == arc_t.olabel:
                            new_weight = (
                                float(arc_lg.weight) + best_token.cost + ac_cost
                            )
                            if (
                                new_weight + adaptive_beam < next_weight_cutoff
                            ):  # make next_weight_cutoff tighter
                                next_weight_cutoff = new_weight + adaptive_beam
                        elif (
                            arc_t.olabel < arc_lg.ilabel
                        ):  # because LG has been arc_sorted according to the input labels
                            break
                else:
                    new_weight = (
                        best_token.cost + ac_cost
                    )  # we don't have to transit in LG.
                    if (
                        new_weight + adaptive_beam < next_weight_cutoff
                    ):  # make next_weight_cutoff tighter
                        next_weight_cutoff = new_weight + adaptive_beam

        # print('Got a hopefully proper next_weight_cutoff')

        # print('Begin to iterate through self.prev_toks.items() %d' % (len(self.prev_toks)))

        # print('Frame', self.num_frames_decoded)
        # for state, tok in self.prev_toks.items():
        # print('state', state)
        # print('tok.arc.ilabel', tok.arc.ilabel)
        # print('tok.arc.olabel', tok.arc.olabel)
        # print('tok.cost', tok.cost)
        # print('tok.prev_tok', tok.prev_tok)

        # print('adaptive_beam', adaptive_beam)
        for (state_t, state_lg), tok in self.prev_toks.items():
            # print('next_weight_cutoff', next_weight_cutoff)
            if tok.cost < weight_cutoff:
                for arc_t in self.t_fst.arcs(state_t):
                    ac_cost = self.log_likelihood_scaled[frame, int(arc_t.ilabel)]
                    if arc_t.olabel == 0:  # we only update T state in this case
                        # print('Found arc_t.olabel == 0')
                        new_weight = tok.cost + ac_cost
                        if new_weight < next_weight_cutoff:
                            dummy_arc = LatticeArc(0, 0, 0.0, state_lg)
                            new_tok = Token(dummy_arc, ac_cost, tok)

                            if (
                                new_weight + adaptive_beam < next_weight_cutoff
                            ):  # make the next_weight_cutoff tighter
                                next_weight_cutoff = new_weight + adaptive_beam

                            if (
                                arc_t.nextstate,
                                state_lg,
                            ) in self.cur_toks:  # only update T state
                                if (
                                    self.cur_toks[(arc_t.nextstate, state_lg)].cost
                                    > new_tok.cost
                                ):
                                    # print('Updating token for ', (arc_t.nextstate, state_lg))
                                    # print('new_tok.cost', new_tok.cost)
                                    delete_token(
                                        self.cur_toks[(arc_t.nextstate, state_lg)]
                                    )
                                    self.cur_toks[(arc_t.nextstate, state_lg)] = new_tok
                                else:
                                    delete_token(new_tok)
                            else:
                                # print('Adding new state', (arc_t.nextstate, state_lg))
                                # print('new_tok.cost', new_tok.cost)

                                self.cur_toks[(arc_t.nextstate, state_lg)] = new_tok
                    else:  # we need to check if we need to up state LG state as well, when arc_lg.ilabel == arc_t.olabel
                        # print('Found act_t.olabel != 0', arc_t.olabel)
                        for arc_lg in self.lg_fst.arcs(state_lg):
                            # print('arc_lg.ilabel', arc_lg.ilabel)
                            if arc_t.olabel == arc_lg.ilabel:
                                # print('processing the arc', arc, 'for the state', state)
                                # print('arc.ilabel', arc.ilabel)

                                # print('Got the log_likelihood for ', arc.ilabel)
                                new_weight = float(arc_lg.weight) + tok.cost + ac_cost
                                if new_weight < next_weight_cutoff:
                                    new_tok = Token(arc_lg, ac_cost, tok)
                                    # print('Created a new token for arc.ilabel', arc.ilabel)

                                    if (
                                        new_weight + adaptive_beam < next_weight_cutoff
                                    ):  # make the next_weight_cutoff tighter
                                        next_weight_cutoff = new_weight + adaptive_beam

                                    if (
                                        arc_t.nextstate,
                                        arc_lg.nextstate,
                                    ) in self.cur_toks:
                                        if (
                                            self.cur_toks[
                                                (arc_t.nextstate, arc_lg.nextstate)
                                            ].cost
                                            > new_tok.cost
                                        ):
                                            # print('Updating token for ', (arc_t.nextstate, arc_lg.nextstate))
                                            # print('new_tok.cost', new_tok.cost)
                                            delete_token(
                                                self.cur_toks[
                                                    (arc_t.nextstate, arc_lg.nextstate)
                                                ]
                                            )
                                            self.cur_toks[
                                                (arc_t.nextstate, arc_lg.nextstate)
                                            ] = new_tok
                                        else:
                                            delete_token(new_tok)
                                    else:
                                        # print('Adding token for ', (arc_t.nextstate, arc_lg.nextstate))
                                        # print('new_tok.cost', new_tok.cost)
                                        self.cur_toks[
                                            (arc_t.nextstate, arc_lg.nextstate)
                                        ] = new_tok
                            elif arc_t.olabel < arc_lg.ilabel:
                                break
            delete_token(self.prev_toks[(state_t, state_lg)])
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
            (state_lg, arc_lg) = queue.pop()
            if self.cur_toks2cost[(state_lg, arc_lg)] > cutoff:
                continue
            toks = self.cur_toks[(state_lg, arc_lg)]
            for tok in toks:
                # if tok.cost > cutoff:
                # continue
                for arc in self.lg_fst.arcs(state_lg):
                    if arc.ilabel == 0:
                        # print('Found Nonemmiting Arc in LG')
                        new_tok = Token(arc, 0.0, tok.state_t, tok)
                        # if new_tok.cost > cutoff:
                        # delete_token(new_tok)
                        # else:
                        if (arc.nextstate, arc) in self.cur_toks.keys():
                            # update the token for that state
                            # print('Nonemmiting Updating token for ', (state_t, arc.nextstate))
                            self.cur_toks[(arc.nextstate, arc)].add(new_tok)
                            self.cur_toks2cost[(arc.nextstate, arc)] = -np.logaddexp(
                                -self.cur_toks2cost[(arc.nextstate, arc)], -new_tok.cost
                            )
                        else:  # Add a new state in self.cur_toks
                            # print('Nonemitting Adding token for ', (state_t, arc.nextstate))
                            self.cur_toks[(arc.nextstate, arc)] = {new_tok}
                            self.cur_toks2cost[(arc.nextstate, arc)] = new_tok.cost
                        queue.append((arc.nextstate, arc))

    def get_cutoff(self):
        """get cutoff used in current and next step

        Returns:
            beam_cutoff: float, beam cutoff
            adaptive_beam: float, adaptive beam
            best_token: float, best token this step
            tok_count: int, the number of tokens we currently keep
        """
        best_cost = float("inf")
        best_token = None
        best_state = None
        tok_count = len(self.prev_toks)

        if self.max_active == sys.maxsize and self.min_active == 0:
            for state, tok in self.prev_toks.items():
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
                    best_state = state
                adaptive_beam = self.beam
                beam_cutoff = best_cost + self.beam
            return beam_cutoff, adaptive_beam, best_state, best_token, tok_count
        else:
            tmp_array = []
            for state, tok in self.prev_toks.items():
                tmp_array.append(tok.cost)
                if tok.cost < best_cost:
                    best_cost = tok.cost
                    best_token = tok
                    best_state = state
            beam_cutoff = best_cost + self.beam
            min_active_cutoff = float("inf")
            max_active_cutoff = float("inf")
            if len(tmp_array) > self.max_active:
                np_tmp_array = np.array(tmp_array)
                k = self.max_active
                np_tmp_array_partitioned = np_tmp_array[
                    np.argpartition(np_tmp_array, k - 1)
                ]
                max_active_cutoff = np_tmp_array_partitioned[k - 1]
            if max_active_cutoff < beam_cutoff:  # tighter
                adaptive_beam = max_active_cutoff - best_cost + self.beam_delta
                # no need to check min_active
                return (
                    max_active_cutoff,
                    adaptive_beam,
                    best_state,
                    best_token,
                    tok_count,
                )
            # max_active_cutoff >= beam_cutoff looser, we need to set an adaptive_beam which keeps at least min_active
            if len(tmp_array) > self.min_active:
                np_tmp_array = np.array(tmp_array)
                k = self.min_active
                if k == 0:
                    min_active_cutoff = best_cost
                else:
                    if len(tmp_array) > self.max_active:
                        np_tmp_array_partitioned_part = np_tmp_array_partitioned[
                            : self.max_active
                        ]
                        min_active_cutoff = np_tmp_array_partitioned_part[
                            np.argpartition(np_tmp_array_partitioned_part, k - 1)
                        ][k - 1]
                    else:
                        min_active_cutoff = np_tmp_array[
                            np.argpartition(np_tmp_array, k - 1)[k - 1]
                        ]
            if (
                min_active_cutoff > beam_cutoff
            ):  # min_active_cutoff if losser than beam_cutoff, we need to make adaptive_beam larger so that we can keep at least min_active tokens
                adaptive_beam = min_active_cutoff - best_cost + self.beam_delta
                return (
                    min_active_cutoff,
                    adaptive_beam,
                    best_state,
                    best_token,
                    tok_count,
                )
            else:
                adaptive_beam = self.beam
                return beam_cutoff, adaptive_beam, best_state, best_token, tok_count

    def get_best_path(self):
        """get decoding result in best completed path

        Returns:
            ans: id array of decoding results
        """
        # print('Checking if reached_final')
        is_final = self.reached_final()
        # print('is_final or not', is_final)
        # print('Finished checking ')
        if not is_final:
            logging.info("Not reached final!")
            best_token = None
            for state, tok in self.cur_toks.items():
                if best_token is None or tok.cost < best_token.cost:
                    best_token = tok
        else:
            best_cost = float("inf")
            best_token = None
            # print('Iterating over self.cur_toks.items(), ', len(self.cur_toks))
            for state, tok in self.cur_toks.items():
                # print('Checking state', state)
                this_cost = tok.cost + float(
                    self.lg_fst.final(state[1])
                )  # We only take the weight from LG.
                if this_cost < best_cost and this_cost != float("inf"):
                    best_cost = this_cost
                    best_token = tok
        if best_token is None:
            return False  # No output
        # print('Found the best_tok.arc', best_token.arc)
        # print('Found the best_tok.cost', best_token.cost)
        # print('Found the best_tok.prev_tok', best_token.prev_tok)

        wordid_result = []
        # arcs_reverse = []
        tok = best_token
        while tok is not None:
            # prev_cost = tok.prev_tok.cost if tok.prev_tok is not None else 0.0
            # tot_cost = tok.cost - prev_cost
            # graph_cost = float(tok.arc.weight)
            # ac_cost = tot_cost - graph_cost
            # arcs_reverse.append(LatticeArc(tok.arc.ilabel, tok.arc.olabel, (graph_cost, ac_cost), tok.arc.nextstate))
            if tok.arc.olabel != 0:
                wordid_result.insert(0, tok.arc.olabel)
            tok = tok.prev_tok

        return wordid_result
