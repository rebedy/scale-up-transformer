import logging
from typing import Iterable, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from helpers import adjust_logits_during_generation

logger = logging.getLogger(__name__)


def temperature_sampling(logits, temperature):
    if temperature is None or temperature == 0.0:
        return torch.argmax(logits)
    probs = F.softmax(logits / temperature)
    pred_ids = probs.cpu().multinomial(probs.size()[1], replacement=False)
    return pred_ids


# nucleus
def top_p_sampling(logits, thres=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > thres  # (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')

    return sorted_logits.scatter(1, sorted_indices, sorted_logits)


# topk
def top_k_sampling(logits, thres=0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def top_k_top_p_filtering(logits, top_k=0, top_p=0.9, filter_value=-float("Inf"), min_tokens_to_keep=1,):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep),
                    logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def postprocess_next_token_scores(
    self,
    scores,
    input_ids,
    no_repeat_ngram_size,
    bad_words_ids,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    repetition_penalty,
    batch_size,
    num_beams,
):

    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores, batch_size, num_beams, input_ids, repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx, :].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(
        hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def generate_beam_search(
    self,
    input_ids,
    output_logits,
    cur_len,
    max_length,
    min_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    pad_token_id,
    eos_token_id,
    batch_size,
    num_beams,
    vocab_size,
    generated_hyps,
    beam_scores,
    done,
):
    """ Generate sequences for each example with beam search.
    """

    outputs = output_logits  # (batch_size * num_beams, cur_len, vocab_size)
    # (batch_size * num_beams, vocab_size)
    next_token_logits = outputs[:, -1, :]

    # if model has past, then set the past variable to speed up decoding
    if do_sample is False:  # self.config.is_encoder_decoder and
        # TODO (PVP) still a bit hacky here - there might be a better solution
        next_token_logits = adjust_logits_during_generation(self,
                                                            next_token_logits, cur_len=cur_len, max_length=max_length
                                                            )

    # (batch_size * num_beams, vocab_size)
    scores = F.log_softmax(next_token_logits, dim=-1)

    scores = postprocess_next_token_scores(self,
                                           scores=scores,
                                           input_ids=input_ids,
                                           no_repeat_ngram_size=no_repeat_ngram_size,
                                           bad_words_ids=bad_words_ids,
                                           cur_len=cur_len,
                                           min_length=min_length,
                                           max_length=max_length,
                                           eos_token_id=eos_token_id,
                                           repetition_penalty=repetition_penalty,
                                           batch_size=batch_size,
                                           num_beams=num_beams,
                                           )

    assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
        scores.shape, (batch_size * num_beams, vocab_size)
    )

    if do_sample:
        # (batch_size * num_beams, vocab_size)
        _scores = scores + beam_scores[:, None].expand_as(scores)
        # Temperature
        if temperature != 1.0:
            _scores = _scores / temperature
        # Top-p/top-k filtering
        _scores = top_k_top_p_filtering(
            _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
        )  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together to sample from all beam_idxs
        _scores = _scores.contiguous().view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
        probs = F.softmax(_scores, dim=-1)
        # (batch_size, num_beams * 2)
        next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
        # Compute next scores
        # (batch_size, num_beams * 2)
        next_scores = torch.gather(_scores, -1, next_tokens)
        # sort the sampled vector to make sure that the first num_beams samples are the best
        next_scores, next_scores_indices = torch.sort(
            next_scores, descending=True, dim=1)
        # (batch_size, num_beams * 2)
        next_tokens = torch.gather(next_tokens, -1, next_scores_indices)

    else:
        # (batch_size * num_beams, vocab_size)
        next_scores = scores + beam_scores[:, None].expand_as(scores)

        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(
            next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

    assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

    # next batch beam content
    next_batch_beam = []

    # for each sentence
    for batch_idx in range(batch_size):

        # if we are done with this sentence, add a pad token
        if done[batch_idx]:
            assert (
                len(generated_hyps[batch_idx]) >= num_beams
            ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
            assert (
                eos_token_id is not None and pad_token_id is not None
            ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
            next_batch_beam.extend(
                [(0, pad_token_id, 0)] * num_beams)  # pad the batch
            continue

        # next sentence beam content, this will get added to next_batch_beam
        next_sent_beam = []

        # next tokens for this sentence
        for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
            zip(next_tokens[batch_idx], next_scores[batch_idx])
        ):
            # get beam and token IDs
            beam_id = beam_token_id // vocab_size
            token_id = beam_token_id % vocab_size

            effective_beam_id = batch_idx * num_beams + beam_id
            # add to generated hypotheses if end of sentence
            if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                # if beam_token does not belong to top num_beams tokens, it should not be added
                is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                if is_beam_token_worse_than_top_num_beams:
                    continue
                generated_hyps[batch_idx].add(
                    input_ids[effective_beam_id].clone(
                    ), beam_token_score.item(),
                )
            else:
                # add next predicted token since it is not eos_token
                next_sent_beam.append(
                    (beam_token_score, token_id, effective_beam_id))

            # once the beam for next step is full, don't add more tokens to it.
            if len(next_sent_beam) == num_beams:
                break

        # Check if we are done so that we can save a pad step if all(done)
        done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
            next_scores[batch_idx].max().item(), cur_len
        )

        # update next beam content
        assert len(next_sent_beam) == num_beams, "Beam should always be full"
        next_batch_beam.extend(next_sent_beam)
        assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

    return done, next_batch_beam, next_scores, next_tokens


def decode_beam_search(
    self,
    input_ids,
    cur_len,
    max_length,
    do_sample,
    pad_token_id,
    eos_token_id,
    batch_size,
    num_return_sequences,
    num_beams,
    vocab_size,
    generated_hyps,
    beam_scores,
    past,
    done,
    next_batch_beam,
    next_scores,
    next_tokens,
):

    # sanity check / prepare next batch
    assert len(next_batch_beam) == batch_size * num_beams
    beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
    beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
    beam_idx = input_ids.new([x[2] for x in next_batch_beam])

    # re-order batch and update current length
    input_ids = input_ids[beam_idx, :]
    input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
    cur_len = cur_len + 1

    # re-order internal states
    if past is not None:
        past = self._reorder_cache(past, beam_idx)

    # extend attention_mask for new generated input if only decoder
        # if self.config.is_encoder_decoder is False:
        #     attention_mask = torch.cat(
        #         [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
        #     )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[
                    batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx], beam_scores.view(batch_size, num_beams)[
                    batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # shorter batches are padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined"
        sent_max_len = min(sent_lengths.max().item() + 1, max_length)
        decoded = input_ids.new(
            output_batch_size, sent_max_len).fill_(pad_token_id)

        # fill with hypothesis and eos_token_id if necessary
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
    else:
        # none of the hypotheses have an eos_token
        assert (len(hypo) == max_length for hypo in best)
        decoded = torch.stack(best).type(torch.long).to(
            next(self.parameters()).device)

    return decoded


@staticmethod
def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)