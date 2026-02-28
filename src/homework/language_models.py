# ========================================================================
# Copyright 2024 Emory University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================

__author__ = 'Jinho D. Choi'

from src.ngram_models import Bigram
from collections import defaultdict
from typing import Dict, List
import math
import string

UNKNOWN = ''
INIT = '[INIT]'


def bigram_model(filepath: str) -> Bigram:
    """
    Build a bigram language model from text file using Laplace smoothing, normalization, and initial word probabilities.

    :param filepath: Path to a text file.
    :return: Nested dictionary where:
             - Outer key is previous word (including INIT and UNKNOWN)
             - Inner key is current word (including UNKNOWN)
             - Value is P(current|previous) probability
    """
    # bigram counts
    bigram_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # previous counts
    prev_counts: Dict[str, int] = defaultdict(int)

    vocab = set()  # words seen in the corpus

    # 1) Count bigrams 
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            words = line.split()  
            for w in words:
                vocab.add(w)

            prev = INIT
            for curr in words:
                bigram_counts[prev][curr] += 1
                prev_counts[prev] += 1
                prev = curr

    # 2) Add UNKNOWN to the set of possible current words for smoothing
    next_vocab = set(vocab)
    next_vocab.add(UNKNOWN)
    # smoothing adds 1 for each possible next word
    smooth = len(next_vocab)  

    # 3) Build probability table with Laplace smoothing + normalization
    model: Bigram = {}

    # We want probabilities for every observed previous word (includes INIT and UNKNOWN)
    prev_vocab = set(prev_counts.keys())
    prev_vocab.add(INIT)
    prev_vocab.add(UNKNOWN)

    for prev in prev_vocab:
        model[prev] = {}
        denom = prev_counts.get(prev, 0) + smooth  

        for curr in next_vocab:
            temp = bigram_counts.get(prev, {}).get(curr, 0)
            model[prev][curr] = (temp + 1) / denom  

    return model

def _is_punct(tok: str) -> bool:
    # Treat tokens that are ONLY punctuation as punctuation tokens
    return tok != "" and all(ch in string.punctuation for ch in tok)


def _get_row(bigram_probs: Bigram, prev: str) -> dict[str, float]:
    # Unknown previous word then use UNKNOWN row (per spec)
    if prev in bigram_probs:
        return bigram_probs[prev]
    return bigram_probs.get(UNKNOWN, {})


def _get_prob(bigram_probs: Bigram, prev: str, curr: str) -> float:
    row = _get_row(bigram_probs, prev)
    # Unknown current word then use UNKNOWN column (per spec)
    if curr in row:
        return row[curr]
    return row.get(UNKNOWN, 0.0)

def sequence_generator(bigram_probs: Bigram, init_word: str, length: int = 20) -> tuple[list[str], float]:
    """
    Generate a sequence of specified length starting with given word.

    :param bigram_probs: Bigram probabilities from bigram_model().
    :param init_word: First word in sequence.
    :param length: Number of words to generate.
    :return: Tuple containing:
             - list[str]: Generated sequence
             - float: Log probability of sequence using natural log
    """
    if length <= 0:
        return ([], 0.0)

    max_punct = length // 5

    seq: List[str] = [init_word]
    logp = math.log(_get_prob(bigram_probs, INIT, init_word)) if init_word else 0.0

    punct_used = 1 if _is_punct(init_word) else 0
    used_nonpunct = set()
    if not _is_punct(init_word):
        used_nonpunct.add(init_word)

    prev = init_word

    while len(seq) < length:
        row = _get_row(bigram_probs, prev)

        # Sort next-token candidates by probability (highest first)
        # (include unknown too if present, but we'll try to avoid it unless necessary)
        candidates = sorted(row.items(), key=lambda kv: kv[1], reverse=True)

        chosen = None
        chosen_p = None

        for tok, p in candidates:
            if tok == INIT:
                continue  # INIT should not appear inside sequence

            is_p = _is_punct(tok)

            # punctuation budget
            if is_p and punct_used >= max_punct:
                continue

            # no repeats among non-punct
            if (not is_p) and (tok in used_nonpunct):
                continue

            # avoid empty UNKNOWN token unless we have no other option!
            if tok == UNKNOWN:
                continue

            chosen, chosen_p = tok, p
            break

        # If we didn't pick anything , relax to allow UNKNOWN
        if chosen is None:
            for tok, p in candidates:
                if tok == INIT:
                    continue
                is_p = _is_punct(tok)
                if is_p and punct_used >= max_punct:
                    continue
                if (not is_p) and (tok in used_nonpunct):
                    continue
                chosen, chosen_p = tok, p
                break

        # Absolute fallback: if row is empty, step using unknown 
        if chosen is None:
            chosen = UNKNOWN
            chosen_p = _get_prob(bigram_probs, prev, UNKNOWN)

        seq.append(chosen)

        # log-likelihood accumulates log P(chosen | prev) using unkown rules
        p_use = _get_prob(bigram_probs, prev, chosen)
        # p_use should be > 0 with Laplace smoothing; but just as a sake guard here 
        logp += math.log(p_use) if p_use > 0 else float("-inf")

        if _is_punct(chosen):
            punct_used += 1
        else:
            used_nonpunct.add(chosen)

        prev = chosen

    return (seq, logp)



def sequence_generator_plus(bigram_probs: Bigram, init_word: str, length: int = 20) -> tuple[list[str], float]:
    """
        Generate a sequence of specified length starting with given word, which generates sequences with
        higher probability scores and better semantic coherence compared to sequence_generator().

        :param bigram_probs: Bigram probabilities from bigram_model().
        :param init_word: First word in sequence.
        :param length: Number of words to generate.
        :return: Tuple containing:
                 - list[str]: Generated sequence
                 - float: Log probability of sequence using natural log
        """
    if length <= 0:
        return ([], 0.0)

    max_punct = length // 5

    seq: List[str] = [init_word]
    logp = math.log(_get_prob(bigram_probs, INIT, init_word)) if init_word else 0.0

    punct_used = 1 if _is_punct(init_word) else 0
    used_nonpunct = set()
    if not _is_punct(init_word):
        used_nonpunct.add(init_word)

    prev = init_word

    while len(seq) < length:
        row1 = _get_row(bigram_probs, prev)
        cand1 = sorted(row1.items(), key=lambda kv: kv[1], reverse=True)

        best_tok = None
        best_score = float("-inf")
        best_p = None

        for tok, p12 in cand1:
            if tok == INIT:
                continue

            is_p = _is_punct(tok)
            if is_p and punct_used >= max_punct:
                continue
            if (not is_p) and (tok in used_nonpunct):
                continue

            # prefer not choosing unknown unless forced later
            if tok == UNKNOWN:
                continue

            # compute lookahead: best next log-prob from tok, with updated constraints
            new_punct_used = punct_used + (1 if is_p else 0)
            new_used_nonpunct = used_nonpunct | ({tok} if not is_p else set())

            row2 = _get_row(bigram_probs, tok)
            cand2 = sorted(row2.items(), key=lambda kv: kv[1], reverse=True)

            best_log_next = float("-inf")
            for tok2, p23 in cand2:
                if tok2 == INIT:
                    continue
                is_p2 = _is_punct(tok2)
                if is_p2 and new_punct_used >= max_punct:
                    continue
                if (not is_p2) and (tok2 in new_used_nonpunct):
                    continue
                # allow unknown in lookahead, but not preferred
                best_log_next = math.log(p23) if p23 > 0 else float("-inf")
                break

            score = (math.log(p12) if p12 > 0 else float("-inf")) + (best_log_next if best_log_next != float("-inf") else 0.0)

            if score > best_score:
                best_score = score
                best_tok = tok
                best_p = p12

        # If no not unknown feasible, allow unknown as last resort
        if best_tok is None:
            for tok, p12 in cand1:
                if tok == INIT:
                    continue
                is_p = _is_punct(tok)
                if is_p and punct_used >= max_punct:
                    continue
                if (not is_p) and (tok in used_nonpunct):
                    continue
                best_tok = tok
                best_p = p12
                break

        if best_tok is None:
            best_tok = UNKNOWN
            best_p = _get_prob(bigram_probs, prev, UNKNOWN)

        seq.append(best_tok)
        p_use = _get_prob(bigram_probs, prev, best_tok)
        logp += math.log(p_use) if p_use > 0 else float("-inf")

        if _is_punct(best_tok):
            punct_used += 1
        else:
            used_nonpunct.add(best_tok)

        prev = best_tok

    return (seq, logp)


if __name__ == '__main__':
    filepath = 'dat/chronicles_of_narnia.txt'
    bigram_probs = bigram_model(filepath)
    sequence_generator(bigram_probs, 'You')

    #testing purpose 
    # row = bigram_probs['the']        # or any prev token
    # print(sum(row.values()))