import math
import random
from collections import defaultdict

import numpy as np


def clean_text(text):
    text = text.lower()
    return text


def load_ngram_index(index_file, max_misspell_frequency=1000000000):
    index = defaultdict(dict)

    with open(index_file, "r", encoding="utf-8") as f:
        for line in f:
            orig, misspell, joint_freq, orig_freq, misspelled_freq = line.strip().split("\t")
            if not misspell.replace("<DELETE>", ""):
                continue
            misspell = misspell.replace("<DELETE>", "=")
            if int(misspelled_freq) > max_misspell_frequency:
                continue
            index[orig][misspell] = int(joint_freq) / int(orig_freq)  # save the "translation probability"
    return index


def index_dictionary(user_phrases, ngram_index, min_log_prob, max_phrases_per_ngram):
    ngram_to_phrase_and_position = defaultdict(list)
    ngrams_to_skip = set()
    for phrase in user_phrases:
        phrase_tokens = phrase.split(" ")
        position_to_ngram_to_freq = [{} for _ in phrase_tokens]

        for begin in range(len(phrase_tokens)):
            for end in range(begin + 1, min(begin + 5, len(phrase_tokens) + 1)):
                phrase_ngram = " ".join(phrase_tokens[begin:end])
                if phrase_ngram not in ngram_index:
                    continue
                for misspell_ngram, freq in ngram_index[phrase_ngram].items():
                    logprob = math.log(freq)
                    for pos in range(max(0, end - 5), end):
                        new_ngrams = {}
                        for prev_ngram, prev_logprob in position_to_ngram_to_freq[pos].items():
                            if len(prev_ngram) + len(misspell_ngram) <= 10 and pos + prev_ngram.count(" ") == begin:
                                combined_logprob = prev_logprob + logprob
                                if combined_logprob > min_log_prob:
                                    combined_ngram = prev_ngram + misspell_ngram + " "
                                    new_ngrams[combined_ngram] = combined_logprob
                        position_to_ngram_to_freq[pos].update(new_ngrams)

                    if logprob > min_log_prob:
                        position_to_ngram_to_freq[begin][misspell_ngram + " "] = logprob

        for pos in range(len(position_to_ngram_to_freq)):
            sorted_ngrams = sorted(position_to_ngram_to_freq[pos].items(), key=lambda item: item[1], reverse=True)
            for ngram, logprob in sorted_ngrams:
                char_length = ngram.count(" ")
                ngram = ngram.replace("+", " ").replace("=", " ")
                if ngram in ngrams_to_skip:
                    continue
                ngram = " ".join(ngram.split())  # here ngram doesn't end with a space anymore
                ngram_to_phrase_and_position[ngram].append((phrase, pos, char_length, logprob))
                if len(ngram_to_phrase_and_position[ngram]) > max_phrases_per_ngram:
                    ngrams_to_skip.add(ngram)
                    del ngram_to_phrase_and_position[ngram]
                    continue

    phrases = []
    phrase2id = {}
    ngram2phrases = defaultdict(list)

    for prev_ngram in ngram_to_phrase_and_position:
        for phrase, pos, length, logprob in ngram_to_phrase_and_position[prev_ngram]:
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[prev_ngram].append((phrase2id[phrase], pos, length, logprob))

    return phrases, ngram2phrases


def search_index_for_hits(ngram2phrases, phrases, letters):
    if " " in letters:
        raise ValueError("space found in input hypothesis: " + str(letters))

    num_phrases = len(phrases)
    num_letters = len(letters)
    hit_matrix = np.zeros((num_phrases, num_letters), dtype=float)
    position2ngrams = [set() for _ in range(num_letters)]

    for start in range(len(letters)):
        for end in range(start + 1, min(num_letters + 1, start + 7)):
            ngram = " ".join(letters[start:end])
            if ngram not in ngram2phrases:
                continue

            for phrase_id, begin_pos, size, logprob in ngram2phrases[ngram]:
                hit_matrix[phrase_id, start:end] = 1.0
            position2ngrams[start].add(ngram)
    return hit_matrix, position2ngrams


def find_best_candidate_coverage(phrases, phrases2positions):
    """Get maximum hit coverage for each phrase - within a moving window of length of the phrase.
    Args:
        phrases: List of all phrases in custom vocabulary. Position corresponds to phrase_id.
        phrases2positions: a matrix of size (len(phrases), len(ASR-hypothesis)).
            It is filled with 1.0 (hits) on intersection of letter n-grams and phrases that are indexed by these n-grams, 0.0 - elsewhere.
    Returns:
        candidate2coverage: list of size len(phrases) containing coverage (0.0 to 1.0) in best window.
        candidate2position: list of size len(phrases) containing starting position of best window.
    """
    num_phrases = len(phrases)
    candidate2coverage = [0.0] * num_phrases
    candidate2position = [-1] * num_phrases

    for phrase_id, phrase in enumerate(phrases):
        phrase_length = phrase.count(" ") + 1
        total_coverage = np.sum(phrases2positions[phrase_id]) / phrase_length

        # Skip phrases with low overall coverage
        if total_coverage < 0.4:
            continue

        # Calculate initial window sum
        current_sum = np.sum(phrases2positions[phrase_id, :phrase_length])
        max_sum = current_sum
        best_position = 0

        # Slide the window across the positions
        for pos in range(1, phrases2positions.shape[1] - phrase_length + 1):
            current_sum -= phrases2positions[phrase_id, pos - 1]
            current_sum += phrases2positions[phrase_id, pos + phrase_length - 1]
            if current_sum > max_sum:
                max_sum = current_sum
                best_position = pos

        # Calculate coverage with smoothing
        coverage = max_sum / (phrase_length + 2)
        candidate2coverage[phrase_id] = coverage
        candidate2position[phrase_id] = best_position

    return candidate2coverage, candidate2position


def select_candidates(ngram2phrases, phrases, letters, min_phrase_coverage=0.8):
    phrases_to_pos, pos_to_ngram = search_index_for_hits(ngram2phrases, phrases, letters)
    candidates_coverage, candidate_to_pos = find_best_candidate_coverage(phrases, phrases_to_pos)
    symbols_mask = [[0] * (phrase.count(" ") + 1) for phrase in phrases]
    candidates = []
    candidates_processed = 0
    for idx, coverage in sorted(enumerate(candidates_coverage), key=lambda item: item[1], reverse=True):
        start_pos = candidate_to_pos[idx]
        phrase_length = phrases[idx].count(" ") + 1
        for pos in range(start_pos, start_pos + phrase_length):
            if pos >= len(pos_to_ngram):
                break
            for ngram in pos_to_ngram[pos]:
                for phrase_id, begin, size, _ in ngram2phrases[ngram]:
                    if phrase_id == idx:
                        for ppos in range(begin, begin + size):
                            if ppos < phrase_length:
                                symbols_mask[phrase_id][ppos] = 1
        candidates_processed += 1
        if candidates_processed > 100:
            break
        real_coverage = sum(symbols_mask[idx]) / len(symbols_mask[idx])
        if real_coverage >= min_phrase_coverage:
            candidates.append((phrases[idx], start_pos, phrase_length, coverage, real_coverage))

    if len(candidates) == 0:
        print("no candidates", candidates)
        return []

    while len(candidates) < 10:
        random_candidate = random.choice(phrases)
        random_candidate = " ".join(list(random_candidate.replace(" ", "_")))
        candidates.append((random_candidate, -1, random_candidate.count(" ") + 1, 0.0, 0.0))

    candidates = candidates[:10]
    random.shuffle(candidates)
    if len(candidates) != 10:
        print("couldn't get 10 candidates", candidates)
        return []

    return candidates


def read_text_file_in_chunks(file_path, chunk_size=1024):
    with open(file_path, 'r', encoding='utf-8') as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            yield chunk


def split_text_into_70_char_chunks(file_path, max_chunk_size=1024):
    buffer = ''
    for chunk in read_text_file_in_chunks(file_path, max_chunk_size):
        buffer += chunk
        while len(buffer) >= 70:
            # Find the last space within the first 70 characters
            end = buffer.rfind(' ', 0, 70)
            if end == -1:  # No space found, forcibly cut at 70
                end = 70
            yield buffer[:end].strip()
            buffer = buffer[end:].strip()

    if buffer:  # Handle any remaining text in the buffer
        yield buffer


def generate_candidates(input_file, ngram_index_file, user_vocab_file, out_file):
    ngram_index_vocab = load_ngram_index(ngram_index_file, max_misspell_frequency=125000)

    user_phrases = []
    with open(user_vocab_file, 'r', encoding='utf-8') as file:
        for line in file:
            line_clean = clean_text(line)
            user_phrases.append(" ".join(list(line_clean.replace(" ", "_"))))

    phrases, ngram2phrases = index_dictionary(user_phrases, ngram_index_vocab, min_log_prob=-4.0,
                                              max_phrases_per_ngram=600)

    out = open(out_file, "w", encoding="utf-8")
    for i, chunk in enumerate(split_text_into_70_char_chunks(input_file)):
        letters = list(chunk)
        candidates = select_candidates(ngram2phrases, phrases, letters)
        targets = []
        spans = []
        for idx, c in enumerate(candidates):
            candidate, begin_pos, length = c
            if begin_pos == -1:
                continue
            targets.append(str(idx + 1))
            start = begin_pos
            end = min(begin_pos + length, len(letters))
            spans.append(str(start) + " " + str(end))

        input_str = " ".join(letters) + "\t" + ";".join([x[0] for x in candidates]) + "\t" + " ".join(
            targets) + "\t" + ";".join(spans) + "\n"
        out.write(input_str)
    out.close()
