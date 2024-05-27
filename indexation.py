import math
import os
from collections import defaultdict


def load_ngram_mappings(input_name, max_misspelled_freq=1000000000):
    """Loads n-gram mapping vocabularies in form required by dynamic programming
    Args:
        input_name: file with n-gram mappings
        max_misspelled_freq: threshold on misspelled n-gram frequency
    Returns:
        vocab: dict {key=original_ngram, value=dict{key=misspelled_ngram, value=frequency}}
        ban_ngram: set of banned misspelled n-grams

    Input format:
        u t o	u+i t o	49	8145	114
        u t o	<DELETE> t e	63	8145	16970
        u t o	o+_ t o	42	8145	1807
    """
    vocab = defaultdict(dict)
    ban_ngram = set()

    with open(input_name, "r", encoding="utf-8") as f:
        for line in f:
            orig, misspelled, joint_freq, orig_freq, misspelled_freq = line.strip().split("\t")
            if orig == "" or misspelled == "":
                raise ValueError("Empty n-gram: orig=" + orig + "; misspelled=" + misspelled)
            misspelled = misspelled.replace("<DELETE>", "=")
            if misspelled.replace("=", "").strip() == "":  # skip if resulting ngram doesn't contain any real character
                continue
            if int(misspelled_freq) > max_misspelled_freq:
                ban_ngram.add(misspelled + " ")  # space at the end is required within get_index function
            vocab[orig][misspelled] = int(joint_freq) / int(orig_freq)
    return vocab, ban_ngram


def get_index(custom_phrases, vocab, ban_ngram_global, min_log_prob=-4.0, max_phrases_per_ngram=100):
    """Given a restricted vocabulary of replacements,
    loops through custom phrases,
    generates all possible conversions and creates index.

    Args:
        custom_phrases: list of all custom phrases, characters should be split by space,  real space replaced to underscore.
        vocab: n-gram mappings vocabulary - dict {key=original_ngram, value=dict{key=misspelled_ngram, value=frequency}}
        ban_ngram_global: set of banned misspelled n-grams
        min_log_prob: minimum log probability, after which we stop growing this n-gram.
        max_phrases_per_ngram: maximum phrases that we allow to store per one n-gram. N-grams exceeding that quantity get banned.

    Returns:
        phrases - list of phrases. Position in this list is used as phrase_id.
        ngram2phrases - resulting index, i.e. dict where key=ngram, value=list of tuples (phrase_id, begin_pos, size, logprob)
    """

    ban_ngram_local = set()  # these ngrams are banned only for given custom_phrases
    ngram_to_phrase_and_position = defaultdict(list)
    seen_ngram_phrase = set()  # to track duplicates

    for custom_phrase in custom_phrases:
        custom_phrase = custom_phrase.lower()  # convert to lowercase
        inputs = custom_phrase.split(" ")
        begin = 0
        index_keys = [{} for _ in inputs]  # key - letter ngram, index - beginning positions in phrase

        for begin in range(len(inputs)):
            for end in range(begin + 1, min(len(inputs) + 1, begin + 5)):
                inp = " ".join(inputs[begin:end])
                if inp not in vocab:
                    continue
                for rep in vocab[inp]:
                    lp = math.log(vocab[inp][rep])

                    for b in range(max(0, end - 5), end):  # try to grow previous ngrams with new replacement
                        new_ngrams = {}
                        for ngram in index_keys[b]:
                            lp_prev = index_keys[b][ngram]
                            if len(ngram) + len(rep) <= 10 and b + ngram.count(" ") == begin:
                                if lp_prev + lp > min_log_prob:
                                    new_ngrams[ngram + rep + " "] = lp_prev + lp
                        index_keys[b].update(new_ngrams)  # join two dictionaries
                    # add current replacement as ngram
                    if lp > min_log_prob:
                        index_keys[begin][rep + " "] = lp

        for b in range(len(index_keys)):
            for ngram, lp in sorted(index_keys[b].items(), key=lambda item: item[1], reverse=True):
                if ngram in ban_ngram_global:  # here ngram ends with a space
                    continue
                real_length = ngram.count(" ")
                ngram = ngram.replace("+", " ").replace("=", " ")
                ngram = " ".join(ngram.split())  # here ngram doesn't end with a space anymore
                if ngram + " " in ban_ngram_global:  # this can happen after deletion of + and =
                    continue
                if ngram in ban_ngram_local:
                    continue
                ngram_tuple = (ngram, custom_phrase, b, real_length, lp)
                if ngram_tuple not in seen_ngram_phrase:
                    ngram_to_phrase_and_position[ngram].append((custom_phrase, b, real_length, lp))
                    seen_ngram_phrase.add(ngram_tuple)
                if len(ngram_to_phrase_and_position[ngram]) > max_phrases_per_ngram:
                    ban_ngram_local.add(ngram)
                    del ngram_to_phrase_and_position[ngram]
                    continue

    phrases = []  # id to phrase
    phrase2id = {}  # phrase to id
    ngram2phrases = defaultdict(list)  # ngram to list of tuples (phrase_id, begin, length, logprob)

    for ngram in ngram_to_phrase_and_position:
        for phrase, b, length, lp in ngram_to_phrase_and_position[ngram]:
            if phrase not in phrase2id:
                phrases.append(phrase)
                phrase2id[phrase] = len(phrases) - 1
            ngram2phrases[ngram].append((phrase2id[phrase], b, length, lp))

    return phrases, ngram2phrases


def index_dictionary(dict_fname, ngram_index_fname):
    ngram_mapping_vocab, ban_ngram = load_ngram_mappings(ngram_index_fname,
                                                         max_misspelled_freq=125000)
    custom_phrases = []
    with open(dict_fname, "r", encoding="utf-8") as dict_file:
        for phrase in dict_file:
            custom_phrases.append(" ".join(list(phrase.replace(" ", "_"))))
    phrases, ngram2phrases = get_index(custom_phrases, ngram_mapping_vocab, ban_ngram, min_log_prob=-4.0,
                                       max_phrases_per_ngram=600)

    new_filename = dict_fname.replace('_dictionary.txt', '_index.txt')
    with open(new_filename, "w", encoding="utf-8") as out:
        for ngram in ngram2phrases:
            for phrase_id, begin, size, logprob in ngram2phrases[ngram]:
                phrase = phrases[phrase_id]
                out.write(ngram + "\t" + phrase + "\t" + str(begin) + "\t" + str(size) + "\t" + str(logprob) + "\n")


def process_files_in_directory(directory, ngram_index_name):
    """Process all files in the directory that end with '_dictionary.txt'."""
    for filename in os.listdir(directory):
        if filename.endswith('_dictionary.txt'):
            file_path = os.path.join(directory, filename)

            # Apply the dictionary indexation function
            index_dictionary(file_path, ngram_index_name)
            print(f"Indexed {file_path}")


if __name__ == "__main__":
    directory = input("Enter the directory path: ")
    ngram_index = input("Enter the ngram index path: ")
    process_files_in_directory(directory, ngram_index)
