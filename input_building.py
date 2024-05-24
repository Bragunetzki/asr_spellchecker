from os import path

from spellchecking_input import SpellcheckingInput


class InputBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

        # Initialize endings
        self.ids_to_endings, self.endings_to_ids = load_ending_dicts("endings_dict.txt")
        self.ending_pairs_set, self.ending_pairs_dict = load_ending_pairs("endings10.txt")

    def build_input_from_file(self, input_filename, infer):
        if not path.exists(input_filename):
            raise ValueError(f"Cannot find file: {input_filename}")

        inputs = []
        all_hypotheses = []

        with open(input_filename, 'r', encoding="utf-8") as file:
            for line in file:
                if len(inputs) % 100000 == 0:
                    print(f"{len(inputs)} inputs processed.")

                parts = line.rstrip('\n').split('\t')
                hypothesis, ref, target, span_info = parts[0], parts[1], None, None
                if len(parts) == 4:
                    target, span_info = parts[2], parts[3]
                elif len(parts) == 3:
                    target = parts[2]

                try:
                    model_input = self.build_input(hypothesis, ref, target, span_info, infer)
                except Exception as e:
                    print(str(e))
                    continue

                if model_input is None:
                    print('Cannot create example')
                    continue

                if infer:
                    all_hypotheses.append((hypothesis, ref))

                inputs.append(model_input)

        print(f"Done. {len(inputs)} inputs converted.")
        return inputs, all_hypotheses

    def build_input(self, hypothesis, candidate_str, target, span_info, infer):
        if candidate_str.count(";") != 9:
            raise ValueError(f"Expected 10 candidates: {candidate_str}")
        span_info_parts = []
        targets = []

        if target and target != "0":
            span_info_parts = span_info.split(";")
            targets = list(map(int, target.split(" ")))
            if len(span_info_parts) != len(targets):
                raise ValueError(
                    f"len(span_info_parts)={len(span_info_parts)} differs from len(targets)={len(targets)}")

        tags = [0] * len(hypothesis.split())

        if not infer:
            for part, target in zip(span_info_parts, targets):
                c, start, end = map(int, part.split(" "))
                tags[start:end] = [target] * (end - start)

        # get input features for characters
        input_features = self.make_input_features(hypothesis, candidate_str, tags)
        (input_ids, input_mask, segment_ids, labels_mask, labels) = input_features

        # get input features for words
        hyp_with_words = hypothesis.replace(" ", "").replace("_", " ")
        candidates_with_words = candidate_str.replace(" ", "").replace("_", " ")
        subword_features = self.make_input_features(hyp_with_words, candidates_with_words, tags=None)
        input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords = subword_features[:3]

        # map characters to subwords
        character_pos_to_subword_pos = self.append_chars_to_subwords(input_ids, input_ids_for_subwords)

        fragment_indices = []
        if infer:
            fragment_indices = self.make_fragment_indices(hypothesis, targets, span_info_parts)
        spans = []
        if not infer:
            spans = self.make_spans(span_info_parts)

        if len(input_ids) > self.cfg['max_seq_len'] or len(spans) > self.cfg['max_spans']:
            print(
                "Max len exceeded: len(input_ids)=",
                len(input_ids),
                "; _max_seq_length=",
                self.cfg['max_seq_len'],
                "; len(spans)=",
                len(spans),
                "; _max_spans_length=",
                self.cfg['max_spans'],
            )
            return None

        # change the targets to account for ending classes
        labels, labels_mask = self.alter_labels_with_endings(labels, labels_mask, input_ids, segment_ids)
        model_input = SpellcheckingInput(
            input_ids, input_mask, segment_ids,
            input_ids_for_subwords, input_mask_for_subwords, segment_ids_for_subwords,
            character_pos_to_subword_pos, fragment_indices,
            labels_mask, labels, spans, default_label=0
        )
        return model_input

    def tokenize_hypothesis_with_labels(self, hypothesis, labels):
        tokens = hypothesis.split()
        bert_tokens = []
        bert_labels = []

        for i, token in enumerate(tokens):
            token_parts = self.cfg['tokenizer'].tokenize(token)
            bert_tokens.extend(token_parts)
            bert_labels.extend([labels[i]] * len(token_parts))
        return bert_tokens, bert_labels

    def tokenize_string(self, sequence):
        tokens = sequence.split()
        bert_tokens = []
        for i, token in enumerate(tokens):
            pieces = self.cfg['tokenizer'].tokenize(token)
            bert_tokens.extend(pieces)
        return bert_tokens

    def make_fragment_indices(self, hyp, targets, span_info_parts):
        fragment_indices = []
        letters = hyp.split()

        for target, span_info in zip(targets, span_info_parts):
            _, start, end = span_info.split(" ")
            start = int(start)
            end = min(int(end), len(hyp))

            # Expand both sides to the nearest space.
            adjusted_start = start
            while adjusted_start > 0 and letters[adjusted_start] != '_':
                adjusted_start -= 1
            if adjusted_start != 0:
                adjusted_start += 1

            adjusted_end = end
            while adjusted_end < len(letters) and letters[adjusted_end] != '_':
                adjusted_end += 1
            fragment_indices.append((adjusted_start + 1, adjusted_end + 1, target))

            # Shrink to the closest space.
            expanded_fragment = "".join(letters[adjusted_start:adjusted_end])
            left_space_position = expanded_fragment.find("_")
            right_space_position = expanded_fragment.rfind("_")
            left_shrinked = False
            right_shrinked = False

            if 0 <= left_space_position < len(expanded_fragment) / 2:
                fragment_indices.append((adjusted_start + left_space_position + 2, adjusted_end + 1, target))
                left_shrinked = True
            if len(expanded_fragment) / 2 <= right_space_position and right_space_position >= 0:
                fragment_indices.append((adjusted_start + 1, adjusted_start + right_space_position + 1, target))
                right_shrinked = True
            if left_shrinked and right_shrinked:
                fragment_indices.append(
                    (adjusted_start + left_space_position + 2, adjusted_start + right_space_position + 1, target))

        return fragment_indices

    def make_input_features(self, hypothesis, candidates_str, tags):
        labels_mask = []
        labels = []
        if tags is None:
            hyp_tokens = self.tokenize_string(hypothesis)
        else:
            hyp_tokens, labels = self.tokenize_hypothesis_with_labels(hypothesis, tags)

        candidates = candidates_str.split(";")
        all_candidate_tokens = []
        all_candidate_segment_ids = []
        for i, candidate in enumerate(candidates):
            candidate_tokens = self.tokenize_string(candidate)
            all_candidate_tokens.extend(candidate_tokens + ["[SEP]"])
            all_candidate_segment_ids.extend([i + 1] * (len(candidate_tokens) + 1))

        input_tokens = ["[CLS]"] + hyp_tokens + ["[SEP]"] + all_candidate_tokens
        input_ids = self.cfg['tokenizer'].convert_tokens_to_ids(input_tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * (len(hyp_tokens) + 2) + all_candidate_segment_ids
        if len(input_ids) != len(segment_ids):
            raise ValueError(f"len(input_ids)={len(input_ids)} differs from len(segment_ids)={len(segment_ids)}")

        if tags:
            labels_mask = [0] + [1] * len(labels) + [0] * (len(all_candidate_tokens) + 1)
            labels = [0] + labels + [0] * (len(all_candidate_tokens) + 1)
        return input_ids, input_mask, segment_ids, labels_mask, labels

    def append_chars_to_subwords(self, input_ids, input_ids_for_subwords):
        character_pos_to_subword_pos = [0] * len(input_ids)
        char_tokens = self.cfg['tokenizer'].convert_ids_to_tokens(input_ids)
        subword_tokens = self.cfg['tokenizer'].convert_ids_to_tokens(input_ids_for_subwords)

        subword_index = 0
        char_index_in_subword = 0

        for i, char in enumerate(char_tokens):
            subword = subword_tokens[subword_index]
            if char == "[CLS]" and subword == "[CLS]":
                character_pos_to_subword_pos[i] = subword_index
                subword_index += 1
                continue
            if char == "[SEP]" and subword == "[SEP]":
                character_pos_to_subword_pos[i] = subword_index
                subword_index += 1
                continue
            if char == "[CLS]" or char == "[SEP]" or subword == "[CLS]" or subword == "[SEP]":
                raise IndexError(f"Unexpected token alignment at char[{i}]={char}; subword[{subword_index}]={subword}")

            if char == "_":
                character_pos_to_subword_pos[i] = subword_index - 1
                continue

            if char_index_in_subword < len(subword):
                if char == subword[char_index_in_subword]:
                    character_pos_to_subword_pos[i] = subword_index
                    char_index_in_subword += 1
                else:
                    raise IndexError(
                        f"Character mismatch at i={i}, subword_index={subword_index}, char_index_in_subword={char_index_in_subword}"
                    )

            if char_index_in_subword >= len(subword):
                subword_index += 1
                char_index_in_subword = 0
                if subword_index >= len(subword_tokens):
                    break
                if subword_tokens[subword_index].startswith("##"):
                    char_index_in_subword = 2

        if subword_index < len(subword_tokens):
            raise IndexError(
                f"Unprocessed subwords remaining at subword_index={subword_index}, len(subword_tokens)={len(subword_tokens)}"
            )

        return character_pos_to_subword_pos

    def make_spans(self, span_info_parts):
        spans = []

        for span_info in span_info_parts:
            if span_info == "":
                break
            _, start, end = span_info.split(" ")
            start = int(start) + 1
            end = int(end) + 1
            spans.append((start, end))
        return spans

    def alter_labels_with_endings(self, labels, labels_mask, input_ids, segment_ids):
        tokenizer = self.cfg['tokenizer']
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        for begin in range(len(tokens)):
            for end in range(min(begin + 1, len(tokens)), min(begin + 11, len(tokens))):
                char_fragment = tokens[begin:end]
                label_fragment = labels[begin:end]
                # skip if ending covers fragments that shouldn't be predicted
                if 0 in label_fragment:
                    continue
                # skip if ending covers mor than a single prediction target
                if not all(x == label_fragment[0] for x in label_fragment):
                    continue
                # skip if next token belongs to the same prediction (not really an ending)
                if end < len(tokens) and labels[end] == label_fragment[0]:
                    continue

                # if found a viable ending, look for paired ending in corresponding candidate
                ending_str = "".join(char_fragment)
                if ending_str in self.endings_to_ids:
                    pairs = self.ending_pairs_dict.get(ending_str, [])
                    candidate_num = label_fragment[0]
                    candidate_start = -1
                    candidate_end = -1
                    for index, idx in enumerate(segment_ids):
                        if idx == candidate_num:
                            # if first viable id found, it is start of candidate
                            if candidate_start == -1:
                                candidate_start = index
                        # if a different id is found after start was established, this is end of candidate (exclusive)
                        elif candidate_start != -1:
                            candidate_end = index
                            break
                    if candidate_start != -1 and candidate_end == -1:
                        candidate_end = len(segment_ids)
                    candidate_tokens = tokens[candidate_start: candidate_end - 1]
                    candidate_str = "".join(candidate_tokens)

                    # candidate is found, now look for paired endings within it.
                    for ending in pairs:
                        if candidate_str.endswith(ending):
                            ending_str2 = ending
                            ending_id1 = self.endings_to_ids[ending_str]
                            labels[begin:end] = [ending_id1] * (end - begin)
                            labels_mask[candidate_end - len(ending_str2):candidate_end] = [1] * len(ending_str2)
                            break
        return labels, labels_mask


def load_ending_dicts(filename):
    ids_to_endings = dict()
    endings_to_ids = dict()
    with open(filename, "r", encoding="utf-8") as input_file:
        for line in input_file:
            idx, ending = line.rstrip("\n").split("\t")
            ids_to_endings[idx] = ending
            endings_to_ids[ending] = idx
    return ids_to_endings, endings_to_ids


def load_ending_pairs(filename):
    pairs_set = set()
    pairs_dict = dict()
    with open(filename, "r", encoding="utf-8") as input_file:
        for line in input_file:
            ending1, ending2, freq = line.rstrip("\n").split("\t")
            pairs_set.add((ending1, ending2))
            if ending1 not in pairs_dict:
                pairs_dict[ending1] = []
            if ending2 not in pairs_dict:
                pairs_dict[ending2] = []
            pairs_dict[ending1].append(ending2)
            pairs_dict[ending2].append(ending1)
    return pairs_set, pairs_dict
