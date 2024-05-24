from collections import OrderedDict


class SpellcheckingInput(object):
    def __init__(self, input_ids, input_mask, segment_ids, input_ids_for_subwords, input_mask_for_subwords,
                 segment_ids_for_subwords, character_pos_to_subword_pos, fragment_indices, labels_mask, labels, spans,
                 default_label):

        if not all(len(lst) == len(input_ids) for lst in
                   [input_mask, segment_ids, labels_mask, labels, character_pos_to_subword_pos]):
            raise ValueError(f"All inputs features should have the same length ({len(input_ids)})")

        if not all(len(lst) == len(input_ids_for_subwords) for lst in
                   [input_mask_for_subwords, segment_ids_for_subwords]):
            raise ValueError(f"All input features should have the same length ({len(input_ids_for_subwords)})")

        self.features = OrderedDict([
            ("input_ids", input_ids),
            ("input_mask", input_mask),
            ("segment_ids", segment_ids),
            ("input_ids_for_subwords", input_ids_for_subwords),
            ("input_mask_for_subwords", input_mask_for_subwords),
            ("segment_ids_for_subwords", segment_ids_for_subwords),
            ("character_pos_to_subword_pos", character_pos_to_subword_pos),
            ("fragment_indices", fragment_indices),
            ("labels_mask", labels_mask),
            ("labels", labels),
            ("spans", spans),
        ])
        self._default_label = default_label
