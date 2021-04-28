import random
from typing import List

import torch

from src.amss.amss_desc.abstract import SAM_Training_Dataset, SAM_Eval_Dataset
from src.utils.functions import encode


class MUSDB_AMSS_Training_Dataset(SAM_Training_Dataset):
    def __init__(self, unmixed_dataset, SAEs, vocab, word_to_idx, idx_to_word):
        super().__init__(unmixed_dataset, SAEs)
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

    def __getitem__(self, item):
        before, after, desc = super().__getitem__(item)
        tokens = encode(self.word_to_idx, desc)
        return before, after, tokens, len(tokens)


class MUSDB_AMSS_Validation_Dataset(SAM_Eval_Dataset):
    def __init__(self, unmixed_dataset, SAEs, vocab, word_to_idx, idx_to_word):
        super().__init__(unmixed_dataset, SAEs)
        self.vocab = vocab
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word

    def __getitem__(self, item):
        before, after, desc, track_idx, window_offset = super().__getitem__(item)
        tokens = encode(self.word_to_idx, desc)
        return before, after, tokens, len(tokens), track_idx, window_offset, desc

