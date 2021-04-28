import abc
import random
from abc import abstractmethod
from typing import List

import torch
from torch.utils.data import Dataset


class Selective_Audio_Editing(object, metaclass=abc.ABCMeta):

    @abstractmethod
    def edit(self, unmixed_track):
        raise NotImplementedError

    @abstractmethod
    def gen_desc(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def gen_desc_default(self) -> str:
        raise NotImplementedError

    def gen_vocab_string(self):
        return '<bgn> ' + ' '.join(self.descriptions) + ' <end>'

    def gen_formatted_desc_default(self):
        return '<bgn> ' + self.gen_desc_default() + ' <end>'


class SAM_Dataset(Dataset):
    def __init__(self, unmixed_dataset, SAEs: list):
        self.unmixed_dataset = unmixed_dataset  # iterator
        self.SAEs = SAEs
        self.num_TGE = len(self.SAEs)

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, i):
        pass


class SAM_Training_Dataset(SAM_Dataset):

    def __init__(self, unmixed_dataset, SAEs):
        super().__init__(unmixed_dataset, SAEs)

    def __len__(self):
        return len(self.unmixed_dataset)

    def __getitem__(self, i):
        unmixed_track = self.unmixed_dataset[i]
        idx = random.randint(0, self.num_TGE - 1)
        tge = self.SAEs[idx]
        text, before, after = tge.edit(unmixed_track)

        return torch.from_numpy(before), torch.from_numpy(after), text


class SAM_Eval_Dataset(SAM_Dataset):

    def __init__(self, unmixed_dataset, SAEs: List[Selective_Audio_Editing]):
        super().__init__(unmixed_dataset, SAEs)

    def __len__(self):
        return len(self.unmixed_dataset) * self.num_TGE

    def __getitem__(self, i):
        track_idx = i // self.num_TGE
        tge_idx = i % self.num_TGE

        unmixed_track, track_idx, window_offset = self.unmixed_dataset[track_idx]
        tge = self.SAEs[tge_idx]

        desc, before, after = tge.edit_with_default_desc(unmixed_track=unmixed_track)
        before, after = torch.from_numpy(before), torch.from_numpy(after)
        return before, after, desc, track_idx, window_offset
