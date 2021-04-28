from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from src.data.musdb_amss_dataset.amss_task1_datasets import Task1_Validation_Dataset, Task1_Training_Dataset
from src.data.musdb_amss_dataset.amss_task2_datasets import Task2_Training_Dataset, Task2_Validation_Dataset
from src.data.musdb_wrapper import MusdbUnmixedTrainSet, MusdbUnmixedValidSet


def pad(seq, max_len):
    return torch.nn.functional.pad(seq, [0, max_len - len(seq)])


def collate_fn_train(batch):
    seq_len = [item[-1] for item in batch]
    max_len = max(seq_len)

    idx_lens = list(enumerate(seq_len))
    idx_lens.sort(key=lambda x: x[1], reverse=True)

    reordered_batch = [batch[idx] for idx, length in idx_lens]

    output_tracks = [torch.stack([item[idx] for item in reordered_batch]) for idx in [0, 1]]
    output_tokens = [item[2] for item in reordered_batch]
    output_tokens = [torch.stack([pad(token, max_len) for token in output_tokens])]

    others = [[item[idx] for item in reordered_batch] for idx in [3]]

    return output_tracks + output_tokens + others

def collate_fn_train_for_task4(batch):
    seq_len = [item[5] for item in batch]
    max_len = max(seq_len)

    idx_lens = list(enumerate(seq_len))
    idx_lens.sort(key=lambda x: x[1], reverse=True)

    reordered_batch = [batch[idx] for idx, length in idx_lens]

    output_tracks = [torch.stack([item[idx] for item in reordered_batch]) for idx in [0, 1, 2]]
    output_tokens_1 = [item[3] for item in reordered_batch]
    output_tokens_1 = [torch.stack([pad(token, max_len) for token in output_tokens_1])]

    output_tokens_2 = [item[4] for item in reordered_batch]
    output_tokens_2 = [torch.stack(output_tokens_2)]

    others = [[item[idx] for item in reordered_batch] for idx in [5,6]]

    return output_tracks + output_tokens_1 + output_tokens_2 + others


def collate_fn_eval(batch):
    # before, after, tokens, len(tokens), track_idx, window_offset, desc

    seq_len = [item[3] for item in batch]
    max_len = max(seq_len)

    idx_lens = list(enumerate(seq_len))
    idx_lens.sort(key=lambda x: x[1], reverse=True)

    reordered_batch = [batch[idx] for idx, length in idx_lens]

    output_tracks = [torch.stack([item[idx] for item in reordered_batch]) for idx in [0, 1]]
    output_tokens = [item[2] for item in reordered_batch]
    output_tokens = [torch.stack([pad(token, max_len) for token in output_tokens])]

    others = [[item[idx] for item in reordered_batch] for idx in [3, 4, 5, 6]]

    return output_tracks + output_tokens + others

def collate_fn_eval_for_task4(batch):
    # before, after, tokens, len(tokens), track_idx, window_offset, desc

    seq_len = [item[5] for item in batch]
    max_len = max(seq_len)

    idx_lens = list(enumerate(seq_len))
    idx_lens.sort(key=lambda x: x[1], reverse=True)

    reordered_batch = [batch[idx] for idx, length in idx_lens]

    output_tracks = [torch.stack([item[idx] for item in reordered_batch]) for idx in [0, 1, 2]]
    output_tokens_1 = [item[3] for item in reordered_batch]
    output_tokens_1 = [torch.stack([pad(token, max_len) for token in output_tokens_1])]

    output_tokens_2 = [item[4] for item in reordered_batch]
    output_tokens_2 = [torch.stack(output_tokens_2)]

    others = [[item[idx] for item in reordered_batch] for idx in [5, 6, 7, 8, 9]]

    return output_tracks + output_tokens_1 + output_tokens_2+ others


class DataProvider(object):

    @staticmethod
    def add_data_provider_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--musdb_root', type=str, default='etc/musdb18_samples_wav/')
        parser.add_argument('--task', type=str, default='task1')

        return parser

    def __init__(self, musdb_root, batch_size, num_workers, pin_memory, task, n_fft, hop_length, num_frame):

        self.musdb_root = musdb_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.task = task
        self.num_frame = num_frame
        self.hop_length = hop_length
        self.n_fft = n_fft

        assert task in ['task1', 'task2', 'task2_pretraining', 'task3', 'task3_pretraining', 'task4', 'task4_pretraining', 'task5']

    # def get_dataset_for_training_and_validation(self):
    #
    #     if self.task == 'task1':
    #
    #         unmixed_set = MusdbUnmixedTrainSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)
    #         training_set = Task1_Training_Dataset(unmixed_set)
    #         unmixed_set = MusdbUnmixedValidSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)
    #         validation_set = Task1_Validation_Dataset(unmixed_set)
    #         return training_set, validation_set
    #     else:
    #         raise ModuleNotFoundError

    def get_training_dataset_and_loader(self):

        unmixed_set = MusdbUnmixedTrainSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        if self.task == 'task1':
            training_set = Task1_Training_Dataset(unmixed_set)
        elif self.task == 'task2':
            training_set = Task2_Training_Dataset(unmixed_set)

        else:
            raise ModuleNotFoundError

        loader = DataLoader(training_set, shuffle=True, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=collate_fn_train)

        return training_set, loader

    def get_validation_dataset_and_loader(self):

        unmixed_set = MusdbUnmixedValidSet(self.musdb_root, self.n_fft, self.hop_length, self.num_frame)

        if self.task == 'task1':
            validation_set = Task1_Validation_Dataset(unmixed_set)
        elif self.task == 'task2':
            validation_set = Task2_Validation_Dataset(unmixed_set)


        else:
            raise ModuleNotFoundError

        loader = DataLoader(validation_set, shuffle=False, batch_size=self.batch_size,
                            num_workers=self.num_workers,
                            pin_memory=self.pin_memory,
                            collate_fn=collate_fn_eval)

        return validation_set, loader
