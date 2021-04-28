from pytorch_lightning.utilities import rank_zero_warn
from warnings import warn
import numpy as np
import torch
import torch.nn as nn
import re


def get_activation_by_name(activation):
    if activation == "leaky_relu":
        return nn.LeakyReLU
    elif activation == "relu":
        return nn.ReLU
    elif activation == "sigmoid":
        return nn.Sigmoid
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "softmax":
        return nn.Softmax
    elif activation == "identity":
        return nn.Identity
    else:
        return None


def get_optimizer_by_name(optimizer):
    if optimizer == "adam":
        return torch.optim.Adam
    elif optimizer == "adagrad":
        return torch.optim.Adagrad
    elif optimizer == "sgd":
        return torch.optim.SGD
    elif optimizer == "rmsprop":
        return torch.optim.RMSprop
    else:
        return torch.optim.Adam


def string_to_tuple(kernel_size):
    kernel_size_ = kernel_size.strip().replace('(', '').replace(')', '').split(',')
    kernel_size_ = [int(kernel) for kernel in kernel_size_]
    return kernel_size_


def string_to_list(int_list):
    int_list_ = int_list.strip().replace('[', '').replace(']', '').split(',')
    int_list_ = [int(v) for v in int_list_]
    return int_list_


def mkdir_if_not_exists(default_save_path):
    if not os.path.exists(default_save_path):
        os.mkdir(default_save_path)


def get_estimation(idx, target_name, estimation_dict):
    estimated = estimation_dict[target_name][idx]
    if len(estimated) == 0:
        warn('TODO: zero estimation, caused by ddp')
        return None
    estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
    return estimated


def flat_word_set(word_set):
    return [subword for word in word_set for subword in word.split(' ')]


def build_vocab(amms_set):
    vocab = [word for amss in amms_set for word in re.split(r'(,|<pad>|<bgn>|<end>|\W)', amss.gen_vocab_string())
             if word not in ['', ' ']]
    vocab = vocab + ['<bgn>', '<end>']  # just in case
    vocab = set(vocab)
    return ['<pad>'] + sorted(vocab)


def encode(word_to_idx, desc):
    tokens = [word_to_idx[word] for word in re.split(r'(,|<pad>|<bgn>|<end>|\W)', desc) if word not in ['', ' ']]
    return torch.tensor(tokens, dtype=torch.long)


def decode(idx_to_word, seq):
    if isinstance(seq, torch.Tensor):
        if len(seq.shape) == 1:
            seq = seq.cpu().detach().numpy()
        else:
            raise NotImplemented

    if isinstance(seq, np.ndarray):
        if len(seq.shape) > 1:
            raise NotImplemented

    return ' '.join([idx_to_word[idx] for idx in seq]).replace(' ,', ',')


def build_vocab_dict(vocab):
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {word_to_idx[idx]: idx for idx in word_to_idx.keys()}

    return word_to_idx, idx_to_word


def swsa_reconstruct(B, F, T, att, v):
    res = torch.matmul(att, v.transpose(-1, -2))  # B, h, T, IL, F
    res = res.transpose(-2, -3)  # B, h, IL, T, F
    res = res.reshape(B, -1, T, F)
    return res


def multhead_channel_wise_skip_attention(B, T, num_head, f_sqrt, q, k, v):
    q, k, v = [elem.view(B, num_head, -1, T, elem.shape[-1]) for elem in [q, k, v]]
    q = q.transpose(-2, -3)  # B, h, T, ch//h, F
    k = k.permute(0, 1, 3, 4, 2)  # B, h, T, F, IL//h
    v = v.permute(0, 1, 3, 4, 2)  # B, h, T, F, IL//h
    att = torch.matmul(q, k)  # B, h, T, IL, ch
    att = (att / f_sqrt).softmax(-1)
    return att, v  # [B, h, T, IL, ch] , [B, h, T, F, IL//h]


def extract_bidirectional_context_from_packedseq(length_seq, rnn_output, rnn_hidden_size):
    indexes = list(range(len(length_seq)))
    left2right = rnn_output[indexes, (length_seq - 1).cpu().detach().numpy(), :rnn_hidden_size]
    right2left = rnn_output[:, 0, rnn_hidden_size:]
    return torch.cat([left2right, right2left], dim=-1)


def extract_bidirectional_context(rnn_output, rnn_hidden_size):
    left2right = rnn_output[:, -1, :rnn_hidden_size]
    right2left = rnn_output[:, 0, rnn_hidden_size:]
    return torch.cat([left2right, right2left], dim=-1)


def normalize(audio_a, audio_b):
    max_scale = max(audio_a.max(), audio_b.max())
    max_scale = 1 if max_scale < 1 else max_scale
    audio_a, audio_b = audio_a / max_scale, audio_b / max_scale
    return audio_a, audio_b


def apply_mask_for_attention(desc_att, length):
    # mul_mask = torch.ones_like(desc_att, requires_grad=False)
    # add_mask = torch.zeros_like(desc_att, requires_grad=False)
    # for i, l in enumerate(length):
    #     mul_mask[i, :, l:] *= 0
    #     add_mask[i, :, l:] -= 2 ** 10
    # desc_att = mul_mask * desc_att
    # desc_att = desc_att + add_mask
    for i, l in enumerate(length):
        desc_att[i, :, l:] = float('-inf')
    return desc_att


import os
from typing import Dict, Any
import yaml


def load_hparams_from_yaml(config_yaml: str) -> Dict[str, Any]:
    if not os.path.isfile(config_yaml):
        rank_zero_warn(f'Missing Tags: {config_yaml}.', RuntimeWarning)
        return {}

    with open(config_yaml) as fp:
        tags = yaml.load(fp, Loader=yaml.SafeLoader)

    return tags


