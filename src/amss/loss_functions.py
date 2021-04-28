import math
from abc import ABC, abstractmethod
from argparse import ArgumentParser

import torch
import torch.nn.functional as f


def add_model_specific_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--loss_raw_scale', type=float, default=None)
    parser.add_argument('--loss_spec_scale', type=float, default=None)
    parser.add_argument('--loss_raw_mode', type=str, default=None)
    parser.add_argument('--loss_spec_mode', type=str, default=None)

    return parser


def get_sam_loss(loss_name, window_length, hop_length, **kwargs):
    assert loss_name in ['spec_l1', 'spec_l2', 'spec_mse',
                         'raw_l1', 'raw_l2', 'raw_mse',
                         'sdr_like', 'sdr', 'dsr', 'nsdr', 'ldsr',
                         'nsdr_44100',
                         'raw_mse', 'distortion', 'kl_lr', 'raw_and_spec',
                         'ncs', 'ncs_44100', 'nlcs']

    if loss_name == 'spec_l1':
        return SAM_Spectrogram_Loss('l1')
    elif loss_name == 'spec_l2':
        return SAM_Spectrogram_Loss('l2')
    elif loss_name == 'spec_mse':
        return SAM_Spectrogram_Loss('mse')

    elif loss_name == 'raw_l1':
        return SAM_RAW_Loss('l1')
    elif loss_name == 'raw_l2':
        return SAM_RAW_Loss('l2')
    elif loss_name == 'raw_mse':
        return SAM_RAW_Loss('mse')

    elif loss_name == 'raw_and_spec':
        return SAM_Raw_And_Spec(**kwargs)

    elif loss_name == 'distortion':
        return SAM_Distortion_Loss(window_length, hop_length)
    elif loss_name == 'sdr_like':
        return SAM_SDR_LIKE_Loss(window_length, hop_length)
    elif loss_name == 'sdr':
        return SAM_SDR(window_length, hop_length)
    elif loss_name == 'dsr':
        return SAM_DSR(window_length, hop_length)
    elif loss_name == 'nsdr':
        return SAM_NSDR(window_length, hop_length)
    elif loss_name == 'nsdr_44100':
        return SAM_NSDR(44100, 22050)
    elif loss_name == 'ldsr':
        return SAM_LDSR(window_length, hop_length)
    elif loss_name == 'ncs':
        return SAM_NCS_Loss(window_length, hop_length)
    elif loss_name == 'ncs_44100':
        return SAM_NCS_Loss(44100, 22050)
    elif loss_name == 'nlcs':
        return SAM_NLCS_Loss(window_length, hop_length)

    else:
        raise ModuleNotFoundError


class SAM_Loss(ABC):

    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def compute(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class SAM_Spectrogram_Loss(SAM_Loss):

    def __init__(self, mode, **kwargs):
        super().__init__()
        assert mode in ['l1', 'l2', 'mse']
        self.criterion = f.l1_loss if mode == 'l1' else f.mse_loss

    def compute(self, model, before, after, desc, length):
        target = model.to_spec(after)
        target_hat = model.forward(before, desc, length )
        return self.criterion(target_hat, target)


class SAM_RAW_Loss(SAM_Loss):

    def __init__(self, mode, **kwargs):
        super().__init__()
        assert mode in ['l1', 'l2', 'mse']
        self.criterion = f.l1_loss if mode == 'l1' else f.mse_loss

    def compute(self, model, before, after, tokens, token_lengths):
        target_signal_hat = model.manipulate(before, tokens, token_lengths)
        return self.criterion(target_signal_hat, after)


class SAM_Raw_And_Spec(SAM_Loss):

    def __init__(self, loss_raw_scale, loss_raw_mode, loss_spec_scale, loss_spec_mode, **kwargs):
        assert (None not in [loss_raw_scale, loss_raw_mode, loss_spec_scale, loss_spec_mode])  # please add arguments
        super().__init__()
        self.raw_scale = float(loss_raw_scale)
        assert loss_raw_mode in ['l1', 'l2', 'mse']
        self.raw_criterion = f.l1_loss if loss_raw_mode == 'l1' else f.mse_loss

        self.spec_scale = float(loss_spec_scale)
        assert loss_spec_mode in ['l1', 'l2', 'mse']
        self.spec_criterion = f.l1_loss if loss_spec_mode == 'l1' else f.mse_loss

    def compute(self, model, before, after, desc, length):
        # ground_truth spec
        after_spec = model.to_spec(after)
        # raw and spec
        target_signal_hat, target_spec_hat = model.manipulate_with_spec(before,  desc, length)

        raw_loss = self.raw_criterion(target_signal_hat, after)
        spec_loss = self.spec_criterion(target_spec_hat, after_spec)
        return self.raw_scale * raw_loss + self.spec_scale * spec_loss


class SAM_Unfolding_Loss(SAM_Loss, ABC):

    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__()
        self.window_length = window_length
        self.hop_length = hop_length

    def compute(self, model, mixture_signal, condition, target_signal):
        target_signal_hat = model.manipulate(mixture_signal, condition, )
        target_signal_hat = self.auto_pad(target_signal_hat)
        target_signal_hat = target_signal_hat.unfold(-2, self.window_length, self.hop_length)
        target_signal = self.auto_pad(target_signal)
        target_signal = target_signal.unfold(-2, self.window_length, self.hop_length)
        return self.criterion(target_signal_hat, target_signal)

    def auto_pad(self, signal):
        n_step = math.floor(signal.shape[-2] / self.hop_length)
        padding = self.hop_length * n_step + self.window_length - signal.shape[-2]
        left_padding = padding // 2
        right_padding = padding // 2 + padding % 1
        return f.pad(signal, (0, 0, left_padding, right_padding))

    @abstractmethod
    def criterion(self, target_signal_hat, target_signal):
        pass


class SAM_Distortion_Loss(SAM_Unfolding_Loss):

    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        s_target = (
                           ((target_signal_hat * target_signal).sum(-1, keepdims=True) + 1e-8) /
                           ((target_signal ** 2).sum(axis=-1, keepdims=True) + 1e-8)
                   ) * target_signal

        distortion = target_signal_hat - s_target

        return torch.norm(distortion, dim=-1).sum(-2).mean()


class SAM_SDR_LIKE_Loss(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        s_target = (
                           ((target_signal_hat * target_signal).sum(-1, keepdims=True) + 1e-8) /
                           ((target_signal ** 2).sum(axis=-1, keepdims=True) + 1e-8)
                   ) * target_signal

        distortion = target_signal_hat - s_target

        loss = ((distortion ** 2).sum(-1) + 1e-8) - ((s_target ** 2).sum(-1) + 1e-8)

        return loss.sum(-2).mean()


class SAM_SDR(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        s_target = (
                           ((target_signal_hat * target_signal).sum(-1, keepdims=True) + 1e-8) /
                           ((target_signal ** 2).sum(axis=-1, keepdims=True) + 1e-8)
                   ) * target_signal

        distortion = target_signal_hat - s_target

        loss = ((s_target ** 2).sum(-1) + 1e-8) / ((distortion ** 2).sum(-1) + 1e-8)

        return loss.mean()


class SAM_DSR(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        s_target = (
                           ((target_signal_hat * target_signal).sum(-1, keepdims=True) + 1e-8) /
                           ((target_signal ** 2).sum(axis=-1, keepdims=True) + 1e-8)
                   ) * target_signal

        distortion = target_signal_hat - s_target

        loss = ((distortion ** 2).sum(-1) + 1e-8) / ((s_target ** 2).sum(-1) + 1e-8)

        return loss.mean()


class SAM_LDSR(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        s_target = (
                           ((target_signal_hat * target_signal).sum(-1, keepdims=True) + 1e-8) /
                           ((target_signal ** 2).sum(axis=-1, keepdims=True) + 1e-8)
                   ) * target_signal

        distortion = target_signal_hat - s_target

        loss = ((distortion ** 2).sum(-1) + 1e-8).log() - ((s_target ** 2).sum(-1) + 1e-8).log()

        return loss.mean()


class SAM_NSDR(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        s_target = (
                           ((target_signal_hat * target_signal).sum(-1, keepdims=True) + 1e-8) /
                           ((target_signal ** 2).sum(axis=-1, keepdims=True) + 1e-8)
                   ) * target_signal

        distortion = target_signal_hat - s_target

        loss = -((s_target ** 2).sum(-1) + 1e-8) / ((distortion ** 2).sum(-1) + 1e-8)

        return loss.mean()


class SAM_NCS_Loss(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        return -f.cosine_similarity(target_signal_hat, target_signal, dim=-1).mean()


class SAM_NLCS_Loss(SAM_Unfolding_Loss):
    def __init__(self, window_length, hop_length, **kwargs):
        super().__init__(window_length, hop_length)

    def criterion(self, target_signal_hat, target_signal):
        cs = f.cosine_similarity(target_signal_hat, target_signal, dim=-1)
        return - (cs + 1 + 1e-8).log().mean()
