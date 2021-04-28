from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from typing import List, Union, Any, Tuple

import numpy as np
import pydub
import pytorch_lightning as pl
import soundfile
import torch
from torch.nn.functional import mse_loss, l1_loss
import wandb

from pytorch_lightning.loggers import WandbLogger

from src.amss import loss_functions
from src.utils import fourier
from src.utils.fourier import get_trim_length
from src.utils.functions import get_optimizer_by_name, get_estimation
from src.utils.weight_initialization import init_weights_functional


class Selective_Audio_Manipulation(pl.LightningModule, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--optimizer', type=str, default='adam')

        return loss_functions.add_model_specific_args(parser)

    def __init__(self, n_fft, hop_length, num_frame, optimizer, lr):
        super(Selective_Audio_Manipulation, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trim_length = get_trim_length(self.hop_length)
        self.n_trim_frames = self.trim_length // self.hop_length
        self.num_frame = num_frame

        self.lr = lr
        self.optimizer = optimizer

        self.target_names = ['vocals', 'drums', 'bass', 'other']

    def configure_optimizers(self):
        optimizer = get_optimizer_by_name(self.optimizer)
        return optimizer(self.parameters(), lr=float(self.lr))

    @abstractmethod
    def training_step(self, batch, batch_idx):
        pass

    def on_test_epoch_start(self):

        import os
        output_folder = 'output'
        if os.path.exists(output_folder):
            os.rmdir(output_folder)
        os.mkdir(output_folder)  # TODO

        self.valid_estimation_dict = None
        self.test_estimation_dict = {}

        self.musdb_test = self.test_dataloader().dataset
        num_tracks = self.musdb_test.unmixed_dataset.num_tracks
        for sae in self.musdb_test.SAEs:
            self.test_estimation_dict[sae.gen_text_default()] = {mixture_idx: {}
                                                                 for mixture_idx
                                                                 in range(num_tracks)}

    def test_step(self, batch, batch_idx):
        mixtures, mixture_ids, window_offsets, input_conditions, target_names = batch

        estimated_targets = self.manipulate(mixtures, input_conditions)[:, self.trim_length:-self.trim_length]

        for mixture, mixture_idx, window_offset, input_condition, target_name, estimated_target \
                in zip(mixtures, mixture_ids, window_offsets, input_conditions, target_names, estimated_targets):
            self.test_estimation_dict[target_name][mixture_idx.item()][
                window_offset.item()] = estimated_target.detach().cpu().numpy()

        return torch.zeros(0)

    def on_test_epoch_end(self):

        import museval
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')
        idx_list = range(self.musdb_test.num_tracks)

        for idx in idx_list:
            estimation = {}
            for target_name in self.target_names:
                estimation[target_name] = get_estimation(idx, target_name, self.test_estimation_dict)
                if estimation[target_name] is not None:
                    estimation[target_name] = estimation[target_name].astype(np.float32)

            # Real SDR
            if len(estimation) == len(self.target_names):
                track_length = self.musdb_test.musdb_test[idx].samples
                estimated_targets = [estimation[target_name][:track_length] for target_name in self.target_names]

                if track_length > estimated_targets[0].shape[0]:
                    raise NotImplementedError
                else:
                    estimated_targets_dict = {target_name: estimation[target_name][:track_length] for target_name in
                                              self.target_names}
                    track_score = museval.eval_mus_track(
                        self.musdb_test.musdb_test[idx],
                        estimated_targets_dict
                    )

                    score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                        ['target', 'metric'])['score'] \
                        .median().to_dict()

                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log(
                            {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})

                    else:
                        print(track_score)

                    results.add_track(track_score)

            if idx == 1 and isinstance(self.logger, WandbLogger):
                self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, target_name): [
                    wandb.Audio(estimation[target_name], caption='{}_{}'.format(idx, target_name), sample_rate=44100)]})

        if isinstance(self.logger, WandbLogger):

            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            self.logger.experiment.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )
        else:
            print(results)

    def export_mp3(self, idx, target_name):
        estimated = self.test_estimation_dict[target_name][idx]
        estimated = np.concatenate([estimated[key] for key in sorted(estimated.keys())], axis=0)
        soundfile.write('tmp_output.wav', estimated, samplerate=44100)
        audio = pydub.AudioSegment.from_wav('tmp_output.wav')
        audio.export('{}_estimated/output_{}.mp3'.format(idx, target_name))

    @abstractmethod
    def forward(self, before, desc, length) -> torch.Tensor:
        pass

    @abstractmethod
    def manipulate(self, before, tokens, token_lengths) -> torch.Tensor:
        pass

    @abstractmethod
    def init_weights(self):
        pass


class Spectrogram_based(Selective_Audio_Manipulation, metaclass=ABCMeta):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--n_fft', type=int, default=2048)
        parser.add_argument('--hop_length', type=int, default=1024)
        parser.add_argument('--num_frame', type=int, default=128)
        parser.add_argument('--spec_type', type=str, default='complex')
        parser.add_argument('--spec_est_mode', type=str, default='mapping')

        parser.add_argument('--train_loss', type=str, default='spec_mse')
        parser.add_argument('--val_loss', type=str, default='raw_l1')
        parser.add_argument('--unfreeze_stft_from', type=int, default=-1)  # -1 means never.

        return Selective_Audio_Manipulation.add_model_specific_args(parser)

    def __init__(self, n_fft, hop_length, num_frame,
                 spec_type, spec_est_mode,
                 spec2spec,
                 optimizer, lr,
                 train_loss, val_loss
                 ):
        super(Spectrogram_based, self).__init__(n_fft, hop_length, num_frame,
                                                optimizer, lr)

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frame = num_frame

        assert spec_type in ['magnitude', 'complex']
        assert spec_est_mode in ['masking', 'mapping']
        self.magnitude_based = spec_type == 'magnitude'
        self.masking_based = spec_est_mode == 'masking'
        self.stft = fourier.multi_channeled_STFT(n_fft=n_fft, hop_length=hop_length)
        self.stft.freeze()

        self.spec2spec = spec2spec
        self.valid_estimation_dict = {}
        self.val_loss = val_loss
        self.train_loss = train_loss

        self.init_weights()

    def init_weights(self):
        init_weights_functional(self.spec2spec,
                                self.spec2spec.activation)

    def training_step(self, batch, batch_idx):
        before, after, desc, length = batch
        loss = self.train_loss(self, before, after, desc, length)
        self.log('train_loss', loss, prog_bar=False, logger=True, on_step=False, on_epoch=True,
                 reduce_fx=torch.mean)

        return loss

    # Validation Process
    def on_validation_epoch_start(self):
        for amss in self.val_dataloader().dataset.SAEs:
            self.valid_estimation_dict[amss.gen_desc_default()] = {mixture_idx: {}
                                                                   for mixture_idx
                                                                   in range(14)}

    def validation_step(self, batch, batch_idx):

        before, after, tokens, token_lengths, track_ids, window_offsets, descs = batch

        loss = self.val_loss(self, before, after, tokens, token_lengths)
        self.log('raw_val_loss', loss, prog_bar=False, logger=False, reduce_fx=torch.mean)

        # Result Cache
        if 0 in track_ids:
            estimated_targets = self.manipulate(before, tokens, token_lengths)[:, self.trim_length:-self.trim_length]

            for track_idx, window_offset, desc, estimated_target \
                    in zip(track_ids, window_offsets, descs, estimated_targets):

                if track_idx == 0:
                    self.valid_estimation_dict[desc][track_idx][window_offset] = estimated_target.detach().cpu().numpy()

        return loss

    def validation_epoch_end(self, outputs: List[Any]) -> None:

        for idx in [0]:
            estimation = {}
            for amss in self.valid_estimation_dict.keys():
                estimation[amss] = get_estimation(idx, amss, self.valid_estimation_dict)
                if estimation[amss] is None:
                    continue
                if estimation[amss] is not None:
                    estimation[amss] = estimation[amss].astype(np.float32)

                    if self.current_epoch > 1 and isinstance(self.logger, WandbLogger):
                        track = estimation[amss]
                        if track.shape[0] > 40 * 44100:
                            track = track[44100 * 20:44100 * 40]

                        self.logger.experiment.log({'result_sample_{}_{}'.format(self.current_epoch, amss): [
                            wandb.Audio(track, caption='{}_{}'.format(idx, amss), sample_rate=44100)]})

        reduced_loss = torch.stack(outputs).mean()
        self.log('val_loss', reduced_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        print(reduced_loss)

    def test_step(self, batch, batch_idx):

        sam_desc_encoded, sam_desc_texts, before, after, track_ids, window_offsets, tge_ids = batch

        estimated_targets = self.manipulate(before, sam_desc_encoded, )[:, self.trim_length:-self.trim_length]
        for mixture, mixture_idx, window_offset, sam_desc_text, estimated_target \
                in zip(before, track_ids, window_offsets, sam_desc_texts, estimated_targets):
            self.test_estimation_dict[sam_desc_text][mixture_idx.item()][
                window_offset.item()] = estimated_target.detach().cpu().numpy()

        return torch.zeros(0)

    def test_epoch_end(self, outputs: List[Any]) -> None:

        import museval
        results = museval.EvalStore(frames_agg='median', tracks_agg='median')
        idx_list = range(self.musdb_test.unmixed_dataset.num_tracks)

        for idx in idx_list:
            estimation = {}
            for sae in self.test_dataloader().dataset.SAEs:
                sae = sae.gen_text_default()
                estimation[sae] = get_estimation(idx, sae, self.test_estimation_dict)

                if estimation[sae] is not None:
                    estimation[sae] = estimation[sae].astype(np.float32)

            # Real SDR
            if len(estimation) == len(self.test_dataloader().dataset.SAEs):
                track_length = self.musdb_test.unmixed_dataset.lengths[idx]
                estimated_targets = [estimation[sae.gen_text_default()][:track_length] for sae in
                                     self.test_dataloader().dataset.SAEs]

                if track_length > estimated_targets[0].shape[0]:
                    raise NotImplementedError
                else:
                    estimated_targets_dict = {sae.gen_text_default(): estimation[sae.gen_text_default()][:track_length]
                                              for sae in
                                              self.test_dataloader().dataset.SAEs}

                    new_dict = {'vocals': estimated_targets_dict['extract vocals'],
                                'drums': estimated_targets_dict['extract drums'],
                                'bass': estimated_targets_dict['extract bass'],
                                'other': estimated_targets_dict['extract other']}

                    track_score = museval.eval_mus_track(
                        self.musdb_test.unmixed_dataset.musdb_test[idx],
                        new_dict
                    )

                    score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                        ['target', 'metric'])['score'] \
                        .median().to_dict()

                    if isinstance(self.logger, WandbLogger):
                        self.logger.experiment.log(
                            {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})

                    else:
                        print(track_score)

                    results.add_track(track_score)

            if idx == 1 and isinstance(self.logger, WandbLogger):
                for sae in self.musdb_test.SAEs:
                    self.logger.experiment.log(
                        {'result_sample_{}_{}'.format(self.current_epoch, sae.gen_text_default()): [
                            wandb.Audio(estimation[sae.gen_text_default()],
                                        caption='{}_{}'.format(idx, sae.gen_text_default()), sample_rate=44100)]})

        if isinstance(self.logger, WandbLogger):

            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            self.logger.experiment.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )
        else:
            print(results)

    def estimate_spec(self, before, token_lengths, tokens):
        phase = None
        if self.magnitude_based:
            mag, phase = self.stft.to_mag_phase(before)
            input_spec = mag.transpose(-1, -3)

        else:
            spec_complex = self.stft.to_spec_complex(before)  # *, N, T, 2, ch
            spec_complex = torch.flatten(spec_complex, start_dim=-2)  # *, N, T, 2ch
            input_spec = spec_complex.transpose(-1, -3)  # *, 2ch, T, N
        output_spec = self.spec2spec(input_spec, tokens, token_lengths)
        if self.masking_based:
            output_spec = input_spec * output_spec
        else:
            pass  # Use the original output_spec

        return output_spec, phase

    @abstractmethod
    def to_spec(self, input_signal) -> torch.Tensor:
        pass

    @abstractmethod
    def manipulate(self, before, tokens, token_lengths) -> torch.Tensor:
        pass

