from argparse import ArgumentParser
from pathlib import Path

import librosa
import museval
import numpy as np
import torch
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import paired_distances
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.amss import model_definition
from src.data.musdb_amss_dataset.amss_task2_datasets import task2_config
from src.data.musdb_amss_dataset.musdb_amss_definitions import musdb_amss_config
from src.data.musdb_wrapper import MusdbUnmixedTestSet, SingleTrackSet_for_Task2
from src.utils.eval_metric_marco import getMSE_MFCC_mc
from src.utils.functions import load_hparams_from_yaml


def get_unmixed_testset(args, musdb_root=None):
    new_args = {key: args[key] for key in args.keys() if key in
                ['musdb_root', 'n_fft', 'hop_length', 'num_frame']}

    if musdb_root is not None:
        new_args['musdb_root'] = musdb_root

    return MusdbUnmixedTestSet(**new_args)


def getMFCC(x, sr=44100, mels=128, mfcc=13, mean_norm=False):
    # melspec = librosa.feature.melspectrogram(y=x, sr=sr, S=None,
    #                                          n_fft=4096, hop_length=2048,
    #                                          n_mels=mels, power=2.0)
    # melspec_dB = librosa.power_to_db(melspec, ref=np.max)
    # mfcc = librosa.feature.mfcc(S=melspec_dB, sr=sr, n_mfcc=mfcc)
    # if mean_norm:
    #     mfcc -= (np.mean(mfcc, axis=0))
    return librosa.feature.mfcc(x,sr=sr, n_mfcc=mfcc, n_mels=mels, n_fft=4096, hop_length=2048)


def eval_amss_track(model, track, amss, batch_size, cuda, hop_length, num_frame, word_to_idx):
    result_dict = {}
    track_length = track.shape[1]

    # 1: Evaluate amss for the i-th track
    _, before, after = amss.edit(track)
    result_dict['amss'] = after

    # 2: Estimate Manipulated Track
    with torch.no_grad():
        dataset_before = SingleTrackSet_for_Task2(before, hop_length, num_frame, amss, word_to_idx)
        dataloader = DataLoader(dataset_before, batch_size, shuffle=False)
        trim_length = dataset_before.trim_length

        manipulated_hat = []

        for before, desc_amss, _, _ in dataloader:
            if cuda:
                before, desc_amss = before.cuda(), desc_amss.cuda()

            afters = model.manipulate(before, desc_amss, token_lengths=[len(d) for d in desc_amss])

            if cuda:
                afters = afters.cpu()
            manipulated_hat.append(afters.detach().numpy())

    manipulated_trim = np.vstack(manipulated_hat)[:, trim_length:-trim_length]
    manipulated_trim = manipulated_trim.reshape(-1, 2)[:track_length]
    result_dict['amss_hat'] = manipulated_trim

    # Sep for target Mute for ACC
    # with torch.no_grad():
    #     dataset_after = SingleTrackSet_for_Task2(manipulated_trim, hop_length, num_frame, amss, word_to_idx)
    #     dataloader = DataLoader(dataset_after, batch_size, shuffle=False)
    #
    #     target_hats = []
    #     acc_hats = []
    #
    #     for after_hat, _, desc_sep, desc_mute in dataloader:
    #         if cuda:
    #             after_hat, desc_sep, desc_mute = after_hat.cuda(), desc_sep.cuda(), desc_mute.cuda()
    #
    #         target_hat = model.manipulate(after_hat, desc_sep, token_lengths=[len(d) for d in desc_sep])
    #         acc_hat = model.manipulate(after_hat, desc_mute, token_lengths=[len(d) for d in desc_sep])
    #
    #         if cuda:
    #             target_hat = target_hat.cpu()
    #             acc_hat = acc_hat.cpu()
    #
    #         target_hats.append(target_hat.detach().numpy())
    #         acc_hats.append(acc_hat.detach().numpy())
    #
    # target_trim = np.vstack(target_hats)[:, trim_length:-trim_length]
    # target_trim = target_trim.reshape(-1, 2)[:track_length]
    #
    # acc_trim = np.vstack(acc_hats)[:, trim_length:-trim_length]
    # acc_trim = acc_trim.reshape(-1, 2)[:track_length]
    #
    # result_dict['tar_hat'] = target_trim
    # result_dict['acc_hat'] = acc_trim

    return result_dict


def eval(ckpt_root, run_id, config_path, ckpt_path, musdb_root=None, batch_size=4, cuda=True, logger='wandb'):

    lr_mode = False
    ckpt_root = Path(ckpt_root)
    run_id = ckpt_root.joinpath(run_id)
    config_path = run_id.joinpath(config_path)
    ckpt_path = run_id.joinpath(ckpt_path)

    # Define Model
    config = load_hparams_from_yaml(config_path)
    args = {key: config[key]['value'] for key in config.keys() if isinstance(config[key], dict)}

    model = model_definition.get_class_by_name(args['model'])
    model = model(**args)
    model = model.load_from_checkpoint(ckpt_path)

    # load word dictionary
    word_to_idx = musdb_amss_config.word_to_idx

    # Load related stft config
    hop_length = args['hop_length']
    num_frame = args['num_frame']

    test_unmixed = get_unmixed_testset(args, musdb_root)

    model = model.eval()
    model = model.cuda() if cuda else model

    if logger == 'wandb':
        project = 'task2_eval_dev' if 'dev' in musdb_root else 'task2_eval'
        wandb_logger = wandb.init(job_type='eval', config=args, project=project, tags=[args['model']],
                                  name='{}_{}'.format(args['model'], ckpt_path))
    else:
        wandb_logger = None

    for amss in task2_config.evaluation_amss_set:

        desc = amss.gen_desc_default()
        skip_keyword = ['separate', 'mute']
        skip = False
        for keyword in skip_keyword:
            if keyword in desc:
                skip = True
                break

        if skip:
            continue

        # clear_output()
        print(amss)

        a_prime_results = []
        tar_results = []
        acc_results = []

        # For the i-th track!
        for track_idx in tqdm(range(test_unmixed.num_tracks)):
            track = test_unmixed[track_idx]
            result_dict = eval_amss_track(model, track, amss, batch_size, cuda, hop_length, num_frame, word_to_idx)

            a_prime_result = multi_channel_dist(result_dict['amss'], result_dict['amss_hat'])
            # tar_result = multi_channel_dist(result_dict['tar'], result_dict['tar_hat'])
            # acc_result = multi_channel_dist(result_dict['acc'], result_dict['acc_hat'])

            if logger == 'wandb':

                for key in a_prime_result.keys():
                    if ('left' in key or 'right' in key) and not lr_mode:
                        continue

                    wandb_logger.log({'a_prime/{}_{}'.format(desc, key): a_prime_result[key]})

                # for key in tar_result.keys():
                #     if ('left' in key or 'right' in key) and not lr_mode:
                #         continue
                #     wandb_logger.log({'target/{}_{}'.format(desc, key): tar_result[key]})
                #
                # for key in acc_result.keys():
                #     if ('left' in key or 'right' in key) and not lr_mode:
                #         continue
                #     wandb_logger.log({'acc/{}_{}'.format(desc, key): acc_result[key]})

                start = result_dict['amss_hat'].shape[0] // 2

                if 'dev' in project:
                    wandb_logger.log({'result_sample_{}_{}'.format(track_idx, amss): [
                        wandb.Audio(result_dict['amss_hat'][start:start + 44100 * 2],
                                    caption='{}_{}'.format(track_idx, amss), sample_rate=44100)]})

            else:
                for result in [a_prime_result]: #, tar_result, acc_result]:
                    print(result)

            a_prime_results.append(
                np.array([a_prime_result['mae'],
                          a_prime_result['mae_left'],
                          a_prime_result['mae_right'],
                          a_prime_result['mfcc_rmse'],
                          a_prime_result['mfcc_rmse_left'],
                          a_prime_result['mfcc_rmse_right']])
            )

            # tar_results.append(
            #     np.array([tar_result['mae'],
            #               tar_result['mae_left'],
            #               tar_result['mae_right'],
            #               tar_result['mfcc_rmse'],
            #               tar_result['mfcc_rmse_left'],
            #               tar_result['mfcc_rmse_right']])
            # )
            #
            # acc_results.append(
            #     np.array([acc_result['mae'],
            #               acc_result['mae_left'],
            #               acc_result['mae_right'],
            #               acc_result['mfcc_rmse'],
            #               acc_result['mfcc_rmse_left'],
            #               acc_result['mfcc_rmse_right']])
            # )

        if logger == 'wandb':
            scores = np.mean(np.stack(a_prime_results), axis=0)
            wandb_logger.log({'agg_mid/a_prime_mae_{}'.format(desc): scores[0]})
            wandb_logger.log({'agg_left/a_prime_mae_{}'.format(desc): scores[1]})
            wandb_logger.log({'agg_right/a_prime_mae_{}'.format(desc): scores[2]})
            wandb_logger.log({'agg_mid/a_prime_mfccrmse_{}'.format(desc): scores[3]})
            wandb_logger.log({'agg_left/a_prime_mfccrmse_{}'.format(desc): scores[4]})
            wandb_logger.log({'agg_right/a_prime_mfccrmse_{}'.format(desc): scores[5]})

            # scores = np.mean(np.stack(tar_results), axis=0)
            # wandb_logger.log({'agg_mid/target_mae_{}'.format(desc): scores[0]})
            # wandb_logger.log({'agg_left/target_mae_{}'.format(desc): scores[1]})
            # wandb_logger.log({'agg_right/target_mae_{}'.format(desc): scores[2]})
            # wandb_logger.log({'agg_mid/target_mfccrmse_{}'.format(desc): scores[3]})
            # wandb_logger.log({'agg_left/target_mfccrmse_{}'.format(desc): scores[4]})
            # wandb_logger.log({'agg_right/target_mfccrmse_{}'.format(desc): scores[5]})
            #
            # scores = np.mean(np.stack(acc_results), axis=0)
            # wandb_logger.log({'agg_mid/acc_mae_{}'.format(desc): scores[0]})
            # wandb_logger.log({'agg_left/acc_mae_{}'.format(desc): scores[1]})
            # wandb_logger.log({'agg_right/acc_mae_{}'.format(desc): scores[2]})
            # wandb_logger.log({'agg_mid/acc_mfccrmse_{}'.format(desc): scores[3]})
            # wandb_logger.log({'agg_left/acc_mfccrmse_{}'.format(desc): scores[4]})
            # wandb_logger.log({'agg_right/acc_mfccrmse_{}'.format(desc): scores[5]})

        else:
            scores = np.mean(np.stack(a_prime_results), axis=0)
            print({'agg_mid/a_prime_mae_{}'.format(desc): scores[0]})
            print({'agg_left/a_prime_mae_{}'.format(desc): scores[1]})
            print({'agg_right/a_prime_mae_{}'.format(desc): scores[2]})
            print({'agg_mid/a_prime_mfccrmse_{}'.format(desc): scores[3]})
            print({'agg_left/a_prime_mfccrmse_{}'.format(desc): scores[4]})
            print({'agg_right/a_prime_mfccrmse_{}'.format(desc): scores[5]})
            #
            # scores = np.mean(np.stack(tar_results), axis=0)
            # print({'agg_mid/target_mae_{}'.format(desc): scores[0]})
            # print({'agg_left/target_mae_{}'.format(desc): scores[1]})
            # print({'agg_right/target_mae_{}'.format(desc): scores[2]})
            # print({'agg_mid/target_mfccrmse_{}'.format(desc): scores[3]})
            # print({'agg_left/target_mfccrmse_{}'.format(desc): scores[4]})
            # print({'agg_right/target_mfccrmse_{}'.format(desc): scores[5]})
            #
            # scores = np.mean(np.stack(acc_results), axis=0)
            # print({'agg_mid/acc_mae_{}'.format(desc): scores[0]})
            # print({'agg_left/acc_mae_{}'.format(desc): scores[1]})
            # print({'agg_right/acc_mae_{}'.format(desc): scores[2]})
            # print({'agg_mid/acc_mfccrmse_{}'.format(desc): scores[3]})
            # print({'agg_left/acc_mfccrmse_{}'.format(desc): scores[4]})
            # print({'agg_right/acc_mfccrmse_{}'.format(desc): scores[5]})

    if logger == 'wandb':
        wandb_logger.finish()


def multi_channel_dist(x, x_hat):
    # Normalize
    # ratio = np.mean(np.abs(x)) / np.mean(np.abs(x_hat))
    # x_hat = ratio * x_hat

    # Multi-channel
    left = x[:, 0]
    left_hat = x_hat[:, 0]
    right = x[:, 1]
    right_hat = x_hat[:, 1]

    metric_dict = {'mae': mean_absolute_error(x, x_hat),
                   'mae_left': mean_absolute_error(left, left_hat),
                   'mae_right': mean_absolute_error(right, right_hat)}

    left, left_hat, right, right_hat = [getMFCC(wave) for wave in [left, left_hat, right, right_hat]]
    metric_dict['mfcc_rmse'] = mean_squared_error(np.concatenate([left, right]), np.concatenate([left_hat, right_hat]), squared=False)
    metric_dict['mfcc_rmse_left'] = mean_squared_error(left, left_hat, squared=False)
    metric_dict['mfcc_rmse_right'] = mean_squared_error(right, right_hat, squared=False)

    return metric_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, default='etc/checkpoints/')
    parser.add_argument('--run_id', type=str)
    parser.add_argument('--config_path', type=str, default='config.yaml')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--musdb_root', type=str, default='../repos/musdb18_wav')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--logger', type=str, default=None)
    namespace = parser.parse_args()

    eval(**vars(namespace))
