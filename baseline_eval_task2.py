import os
from argparse import ArgumentParser
from concurrent import futures
from pathlib import Path

import musdb
import museval
import numpy as np
import soundfile
import wandb
from tqdm import tqdm

from src.data.musdb_amss_dataset.amss_task2_datasets import task2_config
from src.data.musdb_wrapper import MusdbUnmixedTestSet
from task2_eval import multi_channel_dist


def get_model(argument):
    model_name = argument['model']
    if model_name == 'demucs':
        from separators.demucs_wrapper import Demucs_separator
        model_param = argument['model_param']
        if model_param is None:
            model_param = 'demucs'
        if isinstance(model_param, str) and model_param.lower() == "none":
            model_param = 'demucs'

        return Demucs_separator(model_param)

    elif model_name == 'spleeter':
        from separators.spleeter_wrapper import Spleeter_separator
        return Spleeter_separator(json_path='4stems-16kHz.json')

    elif model_name == 'x_umx':
        from separators.x_umx_wrapper import X_umx_wrapper
        import nnabla as nn
        from nnabla.ext_utils import get_extension_context
        ctx = get_extension_context('cudnn')
        nn.set_default_context(ctx)
        nn.set_auto_forward(True)

        return X_umx_wrapper()

    elif model_name == 'lasaftnet':
        model_param = argument['model_param']
        from separators.lasaftnet_wrapper import LaSAFT_separator
        if model_param is None:
            model_param = 'lasaft_large_2020'
        return LaSAFT_separator(model_param)

    else:
        raise ModuleNotFoundError


def eval_amss_track(separated, unmixed_track, amss):
    result_dict = {}

    _before, _after, _tar_before, _tar_after, _acc = amss.edit_for_test(separated)
    result_dict['amss_hat'] = _after

    before, after, tar_before, tar_after, acc = amss.edit_for_test(unmixed_track)
    result_dict['amss'] = after

    return result_dict


def task2_evaluation(_model, _test_set, _unmixed_test_set, _logger, _wandb_logger, _cached=False, _wav_cache=False, _cache_dir=None):
    # eval
    lr_mode = False
    # _cache_dir = Path(_cache_dir)

    # separated_results = []
    # for i in tqdm(range(len(_test_set))):
    #     # https://github.com/facebookresearch/demucs/blob/fa7480d5822945e17bf8a16e4baad9f2b631dffc/demucs/test.py#L53
    #     track = _test_set.tracks[i]
    #     separated = separate_all(_cache_dir, _cached, _model, _wav_cache, track)
    #     separated = [separated[s] for s in ['vocals', 'drums', 'bass', 'other']]
    #     separated = numpy.stack(separated)
    #     separated_results.append(separated)

    for amss in task2_config.evaluation_amss_set:
        desc = amss.gen_desc_default()
        skip_keyword = ['separate', 'mute', 'reverb']
        skip = False
        for keyword in skip_keyword:
            if keyword in desc:
                skip = True
                break
        if skip:
            continue

        print(amss)

        a_prime_results = []
        tar_results = []
        acc_results = []


        for i in tqdm(range(len(_test_set))):
            # https://github.com/facebookresearch/demucs/blob/fa7480d5822945e17bf8a16e4baad9f2b631dffc/demucs/test.py#L53

            unmixed_track = _unmixed_test_set[i]

            before, after, tar_before, tar_after, acc = amss.edit_for_test(unmixed_track)
            separated = separate_all(_cache_dir, _cached, _model, _wav_cache, before)
            separated = [separated[s] for s in ['vocals', 'drums', 'bass', 'other']]
            separated = np.stack(separated)

            _before, _after, _tar_before, _tar_after, _acc = amss.edit_for_test(separated)

            result_dict = {'amss_hat':_after, 'amss':after}
            a_prime_result = multi_channel_dist(result_dict['amss'], result_dict['amss_hat'])

            if logger == 'wandb':
                for key in a_prime_result.keys():
                    if ('left' in key or 'right' in key) and not lr_mode:
                        continue

                    wandb_logger.log({'a_prime/{}_{}'.format(desc, key): a_prime_result[key]})

            a_prime_results.append(
                np.array([a_prime_result['mae'],
                          a_prime_result['mae_left'],
                          a_prime_result['mae_right'],
                          a_prime_result['mfcc_rmse'],
                          a_prime_result['mfcc_rmse_left'],
                          a_prime_result['mfcc_rmse_right']])
            )

        if logger == 'wandb':
            scores = np.mean(np.stack(a_prime_results), axis=0)
            wandb_logger.log({'agg_mid/a_prime_mae_{}'.format(desc): scores[0]})
            wandb_logger.log({'agg_left/a_prime_mae_{}'.format(desc): scores[1]})
            wandb_logger.log({'agg_right/a_prime_mae_{}'.format(desc): scores[2]})
            wandb_logger.log({'agg_mid/a_prime_mfccrmse_{}'.format(desc): scores[3]})
            wandb_logger.log({'agg_left/a_prime_mfccrmse_{}'.format(desc): scores[4]})
            wandb_logger.log({'agg_right/a_prime_mfccrmse_{}'.format(desc): scores[5]})

        else:
            scores = np.mean(np.stack(a_prime_results), axis=0)
            print({'agg_mid/a_prime_mae_{}'.format(desc): scores[0]})
            print({'agg_left/a_prime_mae_{}'.format(desc): scores[1]})
            print({'agg_right/a_prime_mae_{}'.format(desc): scores[2]})
            print({'agg_mid/a_prime_mfccrmse_{}'.format(desc): scores[3]})
            print({'agg_left/a_prime_mfccrmse_{}'.format(desc): scores[4]})
            print({'agg_right/a_prime_mfccrmse_{}'.format(desc): scores[5]})


def separate_all(_cache_dir, _cached, _model, _wav_cache, track):
    if _cached:
        estimated = {source: soundfile.read(_cache_dir.joinpath('{}_{}.wav'.format(track.name, source)))[0]
                     for source
                     in ['vocals', 'drums', 'bass', 'other']}
    else:
        estimated = _model.separate(track)
    if _wav_cache and not _cached:
        for source in estimated.keys():
            soundfile.write(
                _cache_dir.joinpath('{}_{}.wav'.format(track.name, source)),
                estimated[source],
                44100
            )
    return estimated


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_param', type=str)
    parser.add_argument('--musdb_root', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--wav_cache', type=bool, default=False)

    args = parser.parse_args().__dict__
    model_param = args['model_param']

    # # wav_cache
    # wav_cache = args['wav_cache']
    #
    # # check cached.
    # cache_dir = 'etc/result_cached/{}_{}'.format(args['model'], model_param)
    #
    # if os.path.exists(cache_dir):
    #     cached = True
    #     if cached and wav_cache:
    #         print('please remove the existing directory for cache.')
    #         raise RuntimeError
    #
    #     print('result cached')

    # else:
    #     cached = False
    #     if wav_cache:
    #         if not os.path.exists('etc'):
    #             os.mkdir('etc')
    #         if not os.path.exists('etc/result_cached'):
    #             os.mkdir('etc/result_cached')
    #         if not os.path.exists(cache_dir):
    #             os.mkdir(cache_dir)

    # model
    # model = None if cached else get_model(args)
    model = get_model(args)

    if model_param is None:
        model_param = "None"

    # dataset
    musdb_path = args['musdb_root']
    unmixed_test_set = MusdbUnmixedTestSet(musdb_path)
    test_set = musdb.DB(musdb_path, is_wav=True, subsets=["test"])
    assert len(unmixed_test_set) == 50
    assert len(test_set) == 50

    # logger
    logger = args['log']
    assert logger in ['wandb', None]

    if logger == 'wandb':
        wandb_logger = wandb.init(job_type='eval', config=args, project='task2_eval_mm', tags=[args['model']],
                                  name='{}_{}'.format(args['model'], args['model_param']))
    else:
        wandb_logger = None

    # try:
    task2_evaluation(model, test_set, unmixed_test_set, logger, wandb_logger)

    # except Exception as ex:
    #     print(ex)
    #     if wav_cache:
    #         os.rmdir(cache_dir)
