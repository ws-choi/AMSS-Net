import os
from argparse import ArgumentParser
from concurrent import futures
from pathlib import Path

import musdb
import museval
import soundfile
import wandb
from tqdm import tqdm

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


def musdb_evaluation(_model, _test_set, _logger, _wandb_logger, _cached, _wav_cache, _cache_dir):
    # eval
    pendings = []
    _cache_dir = Path(_cache_dir)

    with futures.ProcessPoolExecutor(4) as pool:
        for i in tqdm(range(len(_test_set))):
            # https://github.com/facebookresearch/demucs/blob/fa7480d5822945e17bf8a16e4baad9f2b631dffc/demucs/test.py#L53
            track = _test_set.tracks[i]

            if _cached:
                estimated = {source: soundfile.read(_cache_dir.joinpath('{}_{}.wav'.format(track.name, source)))[0]
                             for source
                             in ['vocals', 'drums', 'bass', 'other']}
            else:
                estimated = _model.separate(track.audio)

            if _wav_cache and not _cached:
                for source in estimated.keys():
                    soundfile.write(
                        _cache_dir.joinpath('{}_{}.wav'.format(track.name, source)),
                        estimated[source],
                        44100
                    )

            # https://github.com/ws-choi/Conditioned-Source-Separation-LaSAFT/blob/a3e60bfdc1d5b4d20f5d5df852241a0c8d80420a/lasaft/source_separation/conditioned/separation_framework.py#L232
            pendings.append((i, track.name, pool.submit(museval.eval_mus_track, track, estimated)))
            del track, estimated

        results = museval.EvalStore(frames_agg='median', tracks_agg='median')

        # Eval each track
        for i, track_name, track_score in tqdm(pendings):
            track_score = track_score.result()

            score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                ['target', 'metric'])['score'] \
                .median().to_dict()

            if _logger == 'wandb':
                _wandb_logger.log(
                    {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})
            else:
                print(track_score)

            results.add_track(track_score)

            # if i == 1 and _logger == 'wandb':
            #     for target_name in ['vocals', 'drums', 'bass', 'other']:
            #         _wandb_logger.log({'result_sample_{}_{}'.format(i, target_name): [
            #             wandb.Audio(estimated[target_name], caption='{}_{}'.format(i, target_name),
            #                         sample_rate=44100)]})

        # Eval average
        if _logger == 'wandb':
            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            _wandb_logger.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )

            _wandb_logger.finish()
        else:
            print(results)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--model_param', type=str)
    parser.add_argument('--musdb_root', type=str)
    parser.add_argument('--log', type=str)
    parser.add_argument('--wav_cache', type=bool, default=False)

    args = parser.parse_args().__dict__
    model_param = args['model_param']

    # wav_cache
    wav_cache = args['wav_cache']

    # check cached.
    cache_dir = 'etc/result_cached/{}_{}'.format(args['model'], model_param)

    if os.path.exists(cache_dir):
        cached = True
        if cached and wav_cache:
            print('please remove the existing directory for cache.')
            raise RuntimeError

        print('result cached')

    else:
        cached = False
        if wav_cache:
            if not os.path.exists('etc'):
                os.mkdir('etc')
            if not os.path.exists('etc/result_cached'):
                os.mkdir('etc/result_cached')
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)


    # model
    model = None if cached else get_model(args)

    if model_param is None:
        model_param = "None"

    # dataset
    musdb_path = args['musdb_root']
    test_set = musdb.DB(musdb_path, is_wav=True, subsets=["test"])
    assert len(test_set) == 50

    # logger
    logger = args['log']
    assert logger in ['wandb', None]



    if logger == 'wandb':
        wandb_logger = wandb.init(job_type='eval', config=args, project='musdb18', tags=[args['model']],
                                  name='{}_{}'.format(args['model'], args['model_param']))
    else:
        wandb_logger = None

    # try:
    musdb_evaluation(model, test_set, logger, wandb_logger, cached, wav_cache, cache_dir)

    # except Exception as ex:
    #     print(ex)
    #     if wav_cache:
    #         os.rmdir(cache_dir)
