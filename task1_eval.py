from argparse import ArgumentParser
from concurrent import futures
from pathlib import Path

from src.data.musdb_amss_dataset.amss_task1_datasets import task1_config
from src.data.musdb_amss_dataset.amss_task2_datasets import task2_config
from src.utils.functions import load_hparams_from_yaml
from src.amss import model_definition
from src.data.musdb_amss_dataset.musdb_amss_definitions import musdb_amss_config

from src.data.musdb_wrapper import MusdbWrapperDataset, SingleTrackSet_for_Task1
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import museval
import wandb


def get_musdb_testset(args, musdb_root=None):
    new_args = {key: args[key] for key in args.keys() if key in
                ['musdb_root', 'n_fft', 'hop_length', 'num_frame']}

    if musdb_root is not None:
        new_args['musdb_root'] = musdb_root

    new_args['subset'] = 'test'
    new_args['target_names'] = None

    return MusdbWrapperDataset(**new_args)


def get_task_config(task):
    if task == 'task1':
        return task1_config
    elif task == 'task2':
        return task2_config
    else:
        raise ModuleNotFoundError


def get_sdr_items_task1(idx, mani_dict, musdb_test):
    new_dict = {}
    new_dict['vocals'] = mani_dict[idx]['separate vocals']
    new_dict['drums'] = mani_dict[idx]['separate drums']
    new_dict['bass'] = mani_dict[idx]['separate bass']
    new_dict['other'] = mani_dict[idx]['mute vocals, drums, bass']

    return musdb_test[idx], new_dict


def eval(ckpt_root, run_id, config_path, ckpt_path, musdb_root=None, batch_size=4, cuda=True, logger='wandb'):
    ckpt_root = Path(ckpt_root)
    run_id = ckpt_root.joinpath(run_id)
    config_path = run_id.joinpath(config_path)
    ckpt_path = run_id.joinpath(ckpt_path)
    assert logger in ['wandb', None]

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

    # Define task
    task_config = get_task_config('task1')
    test_wrapper = get_musdb_testset(args, musdb_root)

    model = model.eval()
    model = model.cuda() if cuda else model

    manipulated_dict = {}

    for i in tqdm(range(test_wrapper.num_tracks)):
        track = test_wrapper.get_audio(i, 'mixture')
        track_length = track.shape[0]
        manipulated_dict[i] = {}

        for amss in task_config.evaluation_amss_set:
            amss_desc_text = amss.gen_desc_default()

            dataset = SingleTrackSet_for_Task1(track, hop_length, num_frame, amss, word_to_idx)
            dataloader = DataLoader(dataset, batch_size, shuffle=False)
            trim_length = dataset.trim_length

            manipulated = []

            with torch.no_grad():
                for j, (before, desc) in enumerate(dataloader):
                    if cuda:
                        before, desc = before.cuda(), desc.cuda()
                    afters = model.manipulate(before, desc, token_lengths=[len(d) for d in desc])
                    if cuda:
                        afters = afters.cpu()
                    manipulated.append(afters.detach().numpy())

            manipulated_trim = np.vstack(manipulated)[:, trim_length:-trim_length]
            manipulated_trim = manipulated_trim.reshape(-1, 2)[:track_length]
            manipulated_dict[i][amss_desc_text] = manipulated_trim

    if logger == 'wandb':
        wandb_logger = wandb.init(job_type='eval', config=args, project='task1_eval', tags=args['model'],
                                  name='{}_{}'.format(args['model'], ckpt_path))
    else:
        wandb_logger = None

    musdb_test = test_wrapper.musdb_reference

    pendings = []
    with futures.ProcessPoolExecutor(4) as pool:
        for i in tqdm(range(test_wrapper.num_tracks)):
            musdb_test_i, mani_dict_i = get_sdr_items_task1(i, manipulated_dict, musdb_test)
            pendings.append((i, pool.submit(museval.eval_mus_track, musdb_test_i, mani_dict_i)))
            del musdb_test_i, mani_dict_i

        results = museval.EvalStore(frames_agg='median', tracks_agg='median')

        # Eval each track
        for i, track_score in tqdm(pendings):
            track_score = track_score.result()

            score_dict = track_score.df.loc[:, ['target', 'metric', 'score']].groupby(
                ['target', 'metric'])['score'] \
                .median().to_dict()

            if logger == 'wandb':
                wandb_logger.log(
                    {'test_result/{}_{}'.format(k1, k2): score_dict[(k1, k2)] for k1, k2 in score_dict.keys()})
            else:
                print(track_score)

            results.add_track(track_score)

        # Eval average
        if logger == 'wandb':
            result_dict = results.df.groupby(
                ['track', 'target', 'metric']
            )['score'].median().reset_index().groupby(
                ['target', 'metric']
            )['score'].median().to_dict()

            wandb_logger.log(
                {'test_result/agg/{}_{}'.format(k1, k2): result_dict[(k1, k2)] for k1, k2 in result_dict.keys()}
            )

            wandb_logger.finish()
        else:
            print(results)


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
