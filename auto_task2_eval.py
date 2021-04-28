from argparse import ArgumentParser
from os import listdir
from pathlib import Path
from task2_eval import eval

def eval_call(ckpt_root, model, musdb_root, batch_size, cuda, logger):
    print(ckpt_root, model, musdb_root, batch_size, cuda, logger)
    ckpt_root = Path(ckpt_root).joinpath(model)

    for run_id in listdir(ckpt_root):
        print(run_id)
        config = 'config.yaml'
        print(config)
        ckpts = [file for file in listdir(ckpt_root.joinpath(run_id)) if 'ckpt' in file]
        for ckpt in ckpts:
            print('run_id: ', run_id)
            print('ckpt: ', ckpt)
            eval(ckpt_root, run_id, config, ckpt, musdb_root, batch_size, cuda, logger)

    print(ckpt_root)
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ckpt_root', type=str, default='etc/checkpoints/12')
    parser.add_argument('--model', type=str, default='isolasion_smpocm')
    parser.add_argument('--musdb_root', type=str, default='../repos/musdb18_wav')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--cuda', type=bool, default=False)
    parser.add_argument('--logger', type=str, default=None)
    namespace = parser.parse_args()

    eval_call(**vars(namespace))
