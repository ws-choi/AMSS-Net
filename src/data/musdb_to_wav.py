from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm

import musdb
import soundfile
import numpy as np


def main(musdb_root, new_root=None):
    musdb_root_parent = Path(musdb_root).parent

    print('initiating...')

    if new_root is None:
        new_root = '{}_{}'.format(Path(musdb_root).name, 'wav')

    target = 'train'
    destination = musdb_root_parent.joinpath(new_root).joinpath(target)
    musdb_train = musdb.DB(root=musdb_root, subsets='train', split='train', is_wav=False)
    to_wav(musdb_train, destination)

    target = 'valid'
    destination = musdb_root_parent.joinpath(new_root).joinpath(target)
    musdb_valid = musdb.DB(root=musdb_root, subsets='train', split='valid', is_wav=False)
    to_wav(musdb_valid, destination)

    target = 'test'
    destination = musdb_root_parent.joinpath(new_root).joinpath(target)
    musdb_test = musdb.DB(root=musdb_root, subsets='test', is_wav=False)
    to_wav(musdb_test, destination)


def to_wav(musdb, destination):
    if not destination.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        print('do not terminate now. if you have to do, please remove the dir: {}'.format(destination))

    for i, track in enumerate(tqdm(musdb)):
        for target in ['linear_mixture', 'vocals', 'drums', 'bass', 'other']:
            soundfile.write(file='{}/{}_{}.wav'.format(destination, i, target),
                            data=track.targets[target].audio.astype(np.float32),
                            samplerate=track.rate
                            )


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--musdb_root', type=str)
    temp_args, _ = parser.parse_known_args()

    if temp_args.musdb_root is None:
        print()
        print('**** usage: fill this option: --musdb_root <dir>                     ****')
        print('*********** , where <dir> is the path that musdb18 dataset stored in ****')
        raise FileNotFoundError

    main(temp_args.musdb_root)
