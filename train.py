from argparse import ArgumentParser
from datetime import datetime

from pytorch_lightning import Trainer

from src.amss.model_definition import get_class_by_name
from src.amss.script import trainer
from src.data.musdb_amss_dataset.data_provider import DataProvider
from src.utils.functions import mkdir_if_not_exists


def main(args):
    pass


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--model', type=str)
    temp_args, _ = parser.parse_known_args()

    # Model
    model = get_class_by_name(temp_args.model)
    parser = model.add_model_specific_args(parser)

    # Dataset
    parser = DataProvider.add_data_provider_args(parser)

    # Environment Setup
    mkdir_if_not_exists('etc')
    mkdir_if_not_exists('etc/checkpoints')

    parser.add_argument('--ckpt_root_path', type=str, default='etc/checkpoints')
    parser.add_argument('--log', type=str, default=True)
    parser.add_argument('--run_id', type=str, default=str(datetime.today().strftime("%Y%m%d_%H%M")))
    parser.add_argument('--save_weights_only', type=bool, default=False)

    # Env parameters
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=False)

    parser.add_argument('--save_top_k', type=int, default=5)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--seed', type=int, default='2020')
    parser.add_argument('--gpu_index', type=str, default=None)

    # Use Pretrained Word Embedding
    parser.add_argument('--pre_trained_word_embedding', type=str, default='glove.6B.100d')

    parser = Trainer.add_argparse_args(parser)
    trainer.train(parser.parse_args())