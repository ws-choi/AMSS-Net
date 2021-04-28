import inspect
from warnings import warn

import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.amss.model_definition import get_class_by_name
from src.data.musdb_amss_dataset.data_provider import DataProvider

from src.utils.functions import mkdir_if_not_exists
from pathlib import Path
from pytorch_lightning import Trainer, seed_everything


def train(param):
    if not isinstance(param, dict):
        args = vars(param)
    else:
        args = param

    # GPUs
    if args['gpu_index'] is not None:
        args['gpus'] = str(args['gpu_index'])

    # DATASET
    ##########################################################
    dp_valid_kwargs = inspect.signature(DataProvider.__init__).parameters
    dp_kwargs = dict((name, args[name]) for name in dp_valid_kwargs if name in args)

    data_provider = DataProvider(**dp_kwargs)
    training_dataset, training_dataloader = data_provider.get_training_dataset_and_loader()
    validation_dataset, validation_dataloader = data_provider.get_validation_dataset_and_loader()
    ##########################################################

    # Set Seed
    if args['resume_from_checkpoint'] is None:
        if args['seed'] is not None:
            seed_everything(args['seed'])

    # MODEL
    ##########################################################

    # Check using pretraining
    pre_trained_word_embedding = args['pre_trained_word_embedding']

    if pre_trained_word_embedding is None:
        pass
    elif pre_trained_word_embedding == 'glove.6B.100d':
        assert args['embedding_dim'] == 100
    else:
        raise ModuleNotFoundError

    # # # get framework
    framework = get_class_by_name(args['model'])
    if args['spec_type'] != 'magnitude':
        args['input_channels'] = 4

    # Model instantiation
    args['vocab_size'] = len(training_dataset.vocab)
    model = framework(**args)

    if pre_trained_word_embedding is None:
        pass
    elif pre_trained_word_embedding == 'glove.6B.100d':
        with torch.no_grad():
            from torchtext.vocab import GloVe
            vocab = training_dataset.vocab
            glove = GloVe(name='6B', dim=100)
            for token in vocab:
                if token in glove.stoi.keys():
                    glove_i = glove.stoi[token]
                    embedding_i = training_dataset.word_to_idx[token]
                    model.spec2spec.embedding.weight[embedding_i] = glove.vectors[glove_i]
                    pass

    else:
        raise ModuleNotFoundError


    if args['last_activation'] != 'identity' and args['spec_est_mode'] != 'masking':
        warn('Please check if you really want to use a mapping-based spectrogram estimation method '
             'with a final activation function. ')
    ##########################################################

    # -- checkpoint
    ckpt_path = Path(args['ckpt_root_path'])
    mkdir_if_not_exists(ckpt_path)
    ckpt_path = ckpt_path.joinpath(args['model'])
    mkdir_if_not_exists(ckpt_path)
    run_id = args['run_id']
    ckpt_path = ckpt_path.joinpath(run_id)
    mkdir_if_not_exists(ckpt_path)
    save_top_k = args['save_top_k']

    checkpoint_callback = ModelCheckpoint(
        filepath=ckpt_path,
        save_top_k=save_top_k,
        verbose=False,
        monitor='val_loss',
        save_last=False,
        save_weights_only=args['save_weights_only']
    )
    args['checkpoint_callback'] = checkpoint_callback

    # -- early stop
    patience = args['patience']
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=patience,
        verbose=False
    )
    args['early_stop_callback'] = early_stop_callback

    # -- logger setting
    log = args['log']
    if log == 'False':
        args['logger'] = False
    elif log == 'wandb':
        args['logger'] = WandbLogger(project=args['task'], tags=args['model'], offline=False, id=run_id)
        args['logger'].log_hyperparams(model.hparams)
        args['logger'].watch(model, log='all')
    elif log == 'tensorboard':
        raise NotImplementedError
    else:
        args['logger'] = True  # default
        default_save_path = 'etc/lightning_logs'
        mkdir_if_not_exists(default_save_path)

    valid_kwargs = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs = dict((name, args[name]) for name in valid_kwargs if name in args)

    # Trainer Definition

    # Trainer
    trainer = Trainer(**trainer_kwargs)

    for key in args.keys():
        print('{}:{}'.format(key, args[key]))

    if args['auto_lr_find']:
        lr_find = trainer.tuner.lr_find(model,
                                        training_dataloader,
                                        validation_dataloader,
                                        early_stop_threshold=None,
                                        min_lr=1e-5)

        print(f"Found lr: {lr_find.suggestion()}")
        return 0

    if args['resume_from_checkpoint'] is not None:
        'resume'

    trainer.fit(model, training_dataloader, validation_dataloader,)

    return None
