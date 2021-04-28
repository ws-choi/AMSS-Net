from src.data.musdb_amss import amss_mute_generator, amss_separate_generator
from src.data.musdb_amss_dataset.abstract import MUSDB_AMSS_Training_Dataset, MUSDB_AMSS_Validation_Dataset
from src.data.musdb_amss_dataset.musdb_amss_definitions import musdb_amss_config
from src.utils.functions import build_vocab, build_vocab_dict


class task1_config:
    # Training
    training_amss_set = musdb_amss_config.amss_mute + musdb_amss_config.amss_separate

    # Evaluation
    evaluation_amss_set = [amss for amss in training_amss_set
                           if amss.gen_desc_default() in
                           ['separate vocals',
                            'separate drums',
                            'separate bass',
                            'mute vocals, drums, bass']]

    assert len(evaluation_amss_set) == 4


class Task1_Training_Dataset(MUSDB_AMSS_Training_Dataset):
    def __init__(self, unmixed_dataset):
        super().__init__(unmixed_dataset,
                         task1_config.training_amss_set,
                         musdb_amss_config.vocab,
                         musdb_amss_config.word_to_idx,
                         musdb_amss_config.idx_to_word)


class Task1_Validation_Dataset(MUSDB_AMSS_Validation_Dataset):
    def __init__(self, unmixed_dataset):
        super().__init__(unmixed_dataset,
                         task1_config.evaluation_amss_set,
                         musdb_amss_config.vocab,
                         musdb_amss_config.word_to_idx,
                         musdb_amss_config.idx_to_word)
