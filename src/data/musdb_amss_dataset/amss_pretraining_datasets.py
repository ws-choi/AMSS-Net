from src.data.musdb_amss import amss_mute_generator, amss_separate_generator
from src.data.musdb_amss_dataset.abstract import MUSDB_AMSS_Training_Dataset, MUSDB_AMSS_Validation_Dataset
from src.utils.functions import build_vocab, build_vocab_dict


class pretraining_config:
    amss_mute = amss_mute_generator()
    amss_separate = amss_separate_generator()

    # Training
    training_amss_set = amss_mute + amss_separate

    # Evaluation
    evaluation_amss_set = [amss for amss in amss_separate
                           if amss.gen_desc_default() in ['separate vocals', 'separate drums', 'separate bass']]
    evaluation_amss_set += [amss for amss in amss_mute
                            if amss.gen_desc_default() == 'mute vocals, drums, bass']

    assert len(evaluation_amss_set) == 4


class Task1_Training_Dataset(MUSDB_AMSS_Training_Dataset):
    def __init__(self, unmixed_dataset):
        super().__init__(unmixed_dataset,
                         pretraining_config.training_amss_set,
                         pretraining_config.vocab,
                         pretraining_config.word_to_idx,
                         pretraining_config.idx_to_word)


class Task1_Validation_Dataset(MUSDB_AMSS_Validation_Dataset):
    def __init__(self, unmixed_dataset):
        super().__init__(unmixed_dataset,
                         pretraining_config.evaluation_amss_set,
                         pretraining_config.vocab,
                         pretraining_config.word_to_idx,
                         pretraining_config.idx_to_word)