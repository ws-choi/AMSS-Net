from src.data.musdb_amss_dataset.abstract import MUSDB_AMSS_Training_Dataset, MUSDB_AMSS_Validation_Dataset
from src.data.musdb_amss_dataset.musdb_amss_definitions import musdb_amss_config


class task2_config:
    training_amss_set = musdb_amss_config.amss_mute + \
                        musdb_amss_config.amss_separate + \
                        musdb_amss_config.amss_pan_control + \
                        musdb_amss_config.amss_effect + \
                        musdb_amss_config.amss_deeffect + \
                        musdb_amss_config.amss_vol_control

    evaluation_amss_set = [amss for amss in training_amss_set
                           if amss.gen_desc_default() in [
                               # Musdb Original Task
                               'separate vocals',  # MUSDB18
                               'separate drums',  # MUSDB18
                               'separate bass',  # MUSDB18
                               'mute vocals, drums, bass',  # MUSDB18

                               # below: 24

                               'pan vocals completely to the left side',
                               'pan drums completely to the left side',
                               'pan bass completely to the left side',

                               'pan vocals completely to the right side',
                               'pan drums completely to the right side',
                               'pan bass completely to the right side',

                               'apply heavy lowpass to vocals',
                               'apply heavy lowpass to drums',
                               'apply heavy lowpass to bass',

                               'apply heavy highpass to vocals',
                               'apply heavy highpass to drums',
                               'apply heavy highpass to bass',

                               'remove reverb from vocals',
                               'remove reverb from drums',
                               'remove reverb from bass',

                               'increase the volume of vocals',
                               'increase the volume of drums',
                               'increase the volume of bass',

                               'decrease the volume of vocals',
                               'decrease the volume of drums',
                               'decrease the volume of bass',

                           ]]


class Task2_Training_Dataset(MUSDB_AMSS_Training_Dataset):
    def __init__(self, unmixed_dataset):
        super().__init__(unmixed_dataset,
                         task2_config.training_amss_set,
                         musdb_amss_config.vocab,
                         musdb_amss_config.word_to_idx,
                         musdb_amss_config.idx_to_word)


class Task2_Validation_Dataset(MUSDB_AMSS_Validation_Dataset):
    def __init__(self, unmixed_dataset):
        super().__init__(unmixed_dataset,
                         task2_config.evaluation_amss_set,
                         musdb_amss_config.vocab,
                         musdb_amss_config.word_to_idx,
                         musdb_amss_config.idx_to_word)
