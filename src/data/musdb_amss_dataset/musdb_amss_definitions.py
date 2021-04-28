from src.data.musdb_amss import amss_mute_generator, amss_separate_generator
from src.data.musdb_amss.audio_deeffect import amss_deeffect_generator
from src.data.musdb_amss.audio_effect import amss_effect_generator
from src.data.musdb_amss.pan_to_left import amss_pan2left_generator
from src.data.musdb_amss.pan_to_left_hard import amss_pan2left_hard_generator
from src.data.musdb_amss.pan_to_right import amss_pan2right_generator
from src.data.musdb_amss.pan_to_right_hard import amss_pan2right_hard_generator
from src.data.musdb_amss.volume_decrease import amss_decrease_generator
from src.data.musdb_amss.volume_increase import amss_increase_generator
from src.utils.functions import build_vocab, build_vocab_dict


class musdb_amss_config:
    amss_mute = amss_mute_generator()
    amss_separate = amss_separate_generator()

    amss_vol_control = amss_decrease_generator() + amss_increase_generator()
    amss_pan_control = amss_pan2left_generator() + amss_pan2right_generator()\
                       + amss_pan2left_hard_generator() + amss_pan2right_hard_generator()

    amss_effect = amss_effect_generator()
    amss_deeffect = amss_deeffect_generator()

    vocab = build_vocab(amss_mute +
                        amss_separate +
                        amss_vol_control +
                        amss_pan_control +
                        amss_effect +
                        amss_deeffect)

    word_to_idx, idx_to_word = build_vocab_dict(vocab)
