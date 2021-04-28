import random
from itertools import permutations, combinations

import numpy as np

from src.amss.amss_desc.abstract import Selective_Audio_Editing
from src.amss.amss_desc.sound_effects import effects_dict_ijcai, SndFx, lowpass, highpass
from src.utils.functions import normalize

targets = ['vocals', 'drums', 'bass']


def get_target_index(target):
    return targets.index(target)


class Musdb_Effect(Selective_Audio_Editing):

    def __init__(self, snd_fx: SndFx, target_names):
        self.targets = target_names
        self.targets_index = [get_target_index(target) for target in self.targets]
        self.snd_fx = snd_fx
        target_permute = list(permutations(self.targets))
        self.descriptions = ['apply {} to {}'.format(snd_fx.name, ', '.join(d))
                             for d
                             in target_permute]
        self.n_descriptions = len(self.descriptions)

        self.__paired_separate__ = 'separate {}'.format(', '.join(target_permute[0]))
        self.__paired_mute__ = 'mute {}'.format(', '.join(target_permute[0]))

    def __edit__(self, unmixed_track: np.ndarray):
        manipulated_track = np.copy(unmixed_track)
        for idx in self.targets_index:
            manipulated_track[idx] = self.snd_fx(manipulated_track[idx])
        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_track, axis=0)
        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return linear_sum, manipulated_linear_sum

    def edit(self, unmixed_track: np.ndarray):
        linear_sum, manipulated_linear_sum = self.__edit__(unmixed_track)
        return self.gen_desc(), linear_sum, manipulated_linear_sum

    def edit_with_default_desc(self, unmixed_track: np.ndarray):
        linear_sum, manipulated_linear_sum = self.__edit__(unmixed_track)
        return self.gen_desc_default(), linear_sum, manipulated_linear_sum

    def edit_for_test(self, unmixed_track):
        manipulated_not4return = np.copy(unmixed_track)
        tar_before = np.zeros_like(manipulated_not4return[0])
        tar_after = np.zeros_like(manipulated_not4return[0])
        acc = np.zeros_like(manipulated_not4return[0])

        for idx in range(4):
            if idx in self.targets_index:
                tar_before = tar_before + manipulated_not4return[idx]
                manipulated_not4return[idx] = self.snd_fx(manipulated_not4return[idx])
                tar_after = tar_after + manipulated_not4return[idx]
            else:
                acc = acc + manipulated_not4return[idx]

        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_not4return, axis=0)
        max_scale = max(linear_sum.max(), manipulated_linear_sum.max())
        max_scale = 1 if max_scale < 1 else max_scale

        linear_sum, manipulated_linear_sum = linear_sum / max_scale, manipulated_linear_sum / max_scale
        tar_before, tar_after, acc = tar_before / max_scale, tar_after / max_scale, acc / max_scale

        return linear_sum, manipulated_linear_sum, tar_before, tar_after, acc

    def gen_desc(self):
        idx = random.randint(0, self.n_descriptions - 1)
        return self.descriptions[idx]

    def gen_desc_default(self):
        return self.descriptions[0]

    def gen_paired_mute(self):
        return self.__paired_mute__

    def gen_paired_separate(self):
        return self.__paired_separate__

    def __str__(self):
        return 'AMSS: ' + self.gen_desc()


def amss_effect_generator():
    p= 10
    light_lp = lowpass('light lowpass', 1378, p)
    medium_lp = lowpass('medium lowpass', 689, p)
    heavy_lp = lowpass('heavy lowpass', 344, p)

    light_lp_b = lowpass('light lowpass', 172, p)
    medium_lp_b = lowpass('medium lowpass', 86, p)
    heavy_lp_b = lowpass('heavy lowpass', 43, p)

    light_hp = highpass('light highpass', 1378, p)
    medium_hp = highpass('medium highpass', 2756, p)
    heavy_hp = highpass('heavy highpass', 5512, p)

    light_hp_b = highpass('light highpass', 86, p)
    medium_hp_b = highpass('medium highpass', 172, p)
    heavy_hp_b = highpass('heavy highpass', 344, p)

    amss_effect_set = []

    for f, f_bass in \
            zip(
                [light_lp, medium_lp, heavy_lp, light_hp, medium_hp, heavy_hp],
                [light_lp_b, medium_lp_b, heavy_lp_b, light_hp_b, medium_hp_b, heavy_hp_b],
            ):
        amss_effect_set.append(Musdb_Effect(f, ['vocals']))
        amss_effect_set.append(Musdb_Effect(f, ['drums']))
        amss_effect_set.append(Musdb_Effect(f, ['vocals', 'drums']))
        amss_effect_set.append(Musdb_Effect(f_bass, ['bass']))

    return amss_effect_set
