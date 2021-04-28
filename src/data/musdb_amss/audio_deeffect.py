import random
from itertools import permutations, combinations

import numpy as np

from src.amss.amss_desc.abstract import Selective_Audio_Editing
from src.amss.amss_desc.sound_effects import SndFx
from src.data.musdb_amss.audio_effect import effects_dict_ijcai
from src.utils.functions import normalize

targets = ['vocals', 'drums', 'bass']


def get_target_index(target):
    return targets.index(target)


class Musdb_DeEffect(Selective_Audio_Editing):

    def __init__(self, snd_fx: SndFx, target_names):
        self.targets = target_names
        self.targets_index = [get_target_index(target) for target in self.targets]
        self.snd_fx = snd_fx
        target_permute = list(permutations(self.targets))
        self.descriptions = ['remove {} from {}'.format(snd_fx.name, ', '.join(d))
                             for d
                             in target_permute]
        self.n_descriptions = len(self.descriptions)

        self.__paired_separate__ = 'separate {}'.format(', '.join(target_permute[0]))
        self.__paired_mute__ = 'mute {}'.format(', '.join(target_permute[0]))


    def edit(self, unmixed_track: np.ndarray):
        manipulated_track = np.copy(unmixed_track)

        for idx in range(3):
            if idx in self.targets_index:
                # unmixed_track[idx] = self.snd_fx(unmixed_track[idx])

                # else:
                manipulated_track[idx] = self.snd_fx(manipulated_track[idx])
                # unmixed_track[idx] = self.snd_fx(unmixed_track[idx])

        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_track, axis=0)

        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return self.gen_desc(), manipulated_linear_sum, linear_sum

    def edit_with_default_desc(self, unmixed_track):
        manipulated_track = np.copy(unmixed_track)
        for idx in range(3):
            if idx in self.targets_index:
                # unmixed_track[idx] = self.snd_fx(unmixed_track[idx])

                # else:
                manipulated_track[idx] = self.snd_fx(manipulated_track[idx])
                # unmixed_track[idx] = self.snd_fx(unmixed_track[idx])

        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_track, axis=0)

        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return self.gen_desc_default(), manipulated_linear_sum, linear_sum

    def edit_for_test(self, unmixed_track):
        manipulated_linear_sum = np.copy(unmixed_track)
        tar_before = np.zeros_like(manipulated_linear_sum[0])
        tar_after = np.zeros_like(manipulated_linear_sum[0])
        acc = np.zeros_like(manipulated_linear_sum[0])

        for idx in range(4):

            if idx in self.targets_index:
                tar_after = tar_after + manipulated_linear_sum[idx]
                manipulated_linear_sum[idx] = self.snd_fx(manipulated_linear_sum[idx])
                tar_before = tar_before + manipulated_linear_sum[idx]
            else:
                acc = acc + manipulated_linear_sum[idx]

        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_linear_sum, axis=0)
        max_scale = max(linear_sum.max(), manipulated_linear_sum.max())
        max_scale = 1 if max_scale < 1 else max_scale

        linear_sum, manipulated_linear_sum = linear_sum / max_scale, manipulated_linear_sum / max_scale
        tar_before, tar_after, acc = tar_before / max_scale, tar_after / max_scale, acc / max_scale

        after, before = linear_sum, manipulated_linear_sum

        return before, after, tar_before, tar_after, acc

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


def amss_deeffect_generator(effect_names=effects_dict_ijcai.keys()):
    amss_deeffect_set = []
    for i in [1, 2, 3]:
        for comb in combinations(targets, i):
            for effect_name in effect_names:
                if 'reverb' in effect_name:
                    amss_deeffect_set.append(Musdb_DeEffect(effects_dict_ijcai[effect_name], comb))

    return amss_deeffect_set
