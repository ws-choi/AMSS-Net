import random
from itertools import combinations, permutations

import numpy as np

from src.amss.amss_desc.abstract import Selective_Audio_Editing
from src.utils.functions import normalize

targets = ['vocals', 'drums', 'bass']


def get_target_index(target):
    return targets.index(target)


class Musdb_Increase_Volume(Selective_Audio_Editing):

    def __init__(self, target_names):
        self.targets = target_names
        self.targets_index = [get_target_index(target) for target in self.targets]
        target_permute = list(permutations(self.targets))
        self.descriptions = ['increase the volume of {}'.format(', '.join(d)) for d in target_permute]
        self.n_descriptions = len(self.descriptions)
        self.__paired_separate__ = 'separate {}'.format(', '.join(target_permute[0]))
        self.__paired_mute__ = 'mute {}'.format(', '.join(target_permute[0]))

    def edit(self, unmixed_track):
        manipulated_track = np.copy(unmixed_track)
        for idx in self.targets_index:
            manipulated_track[idx] *= 2

        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_track, axis=0)

        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return self.gen_desc(), linear_sum, manipulated_linear_sum

    def edit_with_default_desc(self, unmixed_track):
        manipulated_track = np.copy(unmixed_track)
        for idx in self.targets_index:
            manipulated_track[idx] *= 2

        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(manipulated_track, axis=0)

        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return self.gen_desc_default(), linear_sum, manipulated_linear_sum

    def gen_desc(self):
        idx = random.randint(0, self.n_descriptions - 1)
        return self.descriptions[idx]

    def gen_desc_default(self):
        return self.descriptions[0]

    def edit_for_test(self, unmixed_track):
        manipulated_not4return = np.copy(unmixed_track)
        tar_before = np.zeros_like(manipulated_not4return[0])
        tar_after = np.zeros_like(manipulated_not4return[0])
        acc = np.zeros_like(manipulated_not4return[0])

        for idx in range(4):
            if idx in self.targets_index:
                tar_before = tar_before + manipulated_not4return[idx]
                manipulated_not4return[idx] *= 2
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

    def __str__(self):
        return 'AMSS: ' + self.gen_desc_default()

    def gen_paired_mute(self):
        return self.__paired_mute__

    def gen_paired_separate(self):
        return self.__paired_separate__

def amss_increase_generator():
    amss_increase_set = []
    for i in [1, 2, 3]:
        for comb in combinations(targets, i):
            amss_increase_set.append(Musdb_Increase_Volume(comb))

    return amss_increase_set
