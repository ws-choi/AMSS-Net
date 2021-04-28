import random
from itertools import combinations, permutations

import numpy as np

from src.amss.amss_desc.abstract import Selective_Audio_Editing
from src.utils.functions import normalize

targets = ['vocals', 'drums', 'bass']


def get_target_index(target):
    return targets.index(target)


class Musdb_Separate(Selective_Audio_Editing):

    def __init__(self, target_names):
        self.targets = target_names
        self.targets_index = [get_target_index(target) for target in self.targets]
        descriptions = list(permutations(self.targets))
        self.descriptions = []
        for desc in descriptions:
            self.descriptions.append('separate {}'.format(', '.join(desc)))
            self.descriptions.append('isolate {}'.format(', '.join(desc)))
            self.descriptions.append('extract {}'.format(', '.join(desc)))

        self.n_descriptions = len(self.descriptions)

    def edit(self, unmixed_track):
        selected_tracks = [unmixed_track[i] for i in self.targets_index]
        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(selected_tracks, axis=0)

        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return self.gen_desc(), linear_sum, manipulated_linear_sum

    def edit_with_default_desc(self, unmixed_track):
        selected_tracks = [unmixed_track[i] for i in self.targets_index]
        linear_sum = np.sum(unmixed_track, axis=0)
        manipulated_linear_sum = np.sum(selected_tracks, axis=0)
        linear_sum, manipulated_linear_sum = normalize(linear_sum, manipulated_linear_sum)
        return self.gen_desc_default(), linear_sum, manipulated_linear_sum

    def gen_desc(self):
        idx = random.randint(0, self.n_descriptions - 1)
        return self.descriptions[idx]

    def gen_desc_default(self):
        return self.descriptions[0]

    def __str__(self):
        return 'AMSS: ' + self.gen_desc()


def amss_separate_generator():
    amss_set = []
    for i in [1, 2, 3]:
        for comb in combinations(targets, i):
            amss_set.append(Musdb_Separate(comb))

    return amss_set
