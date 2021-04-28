import abc

from pysndfx import AudioEffectsChain as fx_chain


class SndFx(metaclass=abc.ABCMeta):

    def __init__(self, fx, fx_name):
        super().__init__()
        self.fx = fx
        self.name = fx_name

    def __call__(self, x):
        track_len = x.shape[0]
        return self.fx(x.T).T[:track_len, ...]


class lowpass(SndFx):

    def __init__(self, fx_name, frequency, q=0.707):
        fx = fx_chain().lowpass(frequency, q=q)

        super().__init__(fx, fx_name)


class highpass(SndFx):

    def __init__(self, fx_name, frequency, q=0.707):
        fx = fx_chain().highpass(frequency, q=q)

        super().__init__(fx, fx_name)


class reverb(SndFx):

    def __init__(self,
                 fx_name,
                 reverberance=50,
                 hf_damping=50,
                 room_scale=100,
                 stereo_depth=100,
                 pre_delay=20,
                 wet_gain=0,
                 wet_only=False):
        fx = fx_chain().reverb(
            reverberance,
            hf_damping,
            room_scale,
            stereo_depth,
            pre_delay,
            wet_gain,
            wet_only
        )

        super().__init__(fx, fx_name)


effects_list_ijcai = [lowpass('lowpass', frequency=344),
                      highpass('highpass', frequency=5512),
                      reverb('reverb',
                             room_scale=100, pre_delay=32, reverberance=100,
                             hf_damping=50, wet_gain=3, stereo_depth=100)
                      ]

effects_dict_ijcai = {snd_fx.name: snd_fx for snd_fx in effects_list_ijcai}
