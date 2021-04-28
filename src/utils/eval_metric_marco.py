import librosa
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import paired_distances


def getDistances(x, y):
    distances = {}
    distances['mae'] = mean_absolute_error(x, y)
    distances['mse'] = mean_squared_error(x, y)
    distances['euclidean'] = np.mean(paired_distances(x, y, metric='euclidean'))
    distances['manhattan'] = np.mean(paired_distances(x, y, metric='manhattan'))
    distances['cosine'] = np.mean(paired_distances(x, y, metric='cosine'))

    return distances

def getMAEnormalized(ytrue, ypred):
    ratio = np.mean(np.abs(ytrue)) / np.mean(np.abs(ypred))
    return mean_absolute_error(ytrue, ratio * ypred)

# mfcc_cosine


def getMFCC(x, sr, mels=40, mfcc=13, mean_norm=False):
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, S=None,
                                             n_fft=4096, hop_length=2048,
                                             n_mels=mels, power=2.0)
    melspec_dB = librosa.power_to_db(melspec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=melspec_dB, sr=sr, n_mfcc=mfcc)
    if mean_norm:
        mfcc -= (np.mean(mfcc, axis=0))
    return mfcc


def getMSE_MFCC(y_true, y_pred, sr, mels=40, mfcc=13, mean_norm=False):

    y_mfcc = getMFCC(y_true, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    z_mfcc = getMFCC(y_pred, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)

    return getDistances(y_mfcc[:, :], z_mfcc[:, :])

def getMSE_MFCC_mc(y_true, y_pred, sr, mels=40, mfcc=13, mean_norm=False):
    # ratio = np.mean(np.abs(y_true)) / np.mean(np.abs(y_pred))
    # y_pred = ratio * y_pred

    #left
    dist_left = getMSE_MFCC(y_true[:,0], y_pred[: ,0], sr, mels, mfcc, mean_norm)
    dist_right = getMSE_MFCC(y_true[:,1], y_pred[: ,1], sr, mels, mfcc, mean_norm)

    dist = {}
    for key in dist_left.keys():
        dist[key] = dist_left[key] + dist_right[key]
        dist[key] = dist[key] / 2
    return dist
