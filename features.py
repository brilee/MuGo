'''
Features used by AlphaGo, in approximate order of importance.
Feature                 # Notes
Stone colour            3 Player stones; oppo. stones; empty  
Turns since last move   8 How many turns since a move played
Liberties               8 Number of liberties
Capture size            8 How many opponent stones would be captured
Self-atari size         8 How many own stones would be captured
Liberties after move    8 Number of liberties after this move played
ladder capture          1 Whether a move is a successful ladder cap
Ladder escape           1 Whether a move is a successful ladder escape
Sensibleness            1 Whether a move is legal + doesn't fill own eye
Zeros                   1 Constant plane of 0s
Ones                    1 Constant plane of 1s

All features with 8 planes are 1-hot encoded, with plane i marked with 1 
only if the feature was equal to i. Any features >= 8 would be marked as 8.
'''
import itertools

import numpy as np
import go

def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,), dtype=np.float32)
    for i in range(planes - 1):
        onehot_features[:, :, i] = (feature == i+1)
    onehot_features[:, :, planes-1] = (feature >= planes)
    return onehot_features

def planes(num_planes):
    def deco(f):
        f.planes = num_planes
        return f
    return deco

@planes(3)
def stone_color_feature(position):
    board = position.board
    features = np.zeros([go.N, go.N, 3], dtype=np.float32)
    features[board == go.BLACK, 0] = 1
    features[board == go.WHITE, 1] = 1
    features[board == go.EMPTY, 2] = 1
    return features

@planes(8)
def recent_move_feature(position):
    p = 8
    onehot_features = np.zeros([go.N, go.N, p], dtype=np.float32)
    for i, move in enumerate(reversed(position.recent[-p:])):
        if move is not None:
            onehot_features[move[0], move[1], i] = 1
    return onehot_features

@planes(8)
def liberty_feature(position):
    features = np.zeros([go.N, go.N], dtype=np.float32)
    for g in itertools.chain(*position.groups):
        libs = len(g.liberties)
        for s in g.stones:
            features[s] = libs
    return make_onehot(features, 8)


class FeatureExtractor(object):
    def __init__(self, features):
        self.features = features
        self.planes = sum(f.planes for f in features)

    def extract(self, position):
        return np.concatenate([feature(position) for feature in self.features], axis=2)

DEFAULT_FEATURES = FeatureExtractor([stone_color_feature, liberty_feature, recent_move_feature])