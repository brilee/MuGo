'''
Features used by AlphaGo, in order of importance.
Feature                 # Notes
Stone colour            3 Player stones; oppo. stones; empty  
Ones                    1 Constant plane of 1s
Turns since last move   8 How many turns since a move played
Liberties               8 Number of liberties
Capture size            8 How many opponent stones would be captured
Self-atari size         8 How many own stones would be captured
Liberties after move    8 Number of liberties after this move played
ladder capture          1 Whether a move is a successful ladder cap
Ladder escape           1 Whether a move is a successful ladder escape
Sensibleness            1 Whether a move is legal + doesn't fill own eye
Zeros                   1 Constant plane of 0s

All features with 8 planes are 1-hot encoded, with plane i marked with 1 
only if the feature was equal to i. Any features >= 8 would be marked as 8.
'''
import itertools
import re

import numpy as np
import go
RAW_BOARD_EXTRACTOR_RE = re.compile(r'[^BW.]+')

def make_onehot(feature, planes):
    onehot_features = np.zeros(feature.shape + (planes,))
    for i in range(planes - 1):
        onehot_features[feature == i+1, i] = 1
    onehot_features[feature >= planes, planes-1] = 1
    return onehot_features

class Feature(object):
    planes = 1
    
    @staticmethod
    def extract(position):
        return np.zeros([go.N, go.N, Feature.planes], dtype=np.float32)

class StoneColorFeature(Feature):
    planes = 3

    @staticmethod
    def extract(position):
        board = position.board
        features = np.zeros([go.N, go.N, 3], dtype=np.float32)
        features[board == go.BLACK, 0] = 1
        features[board == go.WHITE, 1] = 1
        features[board == go.EMPTY, 2] = 1
        return features

class LibertyFeature(Feature):
    '''
    From the AlphaGo paper: 
    Each integer feature value is split into multiple 19 × 19 planes of binary values (one-hot encoding). For example, separate binary feature planes are used to represent whether an intersection has 1 liberty, 2 liberties,..., >=8 liberties.
    '''
    planes = 8
    
    @staticmethod
    def extract(position):
        features = np.zeros([go.N, go.N], dtype=np.float32)
        for g in itertools.chain(*position.groups):
            libs = len(g.liberties)
            for s in g.stones:
                features[s] = libs
        return make_onehot(features, LibertyFeature.planes)
