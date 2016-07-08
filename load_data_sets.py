from collections import namedtuple
import os
import numpy as np
import sys

from features import DEFAULT_FEATURES
import go
import sgf_wrapper
import utils

def make_onehot(dense_labels, num_classes):
    dense_labels = np.fromiter(dense_labels, dtype=np.int16)
    num_labels = dense_labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + dense_labels.ravel()] = 1
    return labels_one_hot

def load_sgf_positions(*dataset_dirs):
    for dataset_dir in dataset_dirs:
        full_dir = os.path.join(os.getcwd(), dataset_dir)
        dataset_files = [os.path.join(full_dir, name) for name in os.listdir(full_dir)]
        all_datafiles = filter(os.path.isfile, dataset_files)
        for file in all_datafiles:
            with open(file) as f:
                sgf = sgf_wrapper.SgfWrapper(f.read())
                for position_w_context in sgf.get_main_branch():
                    if position_w_context.is_usable():
                        yield position_w_context

def partition_sets(stuff):
    number_of_things = len(stuff)
    cutoff = min([number_of_things // 5, 10000])
    test = stuff[:cutoff]
    validation = stuff[cutoff:2*cutoff]
    training = stuff[2*cutoff:]
    return test, validation, training

def bulk_extract(feature_extractor, positions):
    num_positions = len(positions)
    output = np.zeros([num_positions, 19, 19, feature_extractor.planes])
    for i, pos in enumerate(positions):
        output[i] = feature_extractor.extract(pos)
    return output

class DataSet(object):
    def __init__(self, pos_features, next_moves, results, is_test=False):
        self.pos_features = pos_features
        self.next_moves = next_moves
        self.results = results
        assert pos_features.shape[0] == next_moves.shape[0], "Didn't pass in same number of pos_features and next_moves."
        self.data_size = pos_features.shape[0]
        self.input_planes = pos_features.shape[-1]
        self._index_within_epoch = 0

    def get_batch(self, batch_size):
        assert batch_size < self.data_size
        if self._index_within_epoch + batch_size > self.data_size:
            # Shuffle the data and start over
            perm = np.arange(self.data_size)
            np.random.shuffle(perm)
            self.pos_features = self.pos_features[perm]
            self.next_moves = self.next_moves[perm]
            self._index_within_epoch = 0
        start = self._index_within_epoch
        end = start + batch_size
        self._index_within_epoch += batch_size
        return self.pos_features[start:end], self.next_moves[start:end]

DataSets = namedtuple("DataSets", "test validation training input_planes")

def load_data_sets(*dataset_dirs, feature_extractor=DEFAULT_FEATURES):
    print("Extracting positions from sgfs...", file=sys.stderr)
    positions_w_context = list(load_sgf_positions(*dataset_dirs))
    print("Partitioning %s positions into test, validation, training datasets" % len(positions_w_context))
    test, validation, training = partition_sets(positions_w_context)
    print("Processing positions to extract features")
    datasets = []
    for dataset in (test, validation, training):
        positions, next_moves, results = zip(*dataset)
        encoded_moves = make_onehot(map(utils.parse_sgf_to_flat, next_moves), go.N ** 2)
        extracted_features = bulk_extract(feature_extractor, positions)
        datasets.append(DataSet(extracted_features, encoded_moves, results))
    return DataSets(*(datasets + [extracted_features.shape[-1]]))
