from collections import namedtuple
import os
import numpy as np
import struct
import sys

from features import DEFAULT_FEATURES
import go
import sgf_wrapper
import utils

# Number of data points to store in a chunk on disk
CHUNK_SIZE = 4096
CHUNK_HEADER_FORMAT = "iii?"
CHUNK_HEADER_SIZE = struct.calcsize(CHUNK_HEADER_FORMAT)

def iter_chunks(chunk_size, iterable):
    iterator = iter(iterable)
    while True:
        current_chunk = []
        try:
            for i in range(chunk_size):
                current_chunk.append(next(iterator))
            yield current_chunk
        except StopIteration:
            # return the final partial chunk. 
            # If len(iterable) % chunk_size == 0, don't return an empty chunk.
            if current_chunk: yield current_chunk
            break

def make_onehot(dense_labels, num_classes):
    dense_labels = np.fromiter(dense_labels, dtype=np.int16)
    num_labels = dense_labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int16)
    labels_one_hot.flat[index_offset + dense_labels.ravel()] = 1
    return labels_one_hot

def load_sgf_positions(*dataset_dirs):
    for dataset_dir in dataset_dirs:
        full_dir = os.path.join(os.getcwd(), dataset_dir)
        dataset_files = [os.path.join(full_dir, name) for name in os.listdir(full_dir)]
        all_datafiles = [f for f in dataset_files if os.path.isfile(f) and f.endswith(".sgf")]
        for file in all_datafiles:
            with open(file) as f:
                sgf = sgf_wrapper.SgfWrapper(f.read())
                for position_w_context in sgf.get_main_branch():
                    if position_w_context.is_usable():
                        yield position_w_context

def bulk_extract(feature_extractor, positions):
    num_positions = len(positions)
    output = np.zeros([num_positions, go.N, go.N, feature_extractor.planes], dtype=np.float32)
    for i, pos in enumerate(positions):
        output[i] = feature_extractor.extract(pos)
    return output

class DataSet(object):
    def __init__(self, pos_features, next_moves, results, is_test=False):
        self.pos_features = pos_features
        self.next_moves = next_moves
        self.results = results
        self.is_test = is_test
        assert pos_features.shape[0] == next_moves.shape[0], "Didn't pass in same number of pos_features and next_moves."
        self.data_size = pos_features.shape[0]
        self.board_size = pos_features.shape[1]
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

    @staticmethod
    def from_positions_w_context(positions_w_context, is_test=False):
        positions, next_moves, results = zip(*positions_w_context)
        extracted_features = bulk_extract(DEFAULT_FEATURES, positions)
        encoded_moves = make_onehot(map(utils.parse_sgf_to_flat, next_moves), go.N ** 2)
        return DataSet(extracted_features, encoded_moves, results, is_test=is_test)

    def write(self, filename):
        header_bytes = struct.pack(CHUNK_HEADER_FORMAT, self.data_size, self.board_size, self.input_planes, self.is_test)
        position_bytes = self.pos_features.tostring()
        next_move_bytes = self.next_moves.tostring()
        with open(filename, "wb") as f:
            f.write(header_bytes)
            f.write(position_bytes)
            f.write(next_move_bytes)

    @staticmethod
    def read(filename):
        with open(filename, "rb") as f:
            header_bytes = f.read(CHUNK_HEADER_SIZE)
            data_size, board_size, input_planes, is_test = struct.unpack(CHUNK_HEADER_FORMAT, header_bytes)
            position_bytes = f.read(data_size * board_size * board_size * input_planes * 4)
            next_move_bytes = f.read(data_size * board_size * board_size * 2)
            pos_features = np.fromstring(position_bytes, dtype=np.float32).reshape(data_size, board_size, board_size, input_planes)
            next_moves = np.fromstring(next_move_bytes, dtype=np.int16).reshape(data_size, board_size * board_size)
        return DataSet(pos_features, next_moves, [], is_test=is_test)

def process_raw_data(*dataset_dirs):
    print("Extracting positions from sgfs...", file=sys.stderr)
    positions_w_context = load_sgf_positions(*dataset_dirs)
    print("Partitioning positions into test, training datasets")
    data_chunks = iter_chunks(CHUNK_SIZE, positions_w_context)
    first_chunk = next(data_chunks)
    if len(first_chunk) != CHUNK_SIZE:
        test_size = len(first_chunk) // 2
        test_chunk, training_chunks = first_chunk[:test_size], [first_chunk[test_size:]]
        print("Allocating %s positions as test; %s positions as training" % (test_size, len(first_chunk) - test_size), file=sys.stderr)
    else:
        test_chunk, training_chunks = first_chunk, data_chunks
        print("Allocating %s positions as test; remainder as training" % CHUNK_SIZE, file=sys.stderr)
    print("Processing positions to extract features", file=sys.stderr)
    test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
    training_datasets = map(DataSet.from_positions_w_context, training_chunks)
    return test_dataset, training_datasets
