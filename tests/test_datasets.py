import os
from test_utils import GoPositionTestCase
import load_data_sets

TEST_DIR = os.path.dirname(os.path.realpath(__file__))
TEMP_FILE_NAME = "dataset_unittest_tempfile"

class TestDataSets(GoPositionTestCase):
    def tearDown(self):
        if os.path.isfile(TEMP_FILE_NAME):
            os.remove(TEMP_FILE_NAME)

    def test_dataset_serialization(self):
        sgf_files = list(load_data_sets.find_sgf_files(TEST_DIR))
        positions_w_context = list(load_data_sets.get_positions_from_sgf(sgf_files[0]))

        opts = [{'compression': c, 'packing': p} for c in ['none', 'gzip6', 'gzip9', 'snappy'] for p in ['none', 'half', 'full']]
        for opt_set in opts:
            dataset = load_data_sets.DataSet.from_positions_w_context(positions_w_context)
            dataset.write(TEMP_FILE_NAME, **opt_set)
            recovered = load_data_sets.DataSet.read(TEMP_FILE_NAME, **opt_set)
            self.assertEqual(dataset.is_test, recovered.is_test)
            self.assertEqual(dataset.data_size, recovered.data_size)
            self.assertEqual(dataset.board_size, recovered.board_size)
            self.assertEqual(dataset.input_planes, recovered.input_planes)
            self.assertEqual(dataset.is_test, recovered.is_test)
            self.assertEqual(dataset.pos_features.shape, recovered.pos_features.shape)
            self.assertEqual(dataset.next_moves.shape, recovered.next_moves.shape)
            self.assertEqualNPArray(dataset.next_moves, recovered.next_moves)
            self.assertEqualNPArray(dataset.pos_features, recovered.pos_features)
