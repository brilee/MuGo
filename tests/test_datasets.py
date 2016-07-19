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
        positions_w_context = list(load_data_sets.load_sgf_positions(TEST_DIR))
        dataset = load_data_sets.DataSet.from_positions_w_context(positions_w_context)
        dataset.write(TEMP_FILE_NAME)
        recovered = load_data_sets.DataSet.read(TEMP_FILE_NAME)
        self.assertEqual(dataset.is_test, recovered.is_test)
        self.assertEqual(dataset.data_size, recovered.data_size)
        self.assertEqual(dataset.board_size, recovered.board_size)
        self.assertEqual(dataset.input_planes, recovered.input_planes)
        self.assertEqual(dataset.is_test, recovered.is_test)
        self.assertEqual(dataset.pos_features.shape, recovered.pos_features.shape)
        self.assertEqual(dataset.next_moves.shape, recovered.next_moves.shape)
        self.assertEqualNPArray(dataset.next_moves, recovered.next_moves)
        self.assertEqualNPArray(dataset.pos_features, recovered.pos_features)
