import unittest

import features
import go
from test_utils import load_board, GoPositionTestCase

go.set_board_size(9)
EMPTY_ROW = '.' * go.N + '\n'
TEST_BOARD = load_board('''
.X.....OO
X........
XXXXXXXXX
''' + EMPTY_ROW * 6, player1turn=True)

TEST_POSITION = go.Position(
    board=TEST_BOARD,
    n=0,
    komi=6.5,
    caps=(1,2),
    groups=go.deduce_groups(TEST_BOARD),
    ko=None,
    last=None,
    last2=None,
    player1turn=True,
)


class TestFeatureExtraction(GoPositionTestCase):
    def test_stone_color_feature(self):
        f = features.StoneColorFeature.extract(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, 3))
        # plane 0 is B
        self.assertEqual(f[0, 1, 0], 1)
        self.assertEqual(f[0, 1, 1], 0)
        # plane 1 is W
        self.assertEqual(f[0, 8, 1], 1)
        self.assertEqual(f[0, 8, 0], 0)
        # plane 2 is empty
        self.assertEqual(f[0, 5, 2], 1)
        self.assertEqual(f[0, 5, 1], 0)

    def test_liberty_feature(self):
        f = features.LibertyFeature.extract(TEST_POSITION)
        self.assertEqual(f.shape, (9, 9, features.LibertyFeature.planes))

        self.assertEqual(f[0, 0, 0], 0)
        # the stone at 0, 1 has 3 liberties.
        self.assertEqual(f[0, 1, 2], 1)
        self.assertEqual(f[0, 1, 4], 0)
        # the group at 0, 7 has 3 liberties
        self.assertEqual(f[0, 7, 2], 1)
        self.assertEqual(f[0, 8, 2], 1)
        # the group at 1, 0 has 18 liberties
        self.assertEqual(f[1, 0, 7], 1)
