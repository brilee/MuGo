import numpy as np
import re
import time
import unittest

import go
import utils

go.set_board_size(9)

def load_board(string, player1turn):
    reverse_map = {
        '.': go.EMPTY,
        '#': go.FILL,
        '*': go.KO,
        '?': go.UNKNOWN
    }
    if player1turn:
        reverse_map['X'] = go.BLACK
        reverse_map['O'] = go.WHITE
    else:
        reverse_map['O'] = go.BLACK
        reverse_map['X'] = go.WHITE

    string = re.sub(r'[^XO\.#]+', '', string)
    assert len(string) == go.N ** 2, "Board to load didn't have right dimensions"
    board = np.zeros([go.N, go.N], dtype=np.int8)
    for i, char in enumerate(string):
        np.ravel(board)[i] = reverse_map[char]
    return board

class TestUtils(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(utils.parse_sgf_coords('aa'), (0, 0))
        self.assertEqual(utils.parse_sgf_coords('ac'), (2, 0))
        self.assertEqual(utils.parse_sgf_coords('ca'), (0, 2))
        self.assertEqual(utils.parse_kgs_coords('A1'), (8, 0))
        self.assertEqual(utils.parse_kgs_coords('A9'), (0, 0))
        self.assertEqual(utils.parse_kgs_coords('C2'), (7, 2))
        self.assertEqual(utils.parse_pygtp_coords((1, 1)), (8, 0))
        self.assertEqual(utils.parse_pygtp_coords((1, 9)), (0, 0))
        self.assertEqual(utils.parse_pygtp_coords((3, 2)), (7, 2))
        self.assertEqual(utils.unparse_pygtp_coords((8, 0)), (1, 1))
        self.assertEqual(utils.unparse_pygtp_coords((0, 0)), (1, 9))
        self.assertEqual(utils.unparse_pygtp_coords((7, 2)), (3, 2))

    def test_flatten(self):
        self.assertEqual(utils.flatten_coords((0, 0)), 0)
        self.assertEqual(utils.flatten_coords((0, 3)), 3)
        self.assertEqual(utils.flatten_coords((3, 0)), 27)
        self.assertEqual(utils.unflatten_coords(27), (3, 0))
        self.assertEqual(utils.unflatten_coords(10), (1, 1))
        self.assertEqual(utils.unflatten_coords(80), (8, 8))
        self.assertEqual(utils.flatten_coords(utils.unflatten_coords(10)), 10)
        self.assertEqual(utils.unflatten_coords(utils.flatten_coords((5, 4))), (5, 4))


class GoPositionTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start_time = time.time()

    @classmethod
    def tearDownClass(cls):
        print("\n%s.%s: %.3f seconds" % (cls.__module__, cls.__name__, time.time() - cls.start_time))

    def setUp(self):
        go.set_board_size(9)

    def assertEqualNPArray(self, array1, array2):
        if not np.all(array1 == array2):
            raise AssertionError("Arrays differed in one or more locations:\n%s\n%s" % (array1, array2))
    def assertEqualPositions(self, position1, position2):
        def sort_groups(groups):
            return sorted(groups, key=lambda g: sorted(g.stones) + sorted(g.liberties))
        canonical_p1 = position1._replace(groups=tuple(map(sort_groups, position1.groups)))
        canonical_p2 = position2._replace(groups=tuple(map(sort_groups, position2.groups)))
        self.assertEqualNPArray(canonical_p1.board, canonical_p2.board)
        self.assertEqual(canonical_p1.n, canonical_p2.n)
        self.assertEqual(canonical_p1.groups, canonical_p2.groups)
        self.assertEqual(canonical_p1.caps, canonical_p2.caps)
        self.assertEqual(canonical_p1.ko, canonical_p2.ko)
        r_len = min(len(canonical_p1.recent), len(canonical_p2.recent))
        self.assertEqual(canonical_p1.recent[-r_len:], canonical_p2.recent[-r_len:])
