import numpy as np
import re
import unittest

import go

def load_board(string):
    reverse_map = {
        'W': go.WHITE,
        '.': go.EMPTY,
        'B': go.BLACK,
        '#': go.FILL,
        '*': go.KO,
        '?': go.UNKNOWN
    }
    string = re.sub(r'[^BW\.#]+', '', string)
    assert len(string) == go.N ** 2, "Board to load didn't have right dimensions"
    board = np.zeros([go.N, go.N], dtype=np.int8)
    for i, char in enumerate(string):
        np.ravel(board)[i] = reverse_map[char]
    return board

class GoPositionTestCase(unittest.TestCase):
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

