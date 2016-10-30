import unittest
from go import Position, BLACK
from strategies import is_move_reasonable
from test_utils import load_board
from utils import parse_kgs_coords as pc

def pc_set(string):
    return set(map(pc, string.split()))

class TestHelperFunctions(unittest.TestCase):
    def test_is_move_reasonable(self):
        board = load_board('''
            .XXOOOXXX
            X.XO.OX.X
            XXXOOOXX.
            ...XXX..X
            XXXX.....
            OOOX....O
            X.OXX.OO.
            .XO.X.O.O
            XXO.X.OO.
        ''')
        position = Position(
            board=board,
            to_play=BLACK,
        )
        reasonable_moves = pc_set('E8 B3')
        unreasonable_moves = pc_set('A9 B8 H8 J7 A2 J3 H2 J1')
        for move in reasonable_moves:
            self.assertTrue(is_move_reasonable(position, move), str(move))
        for move in unreasonable_moves:
            self.assertFalse(is_move_reasonable(position, move), str(move))

