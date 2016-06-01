import unittest
from tictactoe import ALL_POSITIONS, load_board

class TestTicTacToe(unittest.TestCase):
    def test_possible_moves(self):
        t = load_board('___ ___ ___', True)
        self.assertEqual(t.possible_moves(), ALL_POSITIONS)

        t = load_board('XXX ___ ___', True)
        self.assertEqual(t.possible_moves(), 'b1 b2 b3 c1 c2 c3'.split())

    def test_win_conditions(self):
        t = load_board('_X_ OXO _XO', True)
        self.assertTrue(t.player1wins)
        self.assertFalse(t.player2wins)

        t = load_board('XXX OOO ___', True)
        self.assertTrue(t.player1wins)
        self.assertTrue(t.player2wins)

        t = load_board('OOO _XX X__', True)
        self.assertFalse(t.player1wins)
        self.assertTrue(t.player2wins)

    def test_updates(self):
        t = load_board('___ ___ ___', True)
        expected_board = load_board('X__ ___ ___', False)
        new_board = t.update('a1')
        self.assertEqual(new_board, expected_board)

        t = load_board('XXX ___ ___', True)
        with self.assertRaises(ValueError):
            t.update('a1')



