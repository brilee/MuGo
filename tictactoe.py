from collections import namedtuple
'''
Tic tac toe board format: a namedtuple with properties
a1, a2, a3, b1, b2, b3, c1, c2, c3, player1turn.
The positional properties have values [True, False, None], 
with True meaning player1; False meaning player2.

A move is represented as a string a1~c3.
'''

ALL_POSITIONS = 'a1 a2 a3 b1 b2 b3 c1 c2 c3'.split()
WINNING_LINES = [s.split() for s in (
    'a1 a2 a3',
    'b1 b2 b3',
    'c1 c2 c3',
    'a1 b1 c1',
    'a2 b2 c2',
    'a3 b3 c3',
    'a1 b2 c3',
    'a3 b2 c1',
)]


class TicTacToe(namedtuple('TicTacToe', ALL_POSITIONS + ['player1turn'])):
    def possible_moves(self):
        return [pos for pos in ALL_POSITIONS if getattr(self, pos) is None]

    @property
    def player1wins(self):
        return any(all(getattr(self, pos) is True for pos in line) for line in WINNING_LINES)

    @property
    def player2wins(self):
        return any(all(getattr(self, pos) is False for pos in line) for line in WINNING_LINES)

    def update(self, move):
        if getattr(self, move) is not None:
            raise ValueError("Move already exists at %s in %s" % (move, self))
        return self._replace(**{move: self.player1turn, 'player1turn': not self.player1turn})

    def __repr__(self):
        char_map = {True: 'X', False: 'O', None: '_'}
        return "{}{}{}\n{}{}{}\n{}{}{} to play: {}".format(*(char_map[k] for k in self))

def load_board(s, player1turn):
    'X = player1, O = player2'
    char_map = {'X': True, 'O': False, '_': None}
    s = [char_map[char] for char in s if char in char_map]
    return TicTacToe(*(s + [player1turn]))

EMPTY_BOARD = load_board('___ ___ ___', True)