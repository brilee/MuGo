import random

import gtp

import go

def parse_pygtp_coords(t):
    'Interprets coords in the format (1, 1), with (1,1) being the bottom left'
    if t == (0, 0):
        return None
    rows_from_top = go.N - t[1]
    return go.W + go.W * rows_from_top + t[0] - 1

def unparse_pygtp_coords(c):
    if c is None:
        return (0, 0)
    c = c - go.W
    row, column = divmod(c, go.W)
    return column + 1, go.N - row

class GtpInterface(object):
    def __init__(self):
        self.size = 9
        self.position = None
        self.komi = 6.5
        self.clear()

    def set_size(self, n):
        self.size = n
        go.set_board_size(n)
        self.clear()

    def set_komi(self, komi):
        self.komi = komi
        self.position = self.position._replace(komi=komi)

    def clear(self):
        self.position = go.Position.initial_state()._replace(komi=self.komi)

    def accomodate_out_of_turn(self, color):
        player1turn = (color == gtp.BLACK)
        if player1turn != self.position.player1turn:
            self.position = self.position._replace(player1turn=not self.position.player1turn)

    def make_move(self, color, vertex):
        coords = parse_pygtp_coords(vertex)
        self.accomodate_out_of_turn(color)
        self.position = self.position.play_move(coords)
        return self.position is not None

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        move = self.suggest_move(self.position)
        return unparse_pygtp_coords(move)

    def suggest_move(self, position):
        raise NotImplementedError

class RandomPlayer(GtpInterface):
    def suggest_move(self, position):
        return random.choice(position.possible_moves())
