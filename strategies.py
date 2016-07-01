import random

import gtp

import go
import utils
import policy, features

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
        coords = utils.parse_pygtp_coords(vertex)
        self.accomodate_out_of_turn(color)
        self.position = self.position.play_move(coords)
        return self.position is not None

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        move = self.suggest_move(self.position)
        return utils.unparse_pygtp_coords(move)

    def suggest_move(self, position):
        raise NotImplementedError

class RandomPlayer(GtpInterface):
    def suggest_move(self, position):
        return random.choice(position.possible_moves())

class PolicyNetworkBestMovePlayer(GtpInterface):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def suggest_move(self, position):
        probabilities = self.network.run(position)
        move_probabilities = {
            utils.unflatten_coords(x): probabilities[x]
            for x in range(361)
        }
        best_move = max(move_probabilities.keys(), key=lambda k: move_probabilities[k])
        return best_move
