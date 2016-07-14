import random
import time

import gtp

import go
import utils

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
            self.position = self.position.flip_playerturn()

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
    def __init__(self, policy_network):
        super().__init__()
        self.policy_network = policy_network

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run(position)
        return move_probabilities[0][1]

class MCTSNode():
    '''
    A MCTSNode has two states: plain, and expanded.
    An plain MCTSNode merely knows its Q + U values, so that a decision
    can be made about which MCTS node to expand during the selection phase.
    An expanded MCTSNode also knows the actual position at that node,
    as well as followup moves/probabilities via the policy network.
    Each of these followup moves is instantiated as a plain MCTSNode.
    '''
    @staticmethod
    def root_node(position, move_probabilities):
        node = MCTSNode(None, None, 1.0)
        node.position = position
        node.expand(move_probabilities)
        return node

    def __init__(self, parent, move, prior):
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.position = None # lazily computed upon expansion
        self.prior = prior
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = self.parent.Q if self.parent is not None else 0.0 # average of all outcomes involving this node
        self.U = prior # monte carlo exploration bonus
        self.N = 0 # number of times node was visited

    @property
    def action_score(self):
        return self.Q + self.U

    def is_expanded(self):
        return self.children == {}

    def compute_position(self):
        self.position = self.parent.position.play_move(self.move)
        return self.position

    def expand(self, move_probabilities):
        self.children = {move: MCTSNode(self, move, prob)
            for prob, move in move_probabilities}

    def backup_value(self, value):
        # Update the average, without having to remember previous values
        self.Q = self.Q + (value - self.Q) / (self.N + 1)
        self.U = self.prior / (1 + self.N)
        self.N += 1
        if self.parent is not None:
            self.parent.backup_value(value)

    def select_leaf(self):
        current = self
        while current.is_expanded():
            current = max(self.children.values(), key=lambda node: node.action_score)
        return current


class MCTS(GtpInterface):
    def __init__(self, policy_network, seconds_per_move=5):
        super().__init__()
        self.policy_network = policy_network
        self.seconds_per_move = seconds_per_move

    def suggest_move(self, position):
        start = time.time()
        move_probs = self.policy_network.run(position)
        root = MCTSNode.root_node(position, move_probs)
        while time.time() - start < self.seconds_per_move:
            self.tree_search(root)
        return max(root.children.keys(), key=lambda move: root.children[move].N)

    def tree_search(self, root):
        # selection
        chosen_leaf = root.select_leaf()
        # expansion
        position = chosen_leaf.compute_position()
        move_probs = self.policy_network.run(position)
        chosen_leaf.expand(move_probs)
        # evaluation
        value = self.estimate_value(chosen_leaf)
        # backup
        chosen_leaf.backup_value(value)

    def estimate_value(self, chosen_leaf):
        # Estimate value of position using rollout only (for now).
        # (TODO: Value network; average the value estimations from rollout + value network)
        return 0
