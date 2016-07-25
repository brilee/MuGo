import random
import sys
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
        possible_moves = go.ALL_COORDS
        random.shuffle(possible_moves)
        for move in possible_moves:
            if position.play_move(move) is not None:
                return move
        return None

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
    When expanded, a MCTSNode also knows the actual position at that node,
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
        self.prior = prior
        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = self.parent.Q if self.parent is not None else 0.0 # average of all outcomes involving this node
        self.U = prior # monte carlo exploration bonus
        self.N = 0 # number of times node was visited

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        return self.Q + self.U

    def is_expanded(self):
        return self.position is not None

    def compute_position(self):
        self.position = self.parent.position.play_move(self.move)
        return self.position

    def expand(self, move_probabilities):
        self.children = {move: MCTSNode(self, move, prob)
            for prob, move in move_probabilities}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSNode(self, None, 0)

    def backup_value(self, value):
        # Update the average, without having to remember previous values
        self.Q, self.U, self.N = (
            self.Q + (value - self.Q) / (self.N + 1),
            self.prior / (1 + self.N),
            self.N + 1
        )
        if self.parent is not None:
            # must invert, because alternate layers have opposite desires
            self.parent.backup_value(-value)

    def select_leaf(self):
        current = self
        while current.is_expanded():
            current = max(current.children.values(), key=lambda node: node.action_score)
        return current


class MCTS(GtpInterface):
    def __init__(self, policy_network, seconds_per_move=5):
        super().__init__()
        self.policy_network = policy_network
        self.seconds_per_move = seconds_per_move
        self.max_rollout_depth = go.N * go.N * 3

    def suggest_move(self, position):
        start = time.time()
        move_probs = self.policy_network.run(position)
        root = MCTSNode.root_node(position, move_probs)
        while time.time() - start < self.seconds_per_move:
            self.tree_search(root)
        # there's a theoretical bug here: if you refuse to pass, this AI will
        # eventually start filling in its own eyes.
        return max(root.children.keys(), key=lambda move, root=root: root.children[move].N)

    def tree_search(self, root):
        print("tree search", file=sys.stderr)
        # selection
        chosen_leaf = root.select_leaf()
        # expansion
        position = chosen_leaf.compute_position()
        if position is None:
            print("illegal move!", file=sys.stderr)
            # See go.Position.play_move for notes on detecting legality
            del chosen_leaf.parent.children[chosen_leaf.move]
            return
        print("Investigating following position:\n%s" % (chosen_leaf.position,), file=sys.stderr)
        move_probs = self.policy_network.run(position)
        chosen_leaf.expand(move_probs)
        # evaluation
        value = self.estimate_value(chosen_leaf)
        # backup
        print("value: %s" % value, file=sys.stderr)
        chosen_leaf.backup_value(value)

    def estimate_value(self, chosen_leaf):
        # Estimate value of position using rollout only (for now).
        # (TODO: Value network; average the value estimations from rollout + value network)
        leaf_position = chosen_leaf.position
        current = leaf_position
        while current.n < self.max_rollout_depth:
            move_probs = self.policy_network.run(current)
            current = self.play_valid_move(current, move_probs)
            if len(current.recent) > 2 and current.recent[-1] == current.recent[-2] == None:
                break
        else:
            print("max rollout depth exceeded!", file=sys.stderr)

        perspective = 1 if leaf_position.player1turn else -1
        return current.score() * perspective

    def play_valid_move(self, position, move_probs):
        for _, move in move_probs:
            if go.is_eye(position.board, move):
                continue
            candidate_pos = position.play_move(move)
            if candidate_pos is not None:
                return candidate_pos
        return position.pass_move()
