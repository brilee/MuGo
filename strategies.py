import math
import random
import sys
import time
import numpy as np
import gtp

import go
import utils

def sorted_moves(probability_array):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    return sorted(coords, key=lambda c: probability_array[c], reverse=True)

def translate_gtp_colors(gtp_color):
    if gtp_color == gtp.BLACK:
        return go.BLACK
    elif gtp_color == gtp.WHITE:
        return go.WHITE
    else:
        return go.EMPTY

def is_move_reasonable(position, move):
    return position.is_move_legal(move) and go.is_eyeish(position.board, move) != position.to_play

def select_most_likely(position, move_probabilities):
    for move in sorted_moves(move_probabilities):
        if is_move_reasonable(position, move):
            return move
    return None

def select_weighted_random(position, move_probabilities):
    selection = random.random()
    selected_move = None
    current_probability = 0
    # technically, don't have to sort in order to correctly simulate a random
    # draw, but it cuts down on how many additions we do.
    for move, move_prob in np.ndenumerate(move_probabilities):
        current_probability += move_prob
        if current_probability > selection:
            selected_move = move
            break
    if is_move_reasonable(position, selected_move):
        return selected_move
    else:
        # fallback in case the selected move was illegal
        return select_most_likely(position, move_probabilities)


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
        self.position.komi = komi

    def clear(self):
        self.position = go.Position(komi=self.komi)

    def accomodate_out_of_turn(self, color):
        if not translate_gtp_colors(color) == self.position.to_play:
            self.position.flip_playerturn(mutate=True)

    def make_move(self, color, vertex):
        coords = utils.parse_pygtp_coords(vertex)
        self.accomodate_out_of_turn(color)
        self.position = self.position.play_move(translate_gtp_colors(color), coords)
        return self.position is not None

    def get_move(self, color):
        self.accomodate_out_of_turn(color)
        move = self.suggest_move(self.position)
        return utils.unparse_pygtp_coords(move)

    def suggest_move(self, position):
        raise NotImplementedError

class RandomPlayer(GtpInterface):
    def suggest_move(self, position):
        possible_moves = go.ALL_COORDS[:]
        random.shuffle(possible_moves)
        for move in possible_moves:
            if is_move_reasonable(position, move):
                return move
        return None

class PolicyNetworkBestMovePlayer(GtpInterface):
    def __init__(self, policy_network, read_file):
        self.policy_network = policy_network
        self.read_file = read_file
        super().__init__()

    def clear(self):
        super().clear()
        self.refresh_network()

    def refresh_network(self):
        # Ensure that the player is using the latest version of the network
        # so that the network can be continually trained even as it's playing.
        self.policy_network.initialize_variables(self.read_file)

    def suggest_move(self, position):
        if position.recent and position.n > 100 and position.recent[-1].move == None:
            # Pass if the opponent passes
            return None
        move_probabilities = self.policy_network.run(position)
        return select_most_likely(position, move_probabilities)

class PolicyNetworkRandomMovePlayer(GtpInterface):
    def __init__(self, policy_network, read_file):
        self.policy_network = policy_network
        self.read_file = read_file
        super().__init__()

    def clear(self):
        super().clear()
        self.refresh_network()

    def refresh_network(self):
        # Ensure that the player is using the latest version of the network
        # so that the network can be continually trained even as it's playing.
        self.policy_network.initialize_variables(self.read_file)

    def suggest_move(self, position):
        if position.recent and position.n > 100 and position.recent[-1].move == None:
            # Pass if the opponent passes
            return None
        move_probabilities = self.policy_network.run(position)
        return select_weighted_random(position, move_probabilities)

# Exploration constant
c_PUCT = 5

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
        node = MCTSNode(None, None, 0)
        node.position = position
        node.expand(move_probabilities)
        return node

    def __init__(self, parent, move, prior):
        self.parent = parent # pointer to another MCTSNode
        self.move = move # the move that led to this node
        self.prior = prior
        self.position = None # lazily computed upon expansion
        self.children = {} # map of moves to resulting MCTSNode
        self.Q = self.parent.Q if self.parent is not None else 0 # average of all outcomes involving this node
        self.U = prior # monte carlo exploration bonus
        self.N = 0 # number of times node was visited

    def __repr__(self):
        return "<MCTSNode move=%s prior=%s score=%s is_expanded=%s>" % (self.move, self.prior, self.action_score, self.is_expanded())

    @property
    def action_score(self):
        # Note to self: after adding value network, must calculate 
        # self.Q = weighted_average(avg(values), avg(rollouts)),
        # as opposed to avg(map(weighted_average, values, rollouts))
        return self.Q + self.U

    def is_expanded(self):
        return self.position is not None

    def compute_position(self):
        self.position = self.parent.position.play_move(self.move)
        return self.position

    def expand(self, move_probabilities):
        self.children = {move: MCTSNode(self, move, prob)
            for move, prob in np.ndenumerate(move_probabilities)}
        # Pass should always be an option! Say, for example, seki.
        self.children[None] = MCTSNode(self, None, 0)

    def backup_value(self, value):
        self.N += 1
        if self.parent is None:
            # No point in updating Q / U values for root, since they are
            # used to decide between children nodes.
            return
        self.Q, self.U = (
            self.Q + (value - self.Q) / self.N,
            c_PUCT * math.sqrt(self.parent.N) * self.prior / self.N,
        )
        # must invert, because alternate layers have opposite desires
        self.parent.backup_value(-value)

    def select_leaf(self):
        current = self
        while current.is_expanded():
            current = max(current.children.values(), key=lambda node: node.action_score)
        return current


class MCTS(GtpInterface):
    def __init__(self, policy_network, read_file, seconds_per_move=5):
        self.policy_network = policy_network
        self.seconds_per_move = seconds_per_move
        self.max_rollout_depth = go.N * go.N * 3
        self.read_file = read_file
        super().__init__()

    def clear(self):
        super().clear()
        self.refresh_network()

    def refresh_network(self):
        # Ensure that the player is using the latest version of the network
        # so that the network can be continually trained even as it's playing.
        self.policy_network.initialize_variables(self.read_file)

    def suggest_move(self, position):
        if position.caps[0] + 50 < position.caps[1]:
            return gtp.RESIGN
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
            if len(current.recent) > 2 and current.recent[-1].move == current.recent[-2].move == None:
                break
        else:
            print("max rollout depth exceeded!", file=sys.stderr)

        perspective = 1 if leaf_position.player1turn else -1
        return current.score() * perspective

    def play_valid_move(self, position, move_probs):
        for move in sorted_moves(move_probs):
            if go.is_eyeish(position.board, move):
                continue
            candidate_pos = position.play_move(move, mutate=True)
            if candidate_pos is not None:
                return candidate_pos
        return position.pass_move(mutate=True)
