import copy
import math
import random
import sys
import time

import gtp
import numpy as np

import go
import utils

# Draw moves from policy net until this threshold, then play moves randomly.
# This speeds up the simulation, and it also provides a logical cutoff
# for which moves to include for reinforcement learning.
POLICY_CUTOFF_DEPTH = int(go.N * go.N * 0.75) # 270 moves for a 19x19
# However, some situations end up as "dead, but only with correct play".
# Random play can destroy the subtlety of these situations, so we'll play out
# a bunch more moves from a smart network before playing out random moves.
POLICY_FINISH_MOVES = int(go.N * go.N * 0.2) # 72 moves for a 19x19

def sorted_moves(probability_array):
    coords = [(a, b) for a in range(go.N) for b in range(go.N)]
    coords.sort(key=lambda c: probability_array[c], reverse=True)
    return coords

def is_move_reasonable(position, move):
    # A move is reasonable if it is legal and doesn't fill in your own eyes.
    return position.is_move_legal(move) and go.is_eyeish(position.board, move) != position.to_play

def select_random(position):
    possible_moves = go.ALL_COORDS[:]
    random.shuffle(possible_moves)
    for move in possible_moves:
        if is_move_reasonable(position, move):
            return move
    return None

def select_most_likely(position, move_probabilities):
    for move in sorted_moves(move_probabilities):
        if is_move_reasonable(position, move):
            return move
    return None

def select_weighted_random(position, move_probabilities):
    selection = random.random()
    cdf = move_probabilities.cumsum()
    selected_move = utils.unflatten_coords(
        cdf.searchsorted(selection, side="right"))
    if is_move_reasonable(position, selected_move):
        return selected_move
    else:
        # inexpensive fallback in case an illegal move is chosen.
        return select_most_likely(position, move_probabilities)

def simulate_game_random(position):
    """Simulates a game to termination, using completely random moves"""
    while not (position.recent[-2].move is None and position.recent[-1].move is None):
        position.play_move(select_random(position), mutate=True)

def simulate_game(policy, position):
    """Simulates a game starting from a position, using a policy network"""
    while position.n <= POLICY_CUTOFF_DEPTH:
        move_probs = policy.run(position)
        move = select_weighted_random(position, move_probs)
        position.play_move(move, mutate=True)

    simulate_game_random(position)

    return position

def simulate_many_games(policy1, policy2, positions):
    """Simulates many games in parallel, utilizing GPU parallelization to
    run the policy network for multiple games simultaneously.

    policy1 is black; policy2 is white."""

    # Assumes that all positions are on the same move number. May not be true
    # if, say, we are exploring multiple MCTS branches in parallel
    while positions[0].n <= POLICY_CUTOFF_DEPTH + POLICY_FINISH_MOVES:
        black_to_play = [pos for pos in positions if pos.to_play == go.BLACK]
        white_to_play = [pos for pos in positions if pos.to_play == go.WHITE]

        for policy, to_play in ((policy1, black_to_play),
                                (policy2, white_to_play)):
            all_move_probs = policy.run_many(to_play)
            for i, pos in enumerate(to_play):
                move = select_weighted_random(pos, all_move_probs[i])
                pos.play_move(move, mutate=True)

    for pos in positions:
        simulate_game_random(pos)

    return positions


class RandomPlayerMixin:
    def suggest_move(self, position):
        return select_random(position)

class GreedyPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run(position)
        return select_most_likely(position, move_probabilities)

class RandomPolicyPlayerMixin:
    def __init__(self, policy_network):
        self.policy_network = policy_network
        super().__init__()

    def suggest_move(self, position):
        move_probabilities = self.policy_network.run(position)
        return select_weighted_random(position, move_probabilities)

# All terminology here (Q, U, N, p_UCT) uses the same notation as in the
# AlphaGo paper.
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
        # This incrementally calculates node.Q = average(Q of children),
        # given the newest Q value and the previous average of N-1 values.
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


class MCTSPlayerMixin:
    def __init__(self, policy_network, seconds_per_move=5):
        self.policy_network = policy_network
        self.seconds_per_move = seconds_per_move
        self.max_rollout_depth = go.N * go.N * 3
        super().__init__()

    def suggest_move(self, position):
        start = time.time()
        move_probs = self.policy_network.run(position)
        root = MCTSNode.root_node(position, move_probs)
        while time.time() - start < self.seconds_per_move:
            self.tree_search(root)
        print("Searched for %s seconds" % (time.time() - start), file=sys.stderr)
        sorted_moves = sorted(root.children.keys(), key=lambda move, root=root: root.children[move].N, reverse=True)
        for move in sorted_moves:
            if is_move_reasonable(position, move):
                return move
        return None

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
        value = self.estimate_value(root, chosen_leaf)
        # backup
        print("value: %s" % value, file=sys.stderr)
        chosen_leaf.backup_value(value)
        sys.stderr.flush()

    def estimate_value(self, root, chosen_leaf):
        # Estimate value of position using rollout only (for now).
        # (TODO: Value network; average the value estimations from rollout + value network)
        leaf_position = chosen_leaf.position
        current = copy.deepcopy(leaf_position)
        simulate_game(self.policy_network, current)
        print(current, file=sys.stderr)

        perspective = 1 if leaf_position.to_play == root.position.to_play else -1
        return current.score() * perspective

