'''
Conventions used: "B" to play, always. "W" is the opponent.
Traditional Black and White players are instead referred to as X and O,
or equivalently, "player 1" and "player 2".
All positions are stored in B to play notation, for easier feeding into
neural networks.

A Coordinate is a tuple index into the board.
A Move is a (Coordinate c | None).

When representing the numpy array as a board, (0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.

'''
from collections import namedtuple
import itertools

import numpy as np

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)

# these are initialized by set_board_size
N = None
ALL_COORDS = []
EMPTY_BOARD = None
NEIGHBORS = {}
DIAGONALS = {}

def set_board_size(n):
    '''
    Hopefully nobody tries to run both 9x9 and 19x19 game instances at once.
    Also, never do "from go import N, W, ALL_COORDS, EMPTY_BOARD".
    '''
    global N, ALL_COORDS, EMPTY_BOARD, NEIGHBORS, DIAGONALS
    if N == n: return
    N = n
    ALL_COORDS = [(i, j) for i in range(n) for j in range(n)]
    EMPTY_BOARD = np.zeros([n, n], dtype=np.int8)
    def check_bounds(c):
        return c[0] % n == c[0] and c[1] % n == c[1]

    NEIGHBORS = {(x, y): list(filter(check_bounds, [(x+1, y), (x-1, y), (x, y+1), (x, y-1)])) for x, y in ALL_COORDS}
    DIAGONALS = {(x, y): list(filter(check_bounds, [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)])) for x, y in ALL_COORDS}

def place_stone(board, color, c):
    new_board = np.copy(board)
    new_board[c] = color
    return new_board

def capture_stones(board, stones):
    for s in stones:
        board[s] = EMPTY

def flood_fill(board, c):
    'From a starting coordinate c, flood-fill (mutate) the board with FILL'
    color = board[c]
    entire_group = set([c])
    frontier = [c]
    while frontier:
        current = frontier.pop()
        board[current] = FILL
        for n in NEIGHBORS[current]:
            if board[n] == color:
                frontier.append(n)
                entire_group.add(n)
    return entire_group

def find_neighbors(color, board, stones):
    'Find all neighbors of a set of stones of a given color'
    potential_neighbors = set(itertools.chain(*(NEIGHBORS[s] for s in stones)))
    equal_color = board == color
    return {c for c in potential_neighbors if equal_color[c]}

def find_liberties(board, stones):
    'Given a board and a set of stones, find liberties of those stones'
    return find_neighbors(EMPTY, board, stones)

def is_koish(board, c):
    'Check if c is surrounded on all sides by 1 color, and return that color'
    if board[c] != EMPTY: return None
    neighbors = {board[n] for n in NEIGHBORS[c]}
    if len(neighbors) == 1 and not EMPTY in neighbors:
        return list(neighbors)[0]
    else:
        return None

def is_eye(board, c):
    'Check if c is an eye, for the purpose of restricting MC rollouts.'
    color = is_koish(board, c)
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[c]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if not (board[d] == color or is_koish(board, d) == color):
            diagonal_faults += 1
    if diagonal_faults > 1:
        return None
    else:
        return color

class Group(namedtuple('Group', 'stones liberties')):
    '''
    stones: a set of Coordinates belonging to this group
    liberties: a set of Coordinates that are empty and adjacent to this group.
    '''
    pass


def deduce_groups(board):
    'Given a board, return a 2-tuple; a list of groups for each player'
    def find_groups(board, color):
        board = np.copy(board)
        groups = []
        while color in board:
            remaining_stones = np.where(board == color)
            c = remaining_stones[0][0], remaining_stones[1][0]
            stones = flood_fill(board, c)
            liberties = find_liberties(board, stones)
            groups.append(Group(stones=set(stones), liberties=liberties))
        return groups

    return find_groups(board, BLACK), find_groups(board, WHITE)

def update_groups(board, existing_B_groups, existing_W_groups, c):
    '''
    When a move is played, update the list of groups and their liberties.
    This means possibly appending the new move to a group, creating a new 1-stone group, or merging existing groups.
    The returned groups represent the state after the move has been played,
    but before captures are processed.
    '''
    updated_B_groups, groups_to_merge = [], []
    for g in existing_B_groups:
        if c in g.liberties:
            groups_to_merge.append(g)
        else:
            updated_B_groups.append(g)

    new_stones = {c}
    new_liberties = set(n for n in NEIGHBORS[c] if board[n] == EMPTY)
    for g in groups_to_merge:
        new_stones = new_stones | g.stones
        new_liberties = new_liberties | g.liberties
    new_liberties = new_liberties - {c}
    updated_B_groups.append(Group(stones=new_stones, liberties=new_liberties))

    updated_W_groups = []
    for g in existing_W_groups:
        if c in g.liberties:
            updated_W_groups.append(Group(stones=g.stones, liberties=g.liberties - {c}))
        else:
            updated_W_groups.append(g)

    return updated_B_groups, updated_W_groups

class Position(namedtuple('Position', 'board n komi caps groups ko recent player1turn')):
    '''
    board: a numpy array, with B to play.
    n: an int representing moves played so far
    komi: a float, representing points given to the second player.
    caps: a (int, int) tuple of captures; caps[0] is the person to play (B).
    groups: a (list(Group), list(Group)) tuple of lists of Groups; groups[0] represents the groups of the person to play.
    ko: a Move
    recent: a tuple of Moves, such that recent[-1] is the last move.
    player1turn: whether player 1 is B.
    '''
    @staticmethod
    def initial_state():
        return Position(EMPTY_BOARD, n=0, komi=7.5, caps=(0, 0), groups=([], []), ko=None, recent=tuple(), player1turn=True)

    def __str__(self):
        pretty_print_map = {
            WHITE: 'O',
            EMPTY: '.',
            BLACK: 'X',
            FILL: '#',
            KO: '*',
        }
        if not self.player1turn:
            board = self.board * -1
            captures = self.caps[1], self.caps[0]
        else:
            board = self.board
            captures = self.caps
        if self.ko is not None:
            board = place_stone(board, KO, self.ko)
        raw_board_contents = []
        for i in range(N):
            row = []
            for j in range(N):
                appended = '<' if (self.recent and (i, j) == self.recent[-1]) else ' '
                row.append(pretty_print_map[board[i,j]] + appended)
            raw_board_contents.append(''.join(row))

        row_labels = ['%2d ' % i for i in range(N, 0, -1)]
        annotated_board_contents = [''.join(r) for r in zip(row_labels, raw_board_contents, row_labels)]
        header_footer_rows = ['   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:N]) + '   ']
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures X: {} O: {}\n".format(self.n, *captures)
        return annotated_board + details

    def pass_move(self):
        return Position(
            board=self.board * -1,
            n=self.n+1,
            komi=-self.komi,
            caps=(self.caps[1], self.caps[0]),
            groups=(self.groups[1], self.groups[0]),
            ko=None,
            recent=self.recent + (None,),
            player1turn=not self.player1turn,
        )

    def flip_playerturn(self):
        return Position(
            board=self.board * -1,
            n=self.n,
            komi=-self.komi,
            caps=(self.caps[1], self.caps[0]),
            groups=(self.groups[1], self.groups[0]),
            ko=self.ko,
            recent=self.recent,
            player1turn=not self.player1turn,
        )

    def play_move(self, c):
        # Obeys CGOS Rules of Play. In short:
        # No suicides
        # Chinese/area scoring
        # Positional superko (this is very crudely approximate at the moment.)

        # Checking a move for legality is actually very expensive, because 
        # the only way to reliably handle all suicide/capture situations is to
        # actually play the move and see if any issues arise.
        # Thus, there is no "is_legal(self, move)" or "get_legal_moves(self)".
        # You can only play the move and check if the return value is None.
        if c is None:
            return self.pass_move()
        if c == self.ko:
            return None
        if self.board[c] != EMPTY:
            return None

        # Convention: B's stone is played. All B/W groups continue to be
        # referred to as B and W. At the end, the return position is flipped.
        B_groups, W_groups = self.groups

        working_board = place_stone(self.board, BLACK, c)
        new_B_groups, new_W_groups = update_groups(working_board, B_groups, W_groups, c)

        # process W's captures first, then your own suicides.
        # As stones are removed, liberty counts become inaccurate.
        W_captured = set()
        final_W_groups = []
        for group in new_W_groups:
            if not group.liberties:
                W_captured |= group.stones
                capture_stones(working_board, group.stones)
            else:
                final_W_groups.append(group)

        if W_captured:
            # recalculate liberties for groups adjacent to a captured W group
            coords_with_updates = find_neighbors(BLACK, working_board, W_captured)
            final_B_groups = [g if not (g.stones & coords_with_updates)
                else Group(stones=g.stones, liberties=find_liberties(working_board, g.stones))
                for g in new_B_groups]
        else:
            # Check for suicide. Can only happen if there were no captures.
            if not all(g.liberties for g in new_B_groups):
                return None
            final_B_groups = new_B_groups


        if len(W_captured) == 1 and is_koish(self.board, c) == WHITE:
            ko = list(W_captured)[0]
        else:
            ko = None

        return Position(
            board=working_board * -1,
            n=self.n + 1,
            komi=-self.komi,
            caps=(self.caps[1], self.caps[0] + len(W_captured)),
            groups=(final_W_groups, final_B_groups),
            ko=ko,
            recent=self.recent + (c,),
            player1turn=not self.player1turn,
        )

    def score(self):
        'Returns score from player 1 perspective'
        working_board = np.copy(self.board)
        while EMPTY in working_board:
            unassigned_spaces = np.where(working_board == EMPTY)
            c = unassigned_spaces[0][0], unassigned_spaces[1][0]
            territory = flood_fill(working_board, c)
            borders = set(itertools.chain(*(NEIGHBORS[t] for t in territory)))
            border_colors = set(working_board[b] for b in borders)
            X_border = BLACK in border_colors
            O_border = WHITE in border_colors
            if X_border and not O_border:
                territory_color = BLACK
            elif O_border and not X_border:
                territory_color = WHITE
            else:
                territory_color = UNKNOWN # dame, or seki
            working_board[working_board == FILL] = territory_color

        score_B_perspective = np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - self.komi
        if self.player1turn:
            score = score_B_perspective
        else:
            score = -score_B_perspective

        return score

set_board_size(19)