from collections import namedtuple
import itertools

import numpy as np
# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# AP and OP refer to "active player" and "other player".
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)
# Other special values:

# A Coordinate is a tuple index into the board. 
# A Move is a (Coordinate c | None).

# When representing the numpy array as a board, (0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.

N = None
ALL_COORDS = [] # initialized later on
EMPTY_BOARD = None # initialized later on
NEIGHBORS = {} # initialized later on
DIAGONALS = {} # initialized later on
COLUMNS = 'ABCDEFGHJKLMNOPQRST'
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

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

def parse_sgf_coords(s):
    'Interprets coords in the format "aj", with "aj" being top right quadrant'
    if not s:
        return None
    return SGF_COLUMNS.index(s[1]), SGF_COLUMNS.index(s[0])

def parse_kgs_coords(s):
    'Interprets coords in the format "H3", with A1 being lower left quadrant.'
    if s == 'pass':
        return None
    s = s.upper()
    col = COLUMNS.index(s[0])
    row_from_bottom = int(s[1:]) - 1
    return N - row_from_bottom - 1, col

def place_stone(board, color, c):
    new_board = np.copy(board)
    new_board[c] = color
    return new_board

def capture_stones(board, stones):
    for s in stones:
        board[s] = EMPTY

def flood_fill(b, c):
    'From a starting coordinate c, flood-fill (mutate) the board with FILL'
    color = b[c]
    entire_group = [c]
    frontier = [c]
    while frontier:
        current = frontier.pop()
        b[current] = FILL
        for n in NEIGHBORS[current]:
            if b[n] == color:
                frontier.append(n)
                entire_group.append(n)
    return entire_group

def find_neighbors(color, board, stones):
    'Find all neighbors of a set of stones of a given color'
    potential_neighbors = set(itertools.chain(*(NEIGHBORS[s] for s in stones)))
    equal_color = board == color
    return {c for c in potential_neighbors if equal_color[c]}

def find_liberties(board, stones):
    'Given a board and a set of stones, find liberties of those stones'
    return find_neighbors(EMPTY, board, stones)

def is_eyeish(board, c):
    'Check if c is a false/likely true eye, and return its color if so'
    if board[c] != EMPTY: return None
    surrounding_colors = {board[n] for n in NEIGHBORS[c]}
    possessed_by = surrounding_colors.intersection({BLACK, WHITE, EMPTY})
    if len(possessed_by) == 1 and not EMPTY in possessed_by:
        return list(possessed_by)[0]
    else:
        return None

def is_likely_eye(board, c):
    '''
    Check if a coordinate c is a likely eye, and return its color if so.
    Does not guarantee that it's an eye. It only guarantees that a player
    wouldn't ever want to play there. For example: both are likely eyes
    BB.B.
    .BB..
    B....
    .....
    '''
    color = is_eyeish(board, c)
    if color is None: return None
    opposite_color = -1 * color

    diagonal_faults = 0
    diagonal_owners = [board[d] for d in DIAGONALS[c]]
    if len(diagonal_owners) < 4:
        diagonal_faults += 1
    diagonal_faults += len([d for d in diagonal_owners if d == opposite_color])

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

def update_groups(board, existing_AP_groups, existing_OP_groups, c):
    '''
    When a move is played, update the list of groups and their liberties.
    This means possibly appending the new move to a group, creating a new 1-stone group, or merging existing groups.
    The board should represent the state after the move has been played at `c`.
    '''
    updated_AP_groups, groups_to_merge = [], []
    for g in existing_AP_groups:
        if c in g.liberties:
            groups_to_merge.append(g)
        else:
            updated_AP_groups.append(g)

    new_stones = {c}
    new_liberties = set(n for n in NEIGHBORS[c] if board[n] == EMPTY)
    for g in groups_to_merge:
        new_stones = new_stones | g.stones
        new_liberties = new_liberties | g.liberties
    new_liberties = new_liberties - {c}
    updated_AP_groups.append(Group(stones=new_stones, liberties=new_liberties))

    updated_OP_groups = []
    for g in existing_OP_groups:
        if c in g.liberties:
            updated_OP_groups.append(Group(stones=g.stones, liberties=g.liberties - {c}))
        else:
            updated_OP_groups.append(g)

    return updated_AP_groups, updated_OP_groups

class Position(namedtuple('Position', 'board n komi caps groups ko last last2 player1turn')):
    '''
    board: a string representation of the board
    n: an int representing moves played so far
    komi: a float, representing points given to the second player.
    caps: a (int, int) tuple of captures; caps[0] is the person to play.
    groups: a (list(Group), list(Group)) tuple of lists of Groups; groups[0] represents the groups of the person to play.
    ko: a Move
    last, last2: a Move
    '''
    @staticmethod
    def initial_state():
        return Position(EMPTY_BOARD, n=0, komi=7.5, caps=(0, 0), groups=(set(), set()), ko=None, last=None, last2=None, player1turn=True)

    def possible_moves(self):
        return [c for c in ALL_COORDS if self.board[c] == EMPTY and not is_likely_eye(self.board, c)]

    def __str__(self):
        pretty_print_map = {
            WHITE: 'W',
            EMPTY: '.',
            BLACK: 'B',
            FILL: '#',
            KO: '*',
        }
        if self.ko is not None:
            board = place_stone(self.board, 3, self.ko)
        else:
            board = self.board
        raw_board_contents = []
        for i in range(N):
            row = []
            for j in range(N):
                row.append(pretty_print_map[board[i,j]])
            raw_board_contents.append(''.join(row))
        captures = self.caps

        row_labels = '12345678901234567890'[:N]
        annotated_board_contents = reversed([''.join(r) for r in zip(row_labels, reversed(raw_board_contents), row_labels)])
        header_footer_rows = [' ' + COLUMNS[:N] + ' ']
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures B: {} W: {}\n".format(self.n + 1, *captures)
        return annotated_board + details

    def pass_move(self):
        return Position(
            board=self.board,
            n=self.n+1,
            komi=self.komi,
            caps=self.caps,
            groups=self.groups,
            ko=None,
            last=None,
            last2=self.last,
            player1turn=not self.player1turn,
        )

    def play_move(self, c):
        # Obeys CGOS Rules of Play. In short:
        # No suicides
        # Chinese/area scoring
        # Positional superko (this is very crudely approximate at the moment.)
        if c is None:
            return self.pass_move()
        if c == self.ko:
            return None
        if self.board[c] != 0:
            return None

        AP_color = BLACK if self.player1turn else WHITE
        AP_groups, OP_groups = self.groups if self.player1turn else self.groups[::-1]

        working_board = place_stone(self.board, AP_color, c)
        new_AP_groups, new_OP_groups = update_groups(working_board, AP_groups, OP_groups, c)

        # process OP captures first, then your own suicides.
        # As stones are removed, liberty counts become inaccurate.
        OP_captures = set()
        surviving_OP_groups = []
        for group in new_OP_groups:
            if not group.liberties:
                OP_captures |= group.stones
                capture_stones(working_board, group.stones)
            else:
                surviving_OP_groups.append(group)

        final_OP_groups = surviving_OP_groups
        final_AP_groups = new_AP_groups
        if OP_captures:
            # recalculate liberties for groups adjacent to a captured OP group
            coords_with_updates = find_neighbors(AP_color, working_board, OP_captures)
            final_AP_groups = [g if not (g.stones & coords_with_updates)
                else Group(stones=g.stones, liberties=find_liberties(working_board, g.stones))
                for g in new_AP_groups]
        else:
            # suicide can only happen if no O captures were made
            for group in new_AP_groups:
                if not group.liberties:
                    # suicides are illegal!
                    return None

        if len(OP_captures) == 1 and is_eyeish(self.board, c) == WHITE:
            ko = list(OP_captures)[0]
        else:
            ko = None

        if self.player1turn:
            groups = (final_AP_groups, final_OP_groups)
            caps = (self.caps[0] + len(OP_captures), self.caps[1])
        else:
            groups = (final_OP_groups, final_AP_groups)
            caps = (self.caps[0], self.caps[1] + len(OP_captures))

        return Position(
            board=working_board,
            n=self.n + 1,
            komi=self.komi,
            caps=caps,
            groups=groups,
            ko=ko,
            last=c,
            last2=self.last,
            player1turn=not self.player1turn,
        )

    def score(self):
        'Returns score from B perspective'
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

        return np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - self.komi

set_board_size(9)