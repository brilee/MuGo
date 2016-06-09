import re
from collections import namedtuple
import itertools

# Represent a board as a string, with '.' empty, 'B' is black, 'W' is white.
# AP and OP refer to "active player" and "other player".
# Whitespace is used as a border (to avoid IndexError when computing neighbors)

# A Coordinate `c` is an int: an index into the board.
# A Move is a (Coordinate c | None).

# Human-readable move notation: columns go from A to T left to right
# rows go from 1 to 19 from bottom to top.
#  ABCD
#  ____\n
# 4....\n
# 3....\n
# 2....\n
# 1....

N = 9
W = N + 1
ALL_COORDS = [] # initialized later on
EMPTY_BOARD = '' # initialized later on
COLUMNS = 'ABCDEFGHJKLMNOPQRST'
SGF_COLUMNS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
SWAP_COLORS = str.maketrans('BW', 'WB')

def set_board_size(n):
    'Hopefully nobody tries to run both 9x9 and 19x19 game instances at once.'
    global N, W, ALL_COORDS, EMPTY_BOARD
    N = n
    W = n + 1
    ALL_COORDS = [parse_coords(col+row) for col in COLUMNS[:N] for row in map(str, range(1, N+1))]
    EMPTY_BOARD = '\n'.join(
        [' ' * N] +
        ['.' * N for i in range(N)] +
        [' ' * W])

def load_board(string):
    string = re.sub(r'[^BW\.#]+', '', string)
    assert len(string) == N ** 2, "Board to load didn't have right dimensions"
    return '\n'.join([' ' * N] + [string[k*N:(k+1)*N] for k in range(N)] + [' ' * W])

def parse_sgf_coords(s):
    'Interprets coords in the format "hc", with "aa" being top left quadrant'
    if not s:
        return None
    x_coord, y_coord = [SGF_COLUMNS.index(coord) for coord in s]
    return W + W * y_coord + x_coord

def parse_coords(s):
    'Interprets coords in the format "H3", with A1 being lower left quadrant.'
    if s == 'pass':
        return None
    s = s.upper()
    col = COLUMNS.index(s[0])
    rows_from_top = N - int(s[1:])
    return W + (W * rows_from_top) + col

def neighbors(c):
    return [c+1, c-1, c+W, c-W]

def diagonals(c):
    return [c-W-1, c-W+1, c+W-1, c+W+1]

def place_stone(board, color, c):
    return board[:c] + color + board[c+1:]

def capture_stones(board, stones):
    b = bytearray(board, encoding='ascii')
    for s in stones:
        b[s] = ord('.')
    return str(b, encoding='ascii')

def flood_fill(board, c):
    'From a starting coordinate c, flood-fill the board with a #'
    b = bytearray(board, encoding='ascii')
    color = b[c]
    entire_group = [c]
    frontier = [c]
    while frontier:
        current = frontier.pop()
        b[current] = ord('#')
        for n in neighbors(current):
            if b[n] == color:
                frontier.append(n)
                entire_group.append(n)
    return str(b, encoding='ascii'), set(entire_group)

def find_neighbors(color, board, stones):
    'Find all neighbors of a set of stones of a given color'
    potential_neighbors = set(itertools.chain(*(neighbors(s) for s in stones)))
    return {c for c in potential_neighbors if board[c] == color}

def find_liberties(board, stones):
    'Given a board and a set of stones, find liberties of those stones'
    return find_neighbors('.', board, stones)

def is_eyeish(board, c):
    'Check if c is a false/likely true eye, and return its color if so'
    if board[c] != '.': return None
    surrounding_colors = {board[n] for n in neighbors(c)}
    possessed_by = surrounding_colors.intersection('BW.')
    if len(possessed_by) == 1 and not '.' in possessed_by:
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
    opposite_color = color.translate(SWAP_COLORS)

    diagonal_faults = 0
    diagonal_owners = [board[d] for d in diagonals(c)]
    if any(d.isspace() for d in diagonal_owners):
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
        groups = []
        while color in board:
            c = board.index(color)
            board, stones = flood_fill(board, c)
            liberties = find_liberties(board, stones)
            groups.append(Group(stones=stones, liberties=liberties))
        return groups

    return find_groups(board, 'B'), find_groups(board, 'W')

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
    new_liberties = set(n for n in neighbors(c) if board[n] == '.')
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
        return Position(EMPTY_BOARD, n=0, komi=7.5, caps=(0, 0), groups=(set(), set()), ko=None, last=None, last2=None)

    def possible_moves(self):
        return [c for c in ALL_COORDS if self.board[c] == '.' and not is_likely_eye(self.board, c)]

    def update(self, input):
        return self.play_move('B' if self.player1turn else 'W', parse_coords(input))

    def __str__(self):
        if self.ko is not None:
            board = place_stone(self.board, '*', self.ko)
        else:
            board = self.board
        captures = self.caps

        raw_board_contents = board.split('\n')[1:-1]
        row_labels = '12345678901234567890'[:N]
        annotated_board_contents = reversed([''.join(r) for r in zip(row_labels, reversed(raw_board_contents), row_labels)])
        header_footer_rows = [' ' + COLUMNS[:N] + ' ']
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures X: {} O: {}\n".format(self.n + 1, *captures)
        return annotated_board + details

    @property
    def player1wins(self):
        return False

    @property
    def player2wins(self):
        return False

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

    def play_move(self, color, c):
        # Obeys CGOS Rules of Play. In short:
        # No suicides
        # Chinese/area scoring
        # Positional superko (this is very crudely approximate at the moment.)
        if c is None:
            return self.pass_move()
        if c == self.ko:
            return None
        if self.board[c] != '.':
            return None

        AP_color = 'B' if self.player1turn else 'W'
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
                working_board = capture_stones(working_board, group.stones)
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

        if len(OP_captures) == 1 and is_eyeish(self.board, c) == 'W':
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
        working_board = self.board
        while '.' in working_board:
            c = working_board.find('.')
            working_board, territory = flood_fill(working_board, c)
            borders = set(itertools.chain(*(neighbors(t) for t in territory)))
            border_colors = set(working_board[b] for b in borders)
            X_border = 'B' in border_colors
            O_border = 'W' in border_colors
            if X_border and not O_border:
                territory_color = 'B'
            elif O_border and not X_border:
                territory_color = 'W'
            else:
                territory_color = '?' # dame, or seki
            working_board = working_board.replace('#', territory_color)

        return working_board.count('B') - working_board.count('W') - self.komi

set_board_size(9)