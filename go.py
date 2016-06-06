import re
from collections import namedtuple
import itertools

N = 9
W = N + 1
# Represent a board as a string, with '.' empty, 'X' to play, 'O' other player.
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

COLUMNS = 'ABCDEFGHJKLMNOPQRST'

EMPTY_BOARD = '\n'.join(
    [' ' * N] + 
    ['.' * N for i in range(N)] + 
    [' ' * W])

SWAP_COLORS = str.maketrans('XO', 'OX')

def load_board(string):
    string = re.sub(r'[^OX\.#]+', '', string)
    assert len(string) == N ** 2, "Board to load didn't have right dimensions"
    return '\n'.join([' ' * N] + [string[k*N:(k+1)*N] for k in range(N)] + [' ' * W])

def parse_coords(s):
    if s == 'pass':
        return None
    s = s.upper()
    col = COLUMNS.index(s[0])
    rows_from_top = N - int(s[1:])
    return W + (W * rows_from_top) + col

ALL_COORDS = [parse_coords(col+row) for col in COLUMNS[:N] for row in map(str, range(1, N+1))]

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
    possessed_by = surrounding_colors.intersection('XO.')
    if len(possessed_by) == 1 and not '.' in possessed_by:
        return list(possessed_by)[0]
    else:
        return None

def is_likely_eye(board, c):
    '''
    Check if a coordinate c is a likely eye, and return its color if so.
    Does not guarantee that it's an eye. It only guarantees that a player
    wouldn't ever want to play there. For example: both are likely eyes
    XX.X.
    .XX..
    X....
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

    return find_groups(board, 'X'), find_groups(board, 'O')

def update_groups(board, existing_X_groups, existing_O_groups, c):
    '''
    When a move is played, update the list of groups and their liberties.
    This means possibly appending the new move to a group, creating a new 1-stone group, or merging existing groups.
    The new move should be of color X.
    The board should represent the state after the move has been played at `c`.
    '''
    assert board[c] == 'X'

    updated_X_groups, groups_to_merge = [], []
    for g in existing_X_groups:
        if c in g.liberties:
            groups_to_merge.append(g)
        else:
            updated_X_groups.append(g)

    new_stones = {c}
    new_liberties = set(n for n in neighbors(c) if board[n] == '.')
    for g in groups_to_merge:
        new_stones = new_stones | g.stones
        new_liberties = new_liberties | g.liberties
    new_liberties = new_liberties - {c}
    updated_X_groups.append(Group(stones=new_stones, liberties=new_liberties))

    updated_O_groups = []
    for g in existing_O_groups:
        if c in g.liberties:
            updated_O_groups.append(Group(stones=g.stones, liberties=g.liberties - {c}))
        else:
            updated_O_groups.append(g)

    return updated_X_groups, updated_O_groups

class Position(namedtuple('Position', 'board n komi caps groups ko last last2')):
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
        return self.play_move(parse_coords(input))

    def __str__(self):
        if self.ko is not None:
            board = place_stone(self.board, '*', self.ko)
        else:
            board = self.board
        captures = self.caps
        if not self.player1turn:
            board = board.translate(SWAP_COLORS)
            captures = captures[::-1]

        raw_board_contents = board.split('\n')[1:-1]
        row_labels = '12345678901234567890'[:N]
        annotated_board_contents = reversed([''.join(r) for r in zip(row_labels, reversed(raw_board_contents), row_labels)])
        header_footer_rows = [' ' + COLUMNS[:N] + ' ']
        annotated_board = '\n'.join(itertools.chain(header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures X: {} O: {}\n".format(self.n + 1, *captures)
        return annotated_board + details

    @property
    def player1turn(self):
        return self.n % 2 == 0

    @property
    def player1wins(self):
        return False

    @property
    def player2wins(self):
        return False

    def pass_move(self):
        return Position(
            board=self.board.translate(SWAP_COLORS),
            n=self.n+1,
            komi=-self.komi,
            caps=(self.caps[1], self.caps[0]),
            groups=(self.groups[1], self.groups[0]),
            ko=None,
            last=None,
            last2=None,
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
        if self.board[c] != '.':
            return None

        working_board = place_stone(self.board, 'X', c)
        new_X_groups, new_O_groups = update_groups(working_board, self.groups[0], self.groups[1], c)

        # process opponent's captures first, then your own suicides.
        # As stones are removed, liberty counts become inaccurate.
        O_captures = set()
        surviving_O_groups = []
        for group in new_O_groups:
            if not group.liberties:
                O_captures |= group.stones
                working_board = capture_stones(working_board, group.stones)
            else:
                surviving_O_groups.append(group)

        final_O_groups = surviving_O_groups
        final_X_groups = new_X_groups
        if O_captures:
            # recalculate liberties for groups adjacent to a captured O group
            coords_with_updates = find_neighbors('X', working_board, O_captures)
            final_X_groups = [g if not (g.stones & coords_with_updates)
                else Group(stones=g.stones, liberties=find_liberties(working_board, g.stones))
                for g in new_X_groups]
        else:
            # suicide can only happen if no O captures were made
            for group in new_X_groups:
                if not group.liberties:
                    # suicides are illegal!
                    return None

        if len(O_captures) == 1 and is_eyeish(self.board, c) == 'O':
            ko = list(O_captures)[0]
        else:
            ko = None

        return Position(
            board=working_board.translate(SWAP_COLORS),
            n=self.n + 1,
            komi=-self.komi,
            caps=(self.caps[1], self.caps[0] + len(O_captures)),
            groups=(final_O_groups, final_X_groups),
            ko=ko,
            last=c,
            last2=self.last,
        )
