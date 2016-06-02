import re
from collections import namedtuple
import itertools

N = 9
W = N + 1
# Represent a board as a string, with '.' empty, 'X' to play, 'x' other player.
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

def load_board(string):
    string = re.sub(r'[^xX\.#]+', '', string)
    assert len(string) == N ** 2, "Board to load didn't have right dimensions"
    return '\n'.join([' ' * N] + [string[k*N:(k+1)*N] for k in range(N)] + [' ' * W])

def parse_coords(s):
    if s == 'pass':
        return None
    s = s.upper()
    col = COLUMNS.index(s[0])
    rows_from_top = N - int(s[1:])
    return W + (W * rows_from_top) + col

def neighbors(c):
    return [c+1, c-1, c+W, c-W]

def place_stone(board, color, c):
    return board[:c] + color + board[c+1:]

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

class Group(namedtuple('Group', 'stones liberties')):
    '''
    stones: a set of Coordinates belonging to this group
    liberties: a set of Coordinates that are empty and adjacent to this group.
    '''
    pass

def find_liberties(board, stones):
    'Given a board and a set of stones, find liberties of those stones'
    potential_liberties = set(itertools.chain(*(neighbors(s) for s in stones)))
    return {c for c in potential_liberties if board[c] == '.'}

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

    return find_groups(board, 'X'), find_groups(board, 'x')

def update_groups(board, existing_X_groups, existing_x_groups, c):
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

    new_stones = set([c])
    new_liberties = set(n for n in neighbors(c) if board[n] == '.')
    for g in groups_to_merge:
        new_stones = new_stones | g.stones
        new_liberties = new_liberties | g.liberties
    new_liberties.remove(c)
    updated_X_groups.append(Group(stones=new_stones, liberties=new_liberties))

    updated_x_groups = []
    for g in existing_x_groups:
        if c in g.liberties:
            updated_x_groups.append(Group(stones=g.stones, liberties=g.liberties - {c}))
        else:
            updated_x_groups.append(g)

    return updated_X_groups, updated_x_groups
