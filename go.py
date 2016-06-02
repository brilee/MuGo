import re
from collections import namedtuple

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
    string = re.sub(r'[^xX\.]+', '', string)
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

def place_stone(board, move, c):
    return board[:c] + move + board[c+1:]

class Group(namedtuple('Group', 'stones liberties')):
    '''
    stones: a set of Coordinates belonging to this group
    liberties: a set of Coordinates that are empty and adjacent to this group.
    '''
    pass

def update_groups(board, existing_groups, c):
    '''
    When a move is played, update the list of groups and their liberties.
    This means possibly appending the new move to a group, creating a new 1-stone group, or merging existing groups.
    The existing groups and the new move are assumed to be of the same color.
    The board should represent the state after the move has been played at `c`.
    '''
    move_color = board[c]
    assert move_color in 'xX'
    move_neighbors = neighbors(c)
    adjacent_stones = [n for n in move_neighbors if board[n] == move_color]

    updated_groups = []
    groups_to_merge = []
    for g in existing_groups:
        if any(s in g.stones for s in adjacent_stones):
            groups_to_merge.append(g)
        else:
            updated_groups.append(g)

    new_stones = set([c])
    new_liberties = set(n for n in move_neighbors if board[n] == '.')
    for g in groups_to_merge:
        new_stones = new_stones | g.stones
        new_liberties = new_liberties | g.liberties
    new_liberties.remove(c)
    updated_groups.append(Group(stones=new_stones, liberties=new_liberties))

    return updated_groups

