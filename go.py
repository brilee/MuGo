import re

N = 9
W = N + 1
# Represent a board as a string, with '.' empty, 'X' to play, 'x' other player.
# Whitespace is used as a border (to avoid IndexError when computing neighbors)

# A Coordinate `c` is (Integer index | None) an index into the board.
# None can indicate lack of a ko restriction or a pass move.

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

