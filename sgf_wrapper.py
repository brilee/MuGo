'''
Code to extract a series of positions + their next moves from an SGF.

Most of the complexity here is dealing with two features of SGF:
- Stones can be added via "play move" or "add move", the latter being used
  to configure L+D puzzles, but also for initial handicap placement.
- Plays don't necessarily alternate colors; they can be repeated B or W moves
  This feature is used to handle free handicap placement.

Since our Go position data structure flips all colors based on whose turn it is
we have to look ahead at next move to correctly create a position.
'''
from collections import namedtuple
import numpy as np

import go
from go import Position, deduce_groups
from utils import parse_sgf_coords as pc
import sgf

def sgf_prop(value_list):
    'Converts raw sgf library output to sensible value'
    if value_list is None:
        return None
    if len(value_list) == 1:
        return value_list[0]
    else:
        return value_list

def sgf_prop_get(props, key, default):
    return sgf_prop(props.get(key, default))

def handle_node(pos, node):
    'A node can either add B+W stones, play as B, or play as W.'
    props = node.properties
    black_stones_added = [pc(coords) for coords in props.get('AB', [])]
    white_stones_added = [pc(coords) for coords in props.get('AW', [])]
    if black_stones_added or white_stones_added:
        return add_stones(pos, black_stones_added, white_stones_added)
    # If B/W props are not present, then there is no move. But if it is present and equal to the empty string, then the move was a pass.
    elif 'B' in props:
        black_move = pc(props.get('B', [''])[0])
        return play_move(pos, black_move, player1turn=True)
    elif 'W' in props:
        white_move = pc(props.get('W', [''])[0])
        return play_move(pos, white_move, player1turn=False)
    else:
        return pos

def add_stones(pos, black_stones_added, white_stones_added):
    black_color, white_color = (go.BLACK, go.WHITE) if pos.player1turn else (go.WHITE, go.BLACK)
    working_board = np.copy(pos.board)
    for b in black_stones_added:
        working_board[b] = black_color
    for w in white_stones_added:
        working_board[w] = white_color
    return pos._replace(board=working_board, groups=deduce_groups(working_board))

def play_move(pos, move, player1turn):
    if pos.player1turn != player1turn:
        pos = pos.flip_playerturn()
    return pos.play_move(move)

def get_next_move(node):
    if not node.next:
        return None
    props = node.next.properties
    if 'W' in props:
        return pc(props['W'][0])
    else:
        return pc(props['B'][0])

def maybe_correct_next(pos, next_node):
    if next_node is None:
        return pos
    if (('B' in next_node.properties and not pos.player1turn) or
        ('W' in next_node.properties and pos.player1turn)):
        pos = pos.flip_playerturn()
    return pos

class GameMetadata(namedtuple("GameMetadata", "result handicap board_size")):
    pass

class SgfWrapper(object):
    '''
    Wrapper for sgf files, exposing contents as go.Position instances
    with open(filename) as f:
        sgf = sgf_wrapper.SgfWrapper(f.read())
        for position, move, result in sgf.get_main_branch():
            print(position)
    '''

    def __init__(self, file_contents):
        self.collection = sgf.parse(file_contents)
        self.game = self.collection.children[0]
        props = self.game.root.properties
        assert int(sgf_prop(props.get('GM', ['1']))) == 1, "Not a Go SGF!"
        self.komi = float(sgf_prop(props.get('KM')))
        self.metadata = GameMetadata(
            result=sgf_prop(props.get('RE')),
            handicap=int(sgf_prop(props.get('HA', [0]))),
            board_size=int(sgf_prop(props.get('SZ'))))
        go.set_board_size(self.metadata.board_size)

    def get_main_branch(self):
        pos = Position.initial_state()
        pos = pos._replace(komi=self.komi)
        current_node = self.game.root
        while pos is not None and current_node is not None:
            pos = handle_node(pos, current_node)
            pos = maybe_correct_next(pos, current_node.next)
            next_move = get_next_move(current_node)
            yield PositionWithContext(pos, next_move, self.metadata)
            current_node = current_node.next

class PositionWithContext(namedtuple("SgfPosition", "position next_move metadata")):
    '''
    Wrapper around go.Position.
    Stores a position, the move that came next, and the eventual result.
    '''
    def is_usable(self):
        return all([
            self.position is not None,
            self.next_move is not None,
            self.metadata.result != "Void",
            self.metadata.board_size == 19,
            self.metadata.handicap <= 4,
        ])

    def __str__(self):
        return str(self.position) + '\nNext move: {} Result: {}'.format(self.next_move, self.result)
