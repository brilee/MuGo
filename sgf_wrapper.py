import go
from go import Position, place_stone, deduce_groups, parse_sgf_coords as pc
import sgf

def interpret_value(value):
    'Attempt to interpret value as an integer, a float, or a string.'
    try:
        f = float(value)
        i = int(f)
        if f == i:
            return i
        else:
            return f
    except ValueError:
        return value

def sgf_prop(value_list):
    'Converts raw sgf library output to sensible value'
    if value_list is None:
        return None
    if len(value_list) == 1:
        return interpret_value(value_list[0])
    else:
        return map(interpret_value, value_list)

# SGFs have a notion of "add stones" and "play stones".
# Add stones can have arbitrary numbers of either color stone, and is used
# to set up L+D puzzles or handicap stones.
# SGF spec says that you shouldn't resolve captures in an add stone node.
def handle_add_stones(pos, node):
    black_stones_added = node.properties.get('AB', [])
    white_stones_added = node.properties.get('AW', [])
    working_board = pos.board
    for b in black_stones_added:
        working_board = place_stone(working_board, 'B', pc(b))
    for w in white_stones_added:
        working_board = place_stone(working_board, 'W', pc(w))
    return pos._replace(board=working_board, groups=deduce_groups(working_board))

# Play stones should have just 1 stone. Play is not necessarily alternating;
# sometimes B plays repeatedly at the start in free handicap placement.
# Must look at next node to figure out who was "supposed" to have played.
def handle_play_stones(pos, node):
    props = node.properties
    if 'W' in props:
        pos = pos.play_move('W', pc(props['W'][0]))
    elif 'B' in props:
        pos = pos.play_move('B', pc(props['B'][0]))
    if node.next:
        props = node.next.properties
        if pos.player1turn and 'W' in props:
            pos = pos._replace(player1turn=False)
        elif not pos.player1turn and 'B' in props:
            pos = pos._replace(player1turn=True)
    return pos

class SgfWrapper(object):
    '''
    Wrapper for sgf files, exposing contents as go.Position instances
    with open(filename) as f:
        sgf = sgf_wrapper.SgfWrapper(f.read())
        for position in sgf.get_main_branch():
            print(position)
    '''

    def __init__(self, file_contents):
        self.collection = sgf.parse(file_contents)
        self.game = self.collection.children[0]
        props = self.game.root.properties
        assert sgf_prop(props.get('GM', [1])) == 1, "Not a Go SGF!"
        self.result = sgf_prop(props.get('RE'))
        self.komi = sgf_prop(props.get('KM'))
        self.board_size = sgf_prop(props.get('SZ'))
        go.set_board_size(self.board_size)

    def get_main_branch(self):
        pos = Position.initial_state()
        pos = pos._replace(komi=self.komi)
        current_node = self.game.root
        while pos is not None and current_node is not None:
            pos = handle_add_stones(pos, current_node)
            pos = handle_play_stones(pos, current_node)
            current_node = current_node.next
            yield pos

