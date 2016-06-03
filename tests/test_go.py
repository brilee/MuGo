import unittest
import go

MANUAL_EMPTY_BOARD = '''         
.........
.........
.........
.........
.........
.........
.........
.........
.........
          '''

TEST_BOARD = go.load_board('''
.X.....OO
X........
.........
.........
.........
.........
.........
.........
.........
''')

pc = go.parse_coords
def pc_set(string):
    return set(map(pc, string.split()))

class TestGoBoard(unittest.TestCase):
    def test_load_board(self):
        self.assertEqual(len(go.EMPTY_BOARD), (go.W * (go.W + 1)))
        self.assertEqual(go.EMPTY_BOARD, MANUAL_EMPTY_BOARD)
        self.assertEqual(go.EMPTY_BOARD, go.load_board('. \n' * go.N ** 2))

    def test_parsing(self):
        self.assertEqual(pc('A' + str(go.N)), go.W)

    def test_neighbors(self):
        corner = pc('A1')
        neighbors = [go.EMPTY_BOARD[c] for c in go.neighbors(corner)]
        self.assertEqual(sum(1 for n in neighbors if n.isspace()), 2)

        side = pc('A2')
        side_neighbors = [go.EMPTY_BOARD[c] for c in go.neighbors(side)]
        self.assertEqual(sum(1 for n in side_neighbors if n.isspace()), 1)

class TestGroupHandling(unittest.TestCase):
    def test_flood_fill(self):
        expected_board = go.load_board('''
            .X.....##
            X........
            .........
            .........
            .........
            .........
            .........
            .........
            .........
        ''')
        actual_board, _ = go.flood_fill(TEST_BOARD, pc('H9'))
        self.assertEqual(expected_board, actual_board)

    def test_find_liberties(self):
        stones = pc_set('H9 J9')
        expected_liberties = pc_set('G9 H8 J8')
        actual_liberties = go.find_liberties(TEST_BOARD, stones)
        self.assertEqual(expected_liberties, actual_liberties)

    def test_deduce_groups(self):
        expected_groups = ([
            go.Group(
                stones=pc_set('B9'),
                liberties=pc_set('A9 C9 B8')
            ),
            go.Group(
                stones=pc_set('A8'),
                liberties=pc_set('A9 A7 B8')
            ),
            ], [
            go.Group(
                stones=pc_set('H9 J9'),
                liberties=pc_set('G9 H8 J8')
            )
            ]
        )
        actual_groups = go.deduce_groups(TEST_BOARD)
        self.assertEqual(expected_groups, actual_groups)

    def test_update_groups(self):
        existing_X_groups, existing_O_groups = go.deduce_groups(TEST_BOARD)
        updated_board = go.place_stone(TEST_BOARD, 'X', pc('A9'))
        updated_X_groups, updated_O_groups = go.update_groups(updated_board, existing_X_groups, existing_O_groups, pc('A9'))
        self.assertEqual(updated_X_groups, [go.Group(
            stones=pc_set('A8 A9 B9'),
            liberties=pc_set('A7 B8 C9')
        )])
        self.assertEqual(existing_O_groups, updated_O_groups)
