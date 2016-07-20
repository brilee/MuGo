import numpy as np
import unittest
import go
from utils import parse_kgs_coords as pc, parse_sgf_coords
from test_utils import GoPositionTestCase, load_board

go.set_board_size(9)

EMPTY_ROW = '.' * go.N + '\n'
TEST_BOARD = load_board('''
.X.....OO
X........
''' + EMPTY_ROW * 7, player1turn=True)

def pc_set(string):
    return set(map(pc, string.split()))

class TestGoBoard(GoPositionTestCase):
    def test_load_board(self):
        self.assertEqualNPArray(go.EMPTY_BOARD, np.zeros([go.N, go.N]))
        self.assertEqualNPArray(go.EMPTY_BOARD, load_board('. \n' * go.N ** 2, player1turn=True))

    def test_parsing(self):
        self.assertEqual(pc('A9'), (0, 0))
        self.assertEqual(parse_sgf_coords('aa'), (0, 0))
        self.assertEqual(pc('A3'), (6, 0))
        self.assertEqual(parse_sgf_coords('ac'), (2, 0))
        self.assertEqual(pc('D4'), parse_sgf_coords('df'))

    def test_neighbors(self):
        corner = pc('A1')
        neighbors = [go.EMPTY_BOARD[c] for c in go.NEIGHBORS[corner]]
        self.assertEqual(len(neighbors), 2)

        side = pc('A2')
        side_neighbors = [go.EMPTY_BOARD[c] for c in go.NEIGHBORS[side]]
        self.assertEqual(len(side_neighbors), 3)

class TestGroupHandling(GoPositionTestCase):
    def test_flood_fill(self):
        expected_board = load_board('''
            .X.....##
            X........
        ''' + EMPTY_ROW * 7, player1turn=True)
        test_board_copy = np.copy(TEST_BOARD)
        stones = go.flood_fill(test_board_copy, pc('H9'))
        self.assertEqualNPArray(expected_board, test_board_copy)
        self.assertEqual(pc_set('H9 J9'), stones)

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
        updated_board = go.place_stone(TEST_BOARD, go.BLACK, pc('A9'))
        updated_X_groups, updated_O_groups = go.update_groups(updated_board, existing_X_groups, existing_O_groups, pc('A9'))
        self.assertEqual(updated_X_groups, [go.Group(
            stones=pc_set('A8 A9 B9'),
            liberties=pc_set('A7 B8 C9')
        )])
        self.assertEqual(existing_O_groups, updated_O_groups)

class TestEyeHandling(GoPositionTestCase):
    def test_is_koish(self):
        self.assertEqual(go.is_koish(TEST_BOARD, pc('A9')), go.BLACK)
        self.assertEqual(go.is_koish(TEST_BOARD, pc('B8')), None)
        self.assertEqual(go.is_koish(TEST_BOARD, pc('B9')), None)
        self.assertEqual(go.is_koish(TEST_BOARD, pc('E5')), None)

    def test_is_eye(self):
        board = load_board('''
            .XX...XXX
            X.X...X.X
            XX.....X.
            ........X
            XXXX.....
            OOOX....O
            X.OXX.OO.
            .XO.X.O.O
            XXO.X.OO.
        ''', player1turn=True)
        B_eyes = pc_set('A9 B8 H8')
        W_eyes = pc_set('H2 J1')
        not_eyes = pc_set('A2 B3 J7 E5 J3')
        for be in B_eyes:
            self.assertEqual(go.is_eye(board, be), go.BLACK)
        for we in W_eyes:
            self.assertEqual(go.is_eye(board, we), go.WHITE)
        for ne in not_eyes:
            self.assertEqual(go.is_eye(board, ne), None)

class TestPosition(GoPositionTestCase):
    def test_legal_moves(self):
        board = load_board('''
            .XXXXXXXO
            XX.OOOOO.
            OOOOOOOOO
            XXXXXXXX.
            OOOOOOOOO
            XXXXXXXXX
            XXXXXXXXX
            XXXXXXXXX
            XXXXXXXX.
        ''', player1turn=True)
        position = go.Position(
            board=board,
            n=0,
            komi=6.5,
            caps=(0, 0),
            groups=go.deduce_groups(board),
            ko=pc('J8'),
            last=None,
            last2=None,
            player1turn=True,
        )
        empty_spots = pc_set('A9 C8 J8 J6 J1')
        B_legal_moves = pc_set('A9 C8 J6')
        for move in empty_spots:
            result = position.play_move(move)
            self.assertEqual(result is not None, move in B_legal_moves)

        pass_position = position.pass_move()
        W_legal_moves = pc_set('C8 J8 J6 J1')
        for move in empty_spots:
            result = pass_position.play_move(move)
            self.assertEqual(result is not None, move in W_legal_moves)

    def test_move(self):
        start_position = go.Position(
            board=TEST_BOARD,
            n=0,
            komi=6.5,
            caps=(1, 2),
            groups=go.deduce_groups(TEST_BOARD),
            ko=None,
            last=None,
            last2=None,
            player1turn=True,
        )
        expected_board = load_board('''
            .XX....OO
            X........
        ''' + EMPTY_ROW * 7, player1turn=False)
        expected_position = go.Position(
            board=expected_board,
            n=1,
            komi=-6.5,
            caps=(2, 1),
            groups=go.deduce_groups(expected_board),
            ko=None,
            last=pc('C9'),
            last2=None,
            player1turn=False,
        )
        actual_position = start_position.play_move(pc('C9'))
        self.assertEqualPositions(actual_position, expected_position)

        expected_board2 = load_board('''
            .XX....OO
            X.......O
        ''' + EMPTY_ROW * 7, player1turn=True)
        expected_position2 = go.Position(
            board=expected_board2,
            n=2,
            komi=6.5,
            caps=(1, 2),
            groups=go.deduce_groups(expected_board2),
            ko=None,
            last=pc('J8'),
            last2=pc('C9'),
            player1turn=True,
        )
        actual_position2 = actual_position.play_move(pc('J8'))
        self.assertEqualPositions(actual_position2, expected_position2)

    def test_move_with_capture(self):
        start_board = load_board(EMPTY_ROW * 5 + '''
            XXXX.....
            XOOX.....
            O.OX.....
            OOXX.....
        ''', player1turn=True)
        start_position = go.Position(
            board=start_board,
            n=0,
            komi=6.5,
            caps=(1, 2),
            groups=go.deduce_groups(start_board),
            ko=None,
            last=None,
            last2=None,
            player1turn=True,
        )
        expected_board = load_board(EMPTY_ROW * 5 + '''
            XXXX.....
            X..X.....
            .X.X.....
            ..XX.....
        ''', player1turn=False)
        expected_position = go.Position(
            board=expected_board,
            n=1,
            komi=-6.5,
            caps=(2, 7),
            groups=go.deduce_groups(expected_board),
            ko=None,
            last=pc('B2'),
            last2=None,
            player1turn=False,
        )
        actual_position = start_position.play_move(pc('B2'))
        self.assertEqualPositions(actual_position, expected_position)

    def test_ko_move(self):
        start_board = load_board('''
            .OX......
            OX.......
        ''' + EMPTY_ROW * 7, player1turn=True)
        start_position = go.Position(
            board=start_board,
            n=0,
            komi=6.5,
            caps=(1, 2),
            groups=go.deduce_groups(start_board),
            ko=None,
            last=None,
            last2=None,
            player1turn=True,
        )
        expected_board = load_board('''
            X.X......
            OX.......
        ''' + EMPTY_ROW * 7, player1turn=False)
        expected_position = go.Position(
            board=expected_board,
            n=1,
            komi=-6.5,
            caps=(2, 2),
            groups=go.deduce_groups(expected_board),
            ko=pc('B9'),
            last=pc('A9'),
            last2=None,
            player1turn=False,
        )
        actual_position = start_position.play_move(pc('A9'))

        self.assertEqualPositions(actual_position, expected_position)

        # Check that retaking ko is illegal until two intervening moves
        ko_immediate_retake = actual_position.play_move(pc('B9'))
        self.assertEqual(ko_immediate_retake, None)
        pass_twice = actual_position.pass_move().pass_move()
        ko_delayed_retake = pass_twice.play_move(pc('B9'))
        expected_position = go.Position(
            board=start_board,
            n=4,
            komi=6.5,
            caps=(2, 3),
            groups=go.deduce_groups(start_board),
            ko=pc('A9'),
            last=pc('B9'),
            last2=None,
            player1turn=True,
        )
        self.assertEqualPositions(ko_delayed_retake, expected_position)

class TestScoring(unittest.TestCase):
    def test_scoring(self):
            board = load_board('''
                .XX......
                OOXX.....
                OOOX...X.
                OXX......
                OOXXXXXX.
                OOOXOXOXX
                .O.OOXOOX
                .O.O.OOXX
                ......OOO
            ''', player1turn=True)
            position = go.Position(
                board=board,
                n=54,
                komi=6.5,
                caps=(2, 5),
                groups=go.deduce_groups(board),
                ko=None,
                last=None,
                last2=None,
                player1turn=True,
            )
            expected_score = 1.5
            self.assertEqual(position.score(), expected_score)

            board = load_board('''
                XXX......
                OOXX.....
                OOOX...X.
                OXX......
                OOXXXXXX.
                OOOXOXOXX
                .O.OOXOOX
                .O.O.OOXX
                ......OOO
            ''', player1turn=False)
            position = go.Position(
                board=board,
                n=55,
                komi=-6.5,
                caps=(5, 2),
                groups=go.deduce_groups(board),
                ko=None,
                last=None,
                last2=None,
                player1turn=False,
            )
            expected_score = 2.5
            self.assertEqual(position.score(), expected_score)
