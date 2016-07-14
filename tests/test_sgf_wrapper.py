import go
import sgf_wrapper
from utils import parse_kgs_coords as pc
from test_utils import GoPositionTestCase, load_board

JAPANESE_HANDICAP_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Japanese]SZ[9]HA[2]KM[5.50]PW[test_white]PB[test_black]AB[gc][cg];W[ee];B[dg])"

CHINESE_HANDICAP_SGF = "(;GM[1]FF[4]CA[UTF-8]AP[CGoban:3]ST[2]RU[Chinese]SZ[9]HA[2]KM[5.50]PW[test_white]PB[test_black]RE[B+39.50];B[gc];B[cg];W[ee];B[gg];W[eg];B[ge];W[ce];B[ec];W[cc];B[dd];W[de];B[cd];W[bd];B[bc];W[bb];B[be];W[ac];B[bf];W[dh];B[ch];W[ci];B[bi];W[di];B[ah];W[gh];B[hh];W[fh];B[hg];W[gi];B[fg];W[dg];B[ei];W[cf];B[ef];W[ff];B[fe];W[bg];B[bh];W[af];B[ag];W[ae];B[ad];W[ae];B[ed];W[db];B[df];W[eb];B[fb];W[ea];B[fa])"


class TestSgfWrapper(GoPositionTestCase):
    def test_sgf_props(self):
        sgf = sgf_wrapper.SgfWrapper(CHINESE_HANDICAP_SGF)
        self.assertEqual(sgf.result, 'B+39.50')
        self.assertEqual(sgf.board_size, 9)
        self.assertEqual(sgf.komi, 5.5)

    def test_japanese_handicap_handling(self):
        final_board = load_board('''
            .........
            .........
            ......X..
            .........
            ....O....
            .........
            ..XX.....
            .........
            .........
        ''', player1turn=False)
        final_position = go.Position(
            board=final_board,
            n=2,
            komi=5.5,
            caps=(0, 0),
            groups=go.deduce_groups(final_board),
            ko=None,
            last=pc('E5'),
            last2=pc('G3'),
            player1turn=False,
        )
        sgf = sgf_wrapper.SgfWrapper(JAPANESE_HANDICAP_SGF)
        positions_w_context = list(sgf.get_main_branch())
        self.assertEqualPositions(final_position, positions_w_context[-1].position)

    def test_chinese_handicap_handling(self):
        final_board = load_board('''
            ....OX...
            .O.OOX...
            O.O.X.X..
            .OXXX....
            OX...XX..
            .X.XXO...
            X.XOOXXX.
            XXXO.OOX.
            .XOOX.O..
        ''', player1turn=False)
        final_position = go.Position(
            board=final_board,
            n=50,
            komi=5.5,
            caps=(2, 7),
            groups=go.deduce_groups(final_board),
            ko=None,
            last=pc('F9'),
            last2=pc('E9'),
            player1turn=False,
        )
        sgf = sgf_wrapper.SgfWrapper(CHINESE_HANDICAP_SGF)
        positions_w_context = list(sgf.get_main_branch())
        self.assertEqualPositions(final_position, positions_w_context[-1].position)
        self.assertFalse(positions_w_context[-1].is_usable())
        self.assertTrue(positions_w_context[-2].is_usable())

