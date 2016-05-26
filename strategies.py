import random

def DEFAULT_VALUE_FUNC(board):
    if board.player1wins: return 1
    if board.player2wins: return -1
    return 0


class BaseStrategy(object):
    '''
    Takes in a board implementing the following interface
    class Board:
        possible_moves(self) => list(Move)
        update(self, Move) throws ValueError => Board # new board instance
        @property player1turn(self) => bool
        @property player1wins(self) => bool
        @property player2wins(self) => bool
    '''
    def __init__(self, value_f=DEFAULT_VALUE_FUNC):
        'Override the default value function to get better evaluations'
        self.value = value_f

    def suggest_move(self, board):
        if not board.possible_moves():
            raise TypeError("No more valid moves left!")
        return self._suggest_move(board)

    def _suggest_move(self, board):
        '''
        Given a board position, suggest a move.
        '''
        raise NotImplementedError

class InteractivePlayer(BaseStrategy):
    def _suggest_move(self, board):
        while True:
            player_input = input("It's your turn! Play a move.\n")
            try:
                board.update(player_input)
                return player_input
            except ValueError:
                print("Invalid move")

class RandomPlayer(BaseStrategy):
    def _suggest_move(self, board):
        return random.choice(board.possible_moves())

class OneMoveLookahead(BaseStrategy):
    def _suggest_move(self, board):
        moves = board.possible_moves()
        strategy = max if board.player1turn else min
        moves_with_valuation = [
            (self.value(board.update(move)), move)
            for move in moves
        ]
        return strategy(moves_with_valuation)[1]

class MinMaxPlayer(BaseStrategy):
    def _suggest_move(self, board, MAX_DEPTH=4):
        moves = board.possible_moves()
        random.shuffle(moves)
        strategy = max if board.player1turn else min
        moves_with_valuation = [
            (self.minimax(board.update(move), MAX_DEPTH), move)
            for move in moves
        ]
        return strategy(moves_with_valuation)[1]

    def minimax(self, board, depth):
        if depth == 0:
            return self.value(board)
        if board.player1wins: return 1
        if board.player2wins: return -1

        possible_moves = board.possible_moves()
        if not possible_moves:
            return self.value(board)

        strategy = max if board.player1turn else min
        moves_with_valuation = [
            (self.minimax(board.update(move), depth-1), move)
            for move in board.possible_moves()
        ]
        return strategy(moves_with_valuation)[0]

class MCTS(BaseStrategy):
    pass

AVAILABLE_STRATEGIES = {
    'interactive': InteractivePlayer(),
    'random': RandomPlayer(),
    'onemove-lookahead': OneMoveLookahead(),
    'minimax': MinMaxPlayer(),
}