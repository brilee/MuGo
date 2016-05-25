from collections import Counter
from strategies import AVAILABLE_STRATEGIES
from tictactoe import EMPTY_BOARD

def choose_strategy():
    choice = None
    while choice is None:
        user_input = input("Choose from available strategies: " + ', '.join(AVAILABLE_STRATEGIES) + '\n')
        if user_input in AVAILABLE_STRATEGIES:
            choice = AVAILABLE_STRATEGIES[user_input]
        else:
            print("Not a valid choice!")
    return choice

def play_game(strategy1, strategy2, board, verbose=False):
    while not (board.player1wins or board.player2wins):
        if verbose: print(board)
        if not board.possible_moves():
            break
        if board.player1turn:
            move = strategy1.suggest_move(board)
        else:
            move = strategy2.suggest_move(board)
        if verbose: print("Player {} played {}".format('1' if board.player1turn else '2', move))
        if verbose: print()
        board = board.update(move)
    assert not (board.player1wins and board.player2wins), "Uhoh, both players won somehow %s" % board

    if board.player1wins:
        return True
    elif board.player2wins:
        return False
    else:
        return None # A draw.

def run_many(strategy1, strategy2, board, num_trials=1000):
    results = [play_game(strategy1, strategy2, board, verbose=False)
        for i in range(num_trials)]
    return Counter(results)

if __name__ == '__main__':
    strategy1 = choose_strategy()
    strategy2 = choose_strategy()
    result = play_game(strategy1, strategy2, EMPTY_BOARD, verbose=True)
    if result is True:
        print("Player 1 wins!")
    elif result is False:
        print("Player 2 wins!")
    else:
        print("Draw!")



