import argparse
import sys
import gtp

from strategies import RandomPlayer

def run_gtp(strategy):
    gtp_engine = gtp.Engine(strategy())
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not gtp_engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = gtp_engine.send(cmd)
            sys.stdout.write(engine_reply)
            sys.stdout.flush()

strategies = {
    'random': RandomPlayer,
}

parser = argparse.ArgumentParser()
parser.add_argument('strategy', choices=strategies.keys())
if __name__ == '__main__':
    args = parser.parse_args()
    strategy = strategies[args.strategy]
    run_gtp(strategy)
