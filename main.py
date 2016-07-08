import argparse
import argh
import sys
import gtp as gtp_lib

from features import DEFAULT_FEATURES
from strategies import RandomPlayer, PolicyNetworkBestMovePlayer
from policy import PolicyNetwork
from load_data_sets import load_data_sets

def gtp(strategy, read_file=None):
    if strategy == 'random':
        instance = RandomPlayer()
    elif strategy == 'policy':
        policy_network = PolicyNetwork(DEFAULT_FEATURES.planes)
        policy_network.initialize_variables(read_file)
        instance = PolicyNetworkBestMovePlayer(policy_network)
    else:
        sys.stderr.write("Unknown strategy")
    gtp_engine = gtp_lib.Engine(instance)
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

def train(read_file=None, save_file=None, epochs=10, logdir=None, *data_sets):
    test_dataset, training_datasets = load_data_sets(*data_sets)
    training_datasets = list(training_datasets)
    n = PolicyNetwork(DEFAULT_FEATURES.planes)
    n.initialize_variables(read_file)
    if logdir is not None:
        n.initialize_logging(logdir)
    for i in range(epochs):
        for dset in training_datasets:
            n.train(dset)
        n.check_accuracy(test_dataset)
    if save_file is not None:
        n.save_variables(save_file)
        print("Finished training. New model saved to %s" % save_file, file=sys.stderr)



parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, train])

if __name__ == '__main__':
    argh.dispatch(parser)
