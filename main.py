import argparse
import argh
import os
import random
import re
import sys
import gtp as gtp_lib

from features import DEFAULT_FEATURES
from strategies import RandomPlayer, PolicyNetworkBestMovePlayer, MCTS
from policy import PolicyNetwork
from load_data_sets import process_raw_data, DataSet

TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

def gtp(strategy, read_file=None):
    if strategy == 'random':
        instance = RandomPlayer()
    elif strategy == 'policy':
        policy_network = PolicyNetwork(DEFAULT_FEATURES.planes)
        policy_network.initialize_variables(read_file)
        instance = PolicyNetworkBestMovePlayer(policy_network)
    elif strategy == 'mcts':
        policy_network = PolicyNetwork(DEFAULT_FEATURES.planes)
        policy_network.initialize_variables(read_file)
        instance = MCTS(policy_network)
    else:
        sys.stderr.write("Unknown strategy")
        sys.exit()
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

def preprocess(*data_sets, processed_dir="processed_data"):
    processed_dir = os.path.join(os.getcwd(), processed_dir)
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)

    test_dataset, training_datasets = process_raw_data(*data_sets)

    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)

    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("Writing chunk %s" % i)
        train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
        train_dataset.write(train_filename)
    print("%s chunks written" % i)


def train(processed_dir, read_file=None, save_file=None, epochs=10, logdir=None):
    test_dataset = DataSet.read(os.path.join(processed_dir, "test.chunk.gz"))
    train_chunk_files = [os.path.join(processed_dir, fname) 
        for fname in os.listdir(processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]
    n = PolicyNetwork(DEFAULT_FEATURES.planes)
    n.initialize_variables(read_file)
    if logdir is not None:
        n.initialize_logging(logdir)
    for i in range(epochs):
        random.shuffle(train_chunk_files)
        for file in train_chunk_files:
            print("Using %s" % file)
            train_dataset = DataSet.read(file)
            n.train(train_dataset)
            n.check_accuracy(test_dataset)
    if save_file is not None:
        n.save_variables(save_file)
        print("Finished training. New model saved to %s" % save_file, file=sys.stderr)



parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, preprocess, train])

if __name__ == '__main__':
    argh.dispatch(parser)
