import argparse
import argh
import os
import random
import re
import sys
import gtp as gtp_lib

from policy import PolicyNetwork
from strategies import RandomPlayer, PolicyNetworkBestMovePlayer, PolicyNetworkRandomMovePlayer, MCTS
from load_data_sets import DataSet, parse_data_sets

TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

def gtp(strategy, read_file=None):
    n = PolicyNetwork(use_cpu=True)
    if strategy == 'random':
        instance = RandomPlayer()
    elif strategy == 'policy':
        instance = PolicyNetworkBestMovePlayer(n, read_file)
    elif strategy == 'randompolicy':
        instance = PolicyNetworkRandomMovePlayer(n, read_file)
    elif strategy == 'mcts':
        instance = MCTS(n, read_file)
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

    test_chunk, training_chunks = parse_data_sets(*data_sets)
    print("Allocating %s positions as test; remainder as training" % len(test_chunk), file=sys.stderr)

    print("Writing test chunk")
    test_dataset = DataSet.from_positions_w_context(test_chunk, is_test=True)
    test_filename = os.path.join(processed_dir, "test.chunk.gz")
    test_dataset.write(test_filename)

    training_datasets = map(DataSet.from_positions_w_context, training_chunks)
    for i, train_dataset in enumerate(training_datasets):
        if i % 10 == 0:
            print("Writing training chunk %s" % i)
        train_filename = os.path.join(processed_dir, "train%s.chunk.gz" % i)
        train_dataset.write(train_filename)
    print("%s chunks written" % (i+1))

def train(processed_dir, read_file=None, save_file=None, epochs=10, logdir=None, checkpoint_freq=10000):
    test_dataset = DataSet.read(os.path.join(processed_dir, "test.chunk.gz"))
    train_chunk_files = [os.path.join(processed_dir, fname) 
        for fname in os.listdir(processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]
    if read_file is not None:
        read_file = os.path.join(os.getcwd(), save_file)
    n = PolicyNetwork()
    n.initialize_variables(read_file)
    if logdir is not None:
        n.initialize_logging(logdir)
    last_save_checkpoint = 0
    for i in range(epochs):
        random.shuffle(train_chunk_files)
        for file in train_chunk_files:
            print("Using %s" % file)
            train_dataset = DataSet.read(file)
            n.train(train_dataset)
            n.save_variables(save_file)
            if n.get_global_step() > last_save_checkpoint + checkpoint_freq:
                n.check_accuracy(test_dataset)
                last_save_checkpoint = n.get_global_step()



parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, preprocess, train])

if __name__ == '__main__':
    argh.dispatch(parser)
