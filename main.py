import argparse
import argh
from contextlib import contextmanager
import os
import random
import re
import sys
import time

from gtp_wrapper import make_gtp_instance
from load_data_sets import DataSet, parse_data_sets
from policy import PolicyNetwork

TRAINING_CHUNK_RE = re.compile(r"train\d+\.chunk.gz")

@contextmanager
def timer(message):
    tick = time.time()
    yield
    tock = time.time()
    print("%s: %.3f" % (message, (tock - tick)))


def gtp(strategy, read_file=None):
    engine = make_gtp_instance(strategy, read_file)
    if engine is None:
        sys.stderr.write("Unknown strategy")
        sys.exit()
    sys.stderr.write("GTP engine ready\n")
    sys.stderr.flush()
    while not engine.disconnect:
        inpt = input()
        # handle either single lines at a time
        # or multiple commands separated by '\n'
        try:
            cmd_list = inpt.split("\n")
        except:
            cmd_list = [inpt]
        for cmd in cmd_list:
            engine_reply = engine.send(cmd)
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

def train(processed_dir, save_file=None, epochs=10, logdir=None, checkpoint_freq=10000):
    test_dataset = DataSet.read(os.path.join(processed_dir, "test.chunk.gz"))
    train_chunk_files = [os.path.join(processed_dir, fname) 
        for fname in os.listdir(processed_dir)
        if TRAINING_CHUNK_RE.match(fname)]
    save_file = os.path.join(os.getcwd(), save_file)
    n = PolicyNetwork()
    try:
        n.initialize_variables(save_file)
    except:
        n.initialize_variables(None)
    if logdir is not None:
        n.initialize_logging(logdir)
    last_save_checkpoint = 0
    for i in range(epochs):
        random.shuffle(train_chunk_files)
        for file in train_chunk_files:
            print("Using %s" % file)
            train_dataset = DataSet.read(file)
            train_dataset.shuffle()
            with timer("training"):
                n.train(train_dataset)
            n.save_variables(save_file)
            if n.get_global_step() > last_save_checkpoint + checkpoint_freq:
                with timer("test set evaluation"):
                    n.check_accuracy(test_dataset)
                last_save_checkpoint = n.get_global_step()



parser = argparse.ArgumentParser()
argh.add_commands(parser, [gtp, preprocess, train])

if __name__ == '__main__':
    argh.dispatch(parser)
