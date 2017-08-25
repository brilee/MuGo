MuGo: A minimalist Go engine modeled after AlphaGo
==================================================

This is a pure Python implementation of a neural-network based Go AI, using TensorFlow.

Currently, the AI consists solely of a policy network, trained using supervised learning. I have implemented Monte Carlo Tree Search, but the simulations are too slow, due to being written in Python. I am hoping to bypass this issue entirely by replacing the simulations with a value network which will take one NN evaluation. (After all, random simulations are but a crude approximation to a value function, so if you have a good enough value function, you won't need a playout...)

The goal of this project is to see how strong a Go AI based purely on neural networks can be. In other words, a UCT-based tree search with moves seeded by a policy network, and a value network to evaluate the choices. An explicit non-goal is diving into the fiddly bits of optimizing Monte Carlo simulations.

Getting Started
===============

Install Tensorflow
------------------
Start by installing Tensorflow. This should be as simple as

```python
pip install -r requirements.txt
```

Optionally, you can install TensorFlow with GPU support, if you intend on training a network yourself. 

Play against MuGo
=================

If you just want to get MuGo working, you can download a pretrained network from [Releases](https://github.com/brilee/MuGo/releases). You will have to be sure to match the code version with the version specified in the release, or else the neural network configuration may not line up correctly - `git checkout v0.1`, replace with version as appropriate.

MuGo uses the GTP protocol, and you can use any gtp-compliant program with it. To invoke the raw policy network, use
```
python main.py gtp policy --read-file=saved_models/20170718
```

(An MCTS version of MuGo has been implemented, using the policy network to simulate games, but it's not that much better than just the raw policy network, because Python is slow at simulating full games.)

One way to play via GTP is to use gogui-display (which implements a UI that speaks GTP.) You can download the gogui set of tools at [http://gogui.sourceforge.net/](http://gogui.sourceforge.net/). See also [documentation on interesting ways to use GTP](http://gogui.sourceforge.net/doc/reference-twogtp.html).
```
gogui-twogtp -black 'python main.py gtp policy --read-file=saved_models/20170718' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to play against GnuGo, while spectating the games
```
BLACK="gnugo --mode gtp"
WHITE="python main.py gtp policy --read-file=saved_models/20170718"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Another way to play via GTP is to connect to CGOS, the [Computer Go Online Server](http://yss-aya.com/cgos/). The CGOS server hosted by boardspace.net is actually abandoned; you'll want to connect to the CGOS server at yss-aya.com. 

After configuring your cgos.config file, you can connect to CGOS with `cgosGtp -c cgos.config` and spectate your own game with `cgosView yss-aya.com 6819`

Training MuGo
=============

Get SGFs for supervised learning
--------------------------------
You can find 15 years of KGS high-dan games at [u-go.net](https://u-go.net/gamerecords/). A database of Tygem 9d games is also out there, and finally, a database of professional games can be purchased from a variety of sources.

Preprocess SGFs
---------------
To use the game data for training, the game positions must first be processed into feature planes describing location of stones, liberty counts, and so on, as well as noting the correct location of the next move.

```
python main.py preprocess data/kgs-*
```

This will generate a series of data chunks and will take a while. It must be repeated if you change the feature extraction steps in `features.py` (This example takes advantage of bash wildcard expansion - say, if the KGS directories are named data/kgs-2006-01, data/kgs-2006-02, and so on.)

Supervised learning (policy network)
------------------------------------
With the preprocessed SGF data (default output directory is `./processed_data/`), you can train the policy network.
```
python main.py train processed_data/ --save-file=/tmp/savedmodel --epochs=1 --logdir=logs/my_training_run
```

As the network is trained, the current model will be saved at `--save-file`. If you reexecute the same command, the network will pick up training where it left off.

Additionally, you can follow along with the training progress with TensorBoard - if you give each run a different name (`logs/my_training_run`, `logs/my_training_run2`), you can overlay the runs on top of each other.
```
tensorboard --logdir=logs/
```

Running unit tests
------------------
```python
python -m unittest discover tests
```