MuGo: A minimalist Go engine modeled after AlphaGo
==================================================

This is a pure Python implementation of the essential parts of AlphaGo.

The logic / control flow of AlphaGo itself is not very complicated and is replicated here. The secret sauce of AlphaGo is in its various neural networks.

(As I understand it) AlphaGo uses three neural networks during play. The first NN is a slow but accurate policy network. This network is trained to predict human moves (~57% accuracy), and it outputs a list of plausible moves, with probabilities attached to each move. This first NN is used to seed the Monte Carlo tree search with plausible moves. One of the reasons this first NN is slow is because of its size, and because the inputs to the neural network are various expensive-to-compute properties of the Go board (liberty counts; ataris; ladder status; etc.). The second NN is a smaller, faster but less accurate (~24% accuracy) policy network, and doesn't use computed properties as input. Once a leaf node of the current MCTS tree is reached, the second faster network is used to play the position out to the end with vaguely plausible moves, and score the end position. The third NN is a value network: it outputs an expected win margin for that board, without attempting to play anything out. The results of the monte carlo playout using NN #2 and the value calculation using NN #3 are averaged, and this value is recorded as the approximate result for that MCTS node.

Using the priors from NN #1 and the accumulating results of MCTS, a new path is chosen for further Monte Carlo exploration.

Getting Started
===============

Install Tensorflow
------------------
Start by installing Tensorflow along with GPU drivers (i.e. CUDA support for Nvidia cards).

Get SGFs for supervised learning
--------------------------------
Second, find a source of SGF files. You can find 15 years of KGS high-dan games at [u-go.net](https://u-go.net/gamerecords/). Alternately, you can download a database of professional games from a variety of sources.

Preprocess SGFs
---------------
Third, preprocess the SGF files. This takes all positions in the SGF files and extracts features for each position, as well as recording the correct next move. These positions are then split into chunks, with one test chunk and the remainder as training chunks. This step may take a while, and must be repeated if you change the feature extraction steps in `features.py`
```
python main.py preprocess data/kgs-*
```
(This example takes advantage of bash wildcard expansion - say, if the KGS directories are named data/kgs-2006-01, data/kgs-2006-02, and so on.)

Supervised learning (policy network)
------------------------------------
With the preprocessed SGF data (default output directory is `./processed_data/`), you can train the policy network.
```
python main.py train processed_data/ --save-file=/tmp/savedmodel --epochs=1 --logdir=logs/my_training_run
```

As the network is trained, the current model will be saved at `--save-file`. You can resume training the same network as follows:
```
python main.py train processed_data/ --read-file=/tmp/savedmodel
 --save-file=/tmp/savedmodel --epochs=10 --logdir=logs/my_training_run
```

Additionally, you can follow along with the training progress with TensorBoard - if you give each run a different name (`logs/my_training_run`, `logs/my_training_run2`), you can overlay the runs on top of each other.
```
tensorboard --logdir=logs/
```

Play against MuGo
-----------------
MuGo uses the GTP protocol, and you can use any gtp-compliant program with it. To invoke the raw policy network, use
```
python main.py gtp policy --read-file=/tmp/savedmodel
```

To invoke the MCTS-integrated version of the policy network, use
```
python main.py gtp mcts --read-file=/tmp/savedmodel
```

One way to play via GTP is to use gogui-display (which implements a UI that speaks GTP.) You can download the gogui set of tools at [http://gogui.sourceforge.net/](http://gogui.sourceforge.net/). See also [documentation on interesting ways to use GTP](http://gogui.sourceforge.net/doc/reference-twogtp.html).
```
gogui-twogtp -black 'python main.py gtp policy --read-file=/tmp/savedmodel' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Another way to play via GTP is to play against GnuGo, while spectating the games
```
BLACK="gnugo --mode gtp"
WHITE="python main.py gtp policy --read-file=/tmp/savedmodel"
TWOGTP="gogui-twogtp -black \"$BLACK\" -white \"$WHITE\" -games 10 \
  -size 19 -alternate -sgffile gnugo"
gogui -size 19 -program "$TWOGTP" -computer-both -auto
```

Another way to play via GTP is to connect to CGOS, the [Computer Go Online Server](http://yss-aya.com/cgos/). The CGOS server hosted by boardspace.net is actually abandoned; you'll want to connect to the CGOS server at yss-aya.com. 

After configuring your cgos.config file, you can connect to CGOS with `cgosGtp -c cgos.config` and spectate your own game with `cgosView yss-aya.com 6819`


Running unit tests
------------------
```python
python -m unittest discover tests
```