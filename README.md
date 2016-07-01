MuGo: A minimalist Go engine modeled after AlphaGo
==================================================

This is a pure Python implementation of the essential parts of AlphaGo.

The logic / control flow of AlphaGo itself is not very complicated and is replicated here. The secret sauce of AlphaGo is in its various neural networks.

(As I understand it) AlphaGo uses three neural networks during play. The first NN is a slow but accurate policy network. This network is trained to predict human moves (~57% accuracy), and it outputs a list of plausible moves, with probabilities attached to each move. This first NN is used to seed the Monte Carlo tree search with plausible moves. One of the reasons this first NN is slow is because of its size, and because the inputs to the neural network are various computed properties of the Go board (liberty counts; ataris; ladders; etc.). The second NN is a smaller, faster but less accurate (~24% accuracy) policy network, and doesn't use computed properties as input. Once a leaf node of the current MCTS tree is reached, the second faster network is used to play the position out to the end with vaguely plausible moves, and score the end position. The third NN is a value network: it outputs an expected win margin for that board, without attempting to play anything out. The results of the monte carlo playout using NN #2 and the value calculation using NN #3 are averaged, and this value is recorded as the approximate result for that MCTS node.

Using the priors from NN #1 and the accumulating results of MCTS, a new path is chosen for further Monte Carlo exploration.

Playing with/against MuGo
=========================
MuGo uses the GTP protocol, and you can use any gtp-compliant program with it.

For example, to play against MuGo using gogui, you can do:
```
gogui-twogtp -black 'python main.py gtp policy --read-file=/tmp/mymodel' -white 'gogui-display' -size 19 -komi 7.5 -verbose -auto
```

Training
========
To train, run
```
python main.py train --read-file=/tmp/savedmodel --save-file=/tmp/savedmodel --epochs=10 data/kgs_data data/pro_data
```
where `data/kgs/data` and `data/pro_data` are directories of sgf files to be used for training. 