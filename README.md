py_digitrec
-----------

Digit recognizer using my own neural network library; it also uses numpy and matplotlib.pyplot.
There are two scripts which produce the learning curves for different network sizes and learning rates. It is
interesting to note that they are somewhat correlated: bigger neural networks are more comfortable with
smaller learning rates, perhaps because the error function gets more complicated as the size grows.

This repository uses [git submodule](http://git-scm.com/docs/git-submodule) to manage the dependency with the
neural network library. To make everything work you need to use git submodule init and git submodule update
after git clone; here's an example:
```
emilio@emilio-notebook:~$ git clone git@github.com:e-dorigatti/py_digitrec.git
Cloning into 'py_digitrec'...
remote: Counting objects: 12, done.
remote: Compressing objects: 100% (10/10), done.
remote: Total 12 (delta 0), reused 9 (delta 0)
Receiving objects: 100% (12/12), 7.80 MiB | 474 KiB/s, done.
emilio@emilio-notebook:~$ cd py_digitrec/
emilio@emilio-notebook:~/py_digitrec$ python learning_curves_single.py 
Traceback (most recent call last):
  File "learning_curves_single.py", line 1, in <module>
    from py_neuralnet.neuralnet import NeuralNetwork
ImportError: No module named py_neuralnet.neuralnet
emilio@emilio-notebook:~/py_digitrec$ git submodule init
Submodule 'py_neuralnet' (git@github.com:e-dorigatti/py_neuralnet.git) registered for path 'py_neuralnet'
emilio@emilio-notebook:~/py_digitrec$ git submodule update
Cloning into 'py_neuralnet'...
remote: Reusing existing pack: 28, done.
remote: Total 28 (delta 0), reused 0 (delta 0)
Receiving objects: 100% (28/28), 5.02 KiB, done.
Resolving deltas: 100% (12/12), done.
Submodule path 'py_neuralnet': checked out '7886987b4f13bec91839d30df43a6a33a39b045f'
emilio@emilio-notebook:~/py_digitrec$ python learning_curves_single.py 
Please specify network sizes as parameters
emilio@emilio-notebook:~/py_digitrec$ 
```

