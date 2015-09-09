py_digitrec
-----------

Digit recognizer using my own neural network library; it also uses numpy and matplotlib.pyplot.

I have put a lot of effort in producing plots to explore and explain the behavior of the network:

 - *learning_curves.py* produces the learning curves for different network sizes and learning rates. It is
interesting to note that they are somewhat correlated: bigger neural networks are more comfortable with
smaller learning rates, perhaps because the error function gets more complicated as the size grows.
Extremely large networks have a flat plot in many cases probably due to a numerical bug in my
library. Networks with more than one hidden layer are slower to train but achieve slightly better
classification performance.

 - *errors.py* analyzes the errors made by the network; we see that the the network is very
confident in its prediction in most of the cases. Also, low 'confidency' values are often associated with
a wrong prediction, therefore, I believe it is possible to accurately detect when the network is wrong
given its outputs.

- *hidden_images.py* trains a 400 x 20 x 10 neural network and plots the connection weights
between the input layer and each neuron of the hidden layer. This picture shows how the neurons "see" the
input image and which parts they think are more important; their "opinion" is then combined by the
output layer's neurons to predict a digit. During training, the output neurons are trained to "trust"
some of the hidden neurons' "opinions" to form a prediction. This is a very simplified version of how
our own brain processes images coming from our eyes: there are different layers of neurons and each neuron
is specialized in detecting particular features such as color, contrast, orientation and so on. The image
is gradually transformed and more complex features are extracted as the image "flows" through layers
until, somehow, we "understand" what we are seeing (whatever understanding means for our brain).

It is amazing to observe such a plot! You can clearly see many strokes which resemble common parts of digits
and, if you are lucky enough, you can even spot an entire digit. Sometimes there are neurons with seemingly
random weights as well; those are useless units and would be removed if the network were to be pruned
(i.e. optimized).

 - *overfitting.py* trains a neural network indefinitely and plots the training error and the validation error
as training proceeds. In theory, the validation error should decrease up to a certain point, after which it
starts increasing; this is called overfitting. Overfitting is bad because the model is not learning how to
generalize the dataset but

 - *network_train.py* trains a full neural network and saves it to a file using the pickle module. It allows
to specify different stopping criteria, such as training/validation error, accuracy or number of epochs
(you can even mix them in an OR fashion). It allows to customize different learning parameters such
as the learning rate, the batch size, the regularization coefficient and the network architecture.

 - *autoencoder.py* trains an autoencoder, i.e. a neural network capable of reconstructing its input via an
intermediate, compressed representation of it (the hidden layer). The training stops after the reconstruction
error is less than 1.5 or the network has been trained on 4500 images and after this the intermediate 
representation is show just like the hidden images script does and a digit is reconstructed.

 - *reconstruction.py* trains a network then tries to reconstruct digits by modifying a random noise image.
Results are pretty disappointing :)

Digits are stored in digits.txt; there is a digit per line, 5000 digits in total. Each digit is a 20x20, black
and white image where the intensity of each pixel is a floating point value. Therefore, each line is a list of
400 floating point values storing pixel intensities starting from the first _column_ of the image; in other
words, you need to transpose the 20x20 matrix to obtain the correct image.
Images are equally distributed between digits: the first 500 are 0s, from 501 to 1000 there are 1s and so on.
These digits are a subset of the [MNIST handwritten digits dataset](http://yann.lecun.com/exdb/mnist/).
[As you can see](http://arxiv.org/abs/1003.0358), a properly trained feedforward neural network is able to achieve the
incredible precision of 99.65% on the complete dataset.

This repository uses [git submodule](http://git-scm.com/docs/git-submodule) to manage the dependency with the
neural network library. To make everything work you need to use git submodule init and git submodule update
after git clone; here's an example:
```
$ git clone git@github.com:e-dorigatti/py_digitrec.git
Cloning into 'py_digitrec'...
remote: Counting objects: 12, done.
remote: Compressing objects: 100% (10/10), done.
remote: Total 12 (delta 0), reused 9 (delta 0)
Receiving objects: 100% (12/12), 7.80 MiB | 474 KiB/s, done.
$ cd py_digitrec/
/py_digitrec$ python learning_curves_single.py 
Traceback (most recent call last):
  File "learning_curves_single.py", line 1, in <module>
    from py_neuralnet.neuralnet import NeuralNetwork
ImportError: No module named py_neuralnet.neuralnet
/py_digitrec$ git submodule init
Submodule 'py_neuralnet' (git@github.com:e-dorigatti/py_neuralnet.git) registered for path 'py_neuralnet'
/py_digitrec$ git submodule update
Cloning into 'py_neuralnet'...
remote: Reusing existing pack: 28, done.
remote: Total 28 (delta 0), reused 0 (delta 0)
Receiving objects: 100% (28/28), 5.02 KiB, done.
Resolving deltas: 100% (12/12), done.
Submodule path 'py_neuralnet': checked out '7886987b4f13bec91839d30df43a6a33a39b045f'
/py_digitrec$ python learning_curves_single.py 
Please specify network sizes as parameters
/py_digitrec$ 
```

