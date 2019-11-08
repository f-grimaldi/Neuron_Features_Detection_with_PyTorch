# Neuron_Features_Detection_with_PyTorch
A project to inspect neuron features doing gradient descent w.r.t to the input

First it has been created a low level Neural Network class ('NN.py') implmented with pytorch, which gives the possiblity to inspect the activations of all the neurons. <br>
With the possibility to get the activations of all the neurons it has been created a FeatureDetector class ('FeatureDetector.py') that perfrom Stochastic Gradient Descent w.r.t to the input in order to maximise the activation of a given neuron, by doing this we are able to retrieve which are the features which excite our neuron. <br>
Note that with this tecniques it's easy to perform Adversiarial Attack on the Network and also to train it in a more robust way.

The code is written in python 3.6 by using the following packages:
  - pandas
  - numpy
  - matplotlib
  - cv2
  - sklearn
  - scipy
  - pytorch
  - time
  - itertools
  
