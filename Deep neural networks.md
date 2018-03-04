# Deep neural networks

In the previous chapters we showed how you could implement multiclass logistic regression (also called *softmax regression*) for classifiying images of handwritten digits into the 10 possible categories.

Recall that before, we mapped our inputs directly onto our outputs through a single linear transformation.
$$
\hat{y} = \mbox{softmax}(W \boldsymbol{x} + b)
$$
![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/simple-softmax-net.png?raw=true)

If our labels really were related to our input data by an approximately linear function, then this approach might be adequate. *But linearity is a strong assumption*！

Teasing out（找出答案） what is depicted in an image generally requires allowing more complex relationships between our inputs and outputs, considering the possibility that our pattern might be characterized by interactions among the many features.

We can model a more general class of functions by incorporating one or more *hidden layers*.This architecture is commonly called a “multilayer perceptron”. With an MLP, we’re going to stack a bunch of layers on top of each other.
$$
h_1 = \phi(W_1\boldsymbol{x} + b_1)\\
h_2 = \phi(W_2\boldsymbol{h_1} + b_2)\\
...\\
h_n = \phi(W_n\boldsymbol{h_{n-1}} + b_n)
$$
**Note that each layer requires its own set of parameters**.Here, we’ve denoted the activation function for the hidden layers as $$\phi$$.Because we’re still focusing on *multiclass classification*, we’ll stick with the softmax activation in the output layer.
$$
\hat{y} = \mbox{softmax}(W_y \boldsymbol{h}_n + b_y)
$$
![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/multilayer-perceptron.png?raw=true)











