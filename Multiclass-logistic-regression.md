---
title: Multiclass_logistic_regression
date: 2018-02-26 11:26:16
tags: machine_learning
categories: AI
copyright: True
---

深度学习多分类问题，logistics regression模型，使用mnist数据集。

<!--more-->

#Multiclass logistic regression

Given $k$ classes, the most naive way to solve a ***multiclass classification*** problem is to train $k$ different binary classifiers $f_i(x)$.There’s a smarter way to go about this. We could force the output layer to be a discrete probability distribution over the $k$ classes.

We accomplish this by using the ***softmax* function**. Given an input vector $z$, softmax does two things. First, it exponentiates (elementwise) $e^z$, forcing all values to be strictly positive. Then it normalizes so that all values sum to 1.
$$
\text{softmax}(\boldsymbol{z}) = \frac{e^{\boldsymbol{z}} }{\sum_{i=1}^k e^{z_i}}
$$
Because now we have $k$ outputs and not 1 we’ll need weights connecting each of our inputs to each of our outputs. Graphically, the network looks something like this:

![](https://github.com/zackchase/mxnet-the-straight-dope/blob/master/img/simple-softmax-net.png?raw=true)



We generate the linear mapping from inputs to outputs via a matrix-vector product $\boldsymbol{x}W+\boldsymbol{b}$.

The whole model, including the activation function can be written:
$$
\hat{y} = \text{softmax}(\boldsymbol{x} W + \boldsymbol{b})
$$
This model is sometimes called *multiclass logistic regression*. Other common names for it include *softmax regression* and *multinomial regression*.

##About batch training

In the above, we used plain lowercase letters for scalar variables, bolded lowercase letters for **row** vectors, and uppercase letters for matrices.

 Assume we have $d$ inputs and $k$ outputs. Let’s note the shapes of the various variables explicitly as follows:
$$
\underset{1 \times k}{\boldsymbol z} = \underset{1 \times d}{\boldsymbol{x}}\ \underset{d \times k}{W} + \underset{1 \times k}{\boldsymbol{b}}
$$
Often we would one-hot encode the output label. So $\hat{y} = \text{softmax}(\boldsymbol z)$ becomes:
$$
\underset{1 \times k}{\boldsymbol{\hat{y}}_{one-hot}} = \text{softmax}_{one-hot}(\underset{1 \times k}{\boldsymbol z})
$$
When we input a batch of $m$ training examples, we would have matrix $\underset{m \times d}{X}$ that is the vertical stacking of individual training examples $\boldsymbol x_i$, due to the choice of using row vectors.
$$
\begin{split}X=
\begin{bmatrix}
    \boldsymbol x_1 \\
    \boldsymbol x_2 \\
    \vdots \\
    \boldsymbol x_m
\end{bmatrix}
=
\begin{bmatrix}
    x_{11} & x_{12} & x_{13} & \dots  & x_{1d} \\
    x_{21} & x_{22} & x_{23} & \dots  & x_{2d} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    x_{m1} & x_{m2} & x_{m3} & \dots  & x_{md}
\end{bmatrix}\end{split}
$$
${\boldsymbol{\hat{y}}_{one-hot}} = \text{softmax}({\boldsymbol z})$turns into:
$$
Y = \text{softmax}(Z) = \text{softmax}(XW + B)
$$
这里$B$是m*k矩阵，其相当于$\boldsymbol{b}$的m次拷贝，如下图所示：
$$
\begin{split} B =
\begin{bmatrix}
    \boldsymbol b \\
    \boldsymbol b \\
    \vdots \\
    \boldsymbol b
\end{bmatrix}
=
\begin{bmatrix}
    b_{1} & b_{2} & b_{3} & \dots  & b_{k} \\
    b_{1} & b_{2} & b_{3} & \dots  & b_{k} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    b_{1} & b_{2} & b_{3} & \dots  & b_{k}
\end{bmatrix}\end{split}
$$
显然，可以通过broadcasting来直接使用$\boldsymbol{b}$。

Each row of matrix $\underset{m \times k}{Z}$ corresponds to one training example. The softmax function operates on each row of matrix $Z$ and returns a matrix $\underset{m \times k}{Y}$, each row of which corresponds to the one-hot encoded prediction of one training example.

##The MNIST dataset

This time we’re going to work with real data, each a 28 by 28 centrally cropped（裁剪） black & white photograph of a handwritten digit. Our task will be come up with a model that can associate each image with the digit (0-9) that it depicts.

##The cross-entropy loss function

The relevant loss function here is called **cross-entropy** and it may be the most common loss function you’ll find in all of deep learning. That’s because at the moment, classification problems tend to be far more abundant than regression problems.

The basic idea is that we’re going to take a target Y that has been formatted as a one-hot vector, meaning one value corresponding to the correct label is set to 1 and the others are set to 0, e.g.`[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]`.

The basic idea of cross-entropy loss is that we only care about how much probability the prediction assigned to the correct label. In other words, for true label 2, we only care about the component of yhat corresponding to 2

```python
def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat+1e-6))
```

 MXNet’s has an efficient function that <u>simultaneously computes the softmax activation and cross-entropy loss</u>. However, if ever need to get the output probabilities,

```python
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
```

## scratch

```python
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from utils import SGD
import matplotlib.pyplot as plt
import time

start_time = time.time()

mx.random.seed(1)

data_ctx = mx.gpu(0)
model_ctx = mx.gpu(0)

num_inputs = 784  # 28 * 28
num_outputs = 10
num_examples = 60000

W = nd.random_normal(shape=(num_inputs, num_outputs), ctx=model_ctx)

b = nd.random_normal(shape=num_outputs, ctx=model_ctx)

params = [W, b]

for param in params:
    param.attach_grad()


def transform(data, label):
    # cast data and label to floats and normalize data to range [0, 1]
    return data.astype(np.float32)/255, label.astype(np.float32)


def softmax(y_linear):
    exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1, 1)))
    norms = nd.sum(exp, axis=1).reshape((-1, 1))
    return exp / norms  # 矩阵除以列向量，矩阵每一行除以列向量norms的每一行元素。


def net(X):
    y_linear = nd.dot(X, W) + b
    yhat = softmax(y_linear)
    return yhat


def cross_entropy(yhat, y):
    return - nd.sum(y * nd.log(yhat+1e-6))


def evaluate_accuracy(data_iterator, net):
    # 计算精确度
    numerator = 0.  # 分子
    denominator = 0.  # 分母
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        # label_one_hot = nd.one_hot(label, 10)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        numerator += nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


mnist_train = gluon.data.vision.MNIST(root='../data_set/mnist', train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(root='../data_set/mnist', train=False, transform=transform)

'''
# each item is a tuple of an image(28*28) and a label
image, label = mnist_train[0]
print(type(image))
print(image.shape, label)  # 28 * 28 * 1

im = mx.nd.tile(image, (1, 1, 3))  # 把图片按照第三维broadcast，这样matplotlib才能画图
print(im.shape)

plt.imshow(im.asnumpy())
plt.show()
'''


batch_size = 64
train_data = mx.gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)

test_data = mx.gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)


epochs = 20
learning_rate = .005

for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        label_one_hot = nd.one_hot(label, 10)
        with autograd.record():
            output = net(data)
            loss = cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)
        cumulative_loss += nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch {}. Loss: {}, Train_acc {:%}, Test_acc {:%}".format(e, cumulative_loss/num_examples, train_accuracy, test_accuracy))


# Define the function to do prediction
def model_predict(net, data):
    output = net(data)
    return nd.argmax(output, axis=1)


# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mnist_test, batch_size=10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data, (1, 0, 2, 3))
    im = nd.reshape(im, (28, 10*28, 1))
    imtiles = nd.tile(im, (1, 1, 3))
    print(imtiles.shape)

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred = model_predict(net, data.reshape((-1, 784)))
    print('model predictions are:', pred)
    break

end_time = time.time()

print('total time:%.0fs' % (end_time-start_time))


```



## gluon

```python
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
import matplotlib.pyplot as plt

data_ctx = mx.gpu(0)
model_ctx = mx.gpu(0)

mx.random.seed(1)


batch_size = 64
num_inputs = 784
num_outputs = 10
num_examples = 60000


def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)


train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(root='../data_set/mnist', train=True, transform=transform), batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(root='../data_set/mnist', train=False, transform=transform), batch_size, shuffle=False)

net = gluon.nn.Dense(num_outputs)

net.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=model_ctx)

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


print('未训练时精确度 %.2f' % evaluate_accuracy(test_data, net))

epochs = 10
moving_loss = 0


for e in range(epochs):
    cumulative_loss = 0
    for data, label in train_data:
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        cumulative_loss += nd.sum(loss).asscalar()
    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch {}. Loss: {}, Train_acc {:%}, Test_acc {:%}".format(e, cumulative_loss / num_examples, train_accuracy,
                                                                     test_accuracy))

def model_predict(net,data):
    output = net(data.as_in_context(model_ctx))
    return nd.argmax(output, axis=1)


# let's sample 10 random data points from the test set
sample_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform), 10, shuffle=True)
for i, (data, label) in enumerate(sample_data):
    data = data.as_in_context(model_ctx)
    print(data.shape)
    im = nd.transpose(data, (1, 0, 2, 3))
    im = nd.reshape(im, (28, 10*28, 1))
    imtiles = nd.tile(im, (1, 1, 3))

    plt.imshow(imtiles.asnumpy())
    plt.show()
    pred = model_predict(net, data.reshape((-1, 784)))
    print('model predictions are:', pred)
    break

```

