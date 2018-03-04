---
title: Overfitting and regularization
date: 2018-02-27 18:02:54
tags: Mxnet Deep_learning
categories: AI
copyright: True
---

# Overfitting and regularization

The goal of supervised learning is to produce models that *generalize* to previously unseen data. When a model achieves low error on training data but performs much worse on test data, we say that the model has ***overfit*.**

下图反映了固定样本数目的情况下，预测误差的情况：

![](http://gluon.mxnet.io/_images/regularization-overfitting.png)

很多因素决定一个模型是否能被很好的泛化（generalize），参数越多的模型复杂度越高，模型参数取值范围越大的模型越复杂。就神经网络而言，训练步数越多越复杂。



不同类别模型的复杂度很难直接比较，一个基本的经验法则是：<u>A model that can readily（方便的） explain *arbitrary* facts is what statisticians view as complex, whereas one that has only a limited expressive power but still manages to（manage to do sth 设法完成，努力完成） explain the data well is probably closer to the truth.</u>

## Regularization

Broadly speaking the family of techniques geared towards（旨在） mitigating（=alleviate 缓解） overfitting are referred to as ***regularization***.

Given the intuition from the previous chart, we might attempt to make our model less complex. 

* One way to do this would be to lower the number of free parameters. For example, we could **throw away**（丢弃） some subset of our input features (and thus the corresponding parameters) that we thought were least informative.

* Another approach is to limit the values that our weights might take. One common approach is to force the weights to **take small values**. We can accomplish this by changing our optimization objective to penalize the value of our weights.The most popular regularizer is the $$ℓ_2^2 $$ norm:
  $$
  \sum_{i}(\hat{y}-y)^2 + \lambda \| \textbf{w} \|^2_2
  $$
  Here, $$\|\textbf{w}\|$$ is the $$l_2^2$$ norm and $$λ$$ is a hyper-parameter that determines how aggressively we want to push the weights towards 0.

  ​


除了上面提到的$L_2$正则化，还有其它的一些手段。Basically we assumed that small weight values are good：

- We could require that the total sum of the weights is small. That is what $$L1$$ regularization does via the penalty $$∑_i|wi|$$.
- We could require that the largest weight is not too large. This is what $$L∞$$ regularization does via the penalty $$max_i|wi|$$.
- We could require that the number of nonzero weights is small, i.e. that the weight vectors are *sparse*. This is what the $$L0$$ penalty does, i.e. $$\sum_i I\{w_i \neq 0\}$$

![](http://gluon.mxnet.io/_images/regularization.png)

From left to right: L2 regularization, which constrains the parameters to a ball, L1 regularization, which constrains the parameters to a diamond (for lack of a better name, this is often referred to as an L1-ball), and $$L_\infty$$ regularization, which constrains the parameters to a hypercube.



All of this raises the question of **why** regularization is any good. There is an entire field of statistics devoted to this issue - Statistical Learning Theory. For now, a few simple rules of thumb（rule of thumb 〔根据实际经验的〕粗略的数字（计算方法）） suffice（v.足够）:

- Fewer parameters tend to be better than more parameters.
- Better engineering for a specific problem that takes the actual problem into account will lead to better models, due to the prior knowledge that data scientists have about the problem at hand.
- L2 is easier to optimize for than L1. In particular, many optimizers will not work well out of the box for L1. Using the latter requires something called *proximal operators*.
- Dropout and other methods to make the model robust to perturbations（小变化） in the data often work better than off-the-shelf（现成的） L2 regularization.

## code 

其中，`gluon.Trainer`中有一个参数`wd`代表*weight decay*，When we add an L2 penalty to the weights we are effectively（实际上） adding $$\frac{\lambda}{2} \|w\|^2$$ to the loss. Hence, every time we compute the gradient it gets an additional $$λw$$ term that is added to $$g_t$$, since this is the very derivative of the L2 penalty. As a result we end up taking a descent step not in the direction $$−\eta gt$$ but rather in the direction $$−\eta (gt+λw)$$. This effectively shrinks $$w$$ at each step by $$\eta λw$$, thus the name weight decay. To make this work in practice we just need to set the weight decay to something nonzero.

```python
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
import mxnet.ndarray as nd
import numpy as np
ctx = mx.gpu(0)

# for plotting purposes
import matplotlib
import matplotlib.pyplot as plt


mnist = mx.test_utils.get_mnist()
num_examples = 1000
batch_size = 64
train_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["train_data"][:num_examples],
                               mnist["train_label"][:num_examples].astype(np.float32)),
                               batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(
    mx.gluon.data.ArrayDataset(mnist["test_data"][:num_examples],
                               mnist["test_label"][:num_examples].astype(np.float32)),
                               batch_size, shuffle=False)

net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(10))


net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)

loss = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.0})


def evaluate_accuracy(data_iterator, net, loss_fun):
    acc = mx.metric.Accuracy()
    loss_avg = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        output = net(data)
        loss = loss_fun(output, label)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
        loss_avg = loss_avg*i/(i+1) + nd.mean(loss).asscalar()/(i+1)
    return acc.get()[1], loss_avg


def plot_learningcurves(loss_tr,loss_ts, acc_tr,acc_ts):
    xs = list(range(len(loss_tr)))

    f = plt.figure(figsize=(12,6))
    fg1 = f.add_subplot(121)
    fg2 = f.add_subplot(122)

    fg1.set_xlabel('epoch',fontsize=14)
    fg1.set_title('Comparing loss functions')
    fg1.semilogy(xs, loss_tr)
    fg1.semilogy(xs, loss_ts)
    fg1.grid(True,which="both")

    fg1.legend(['training loss', 'testing loss'],fontsize=14)

    fg2.set_title('Comparing accuracy')
    fg1.set_xlabel('epoch',fontsize=14)
    fg2.plot(xs, acc_tr)
    fg2.plot(xs, acc_ts)
    fg2.grid(True,which="both")
    fg2.legend(['training accuracy', 'testing accuracy'],fontsize=14)
    f.show()


epochs = 700
moving_loss = 0.
niter=0

loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1, 784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter += 1
        moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
        est_loss = moving_loss/(1-0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net, loss)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net, loss)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 20 == 0:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

# Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)


net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx, force_reinit=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01, 'wd': 0.001})

moving_loss = 0.
niter=0
loss_seq_train = []
loss_seq_test = []
acc_seq_train = []
acc_seq_test = []

for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx).reshape((-1,784))
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            cross_entropy = loss(output, label)
        cross_entropy.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        niter +=1
        moving_loss = .99 * moving_loss + .01 * nd.mean(cross_entropy).asscalar()
        est_loss = moving_loss/(1-0.99**niter)

    test_accuracy, test_loss = evaluate_accuracy(test_data, net,loss)
    train_accuracy, train_loss = evaluate_accuracy(train_data, net, loss)

    # save them for later
    loss_seq_train.append(train_loss)
    loss_seq_test.append(test_loss)
    acc_seq_train.append(train_accuracy)
    acc_seq_test.append(test_accuracy)

    if e % 20 == 0:
        print("Completed epoch %s. Train Loss: %s, Test Loss %s, Train_acc %s, Test_acc %s" %
              (e+1, train_loss, test_loss, train_accuracy, test_accuracy))

## Plotting the learning curves
plot_learningcurves(loss_seq_train,loss_seq_test,acc_seq_train,acc_seq_test)

```

## result

![](http://on7mhq4kh.bkt.clouddn.com/2018-3-1-9-20.png)



![](http://on7mhq4kh.bkt.clouddn.com/2018-3-1-9-21.png)









​