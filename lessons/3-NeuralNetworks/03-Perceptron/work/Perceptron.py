import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import pickle
import os
import gzip

# pick the seed for reproducability - change it to explore the effects of random variations
np.random.seed(1)
import random


n = 50

# 通过调用make_classification函数，
# 生成了一个包含50个样本的数据集，
# 每个样本具有2个特征。
# 这个数据集没有冗余特征（n_redundant=0），
# 所有特征都是有信息的（n_informative=2），
# 并且没有对标签进行随机翻转（flip_y=0）
X, Y = make_classification(
    n_samples=n, n_features=2, n_redundant=0, n_informative=2, flip_y=0
)
Y = Y * 2 - 1  # convert initial 0/1 values into -1/1
X = X.astype(np.float32)
Y = Y.astype(np.int32)  # features - float, label - int

# Split the dataset into training and test
train_x, test_x = np.split(X, [n * 8 // 10])
train_labels, test_labels = np.split(Y, [n * 8 // 10])
print("Features:\n", train_x[0:4])
print("Labels:\n", train_labels[0:4])


def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = plt.subplots(1, 1)
    # pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize=16)
    ax.set_xlabel("$x_i[0]$ -- (feature 1)")
    ax.set_ylabel("$x_i[1]$ -- (feature 2)")

    colors = ["r" if l > 0 else "b" for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker="o", c=colors, s=100, alpha=0.5)
    plt.show()


# show graphics
# plot_dataset("Training data", train_x, train_labels)

pos_examples = np.array(
    [[t[0], t[1], 1] for i, t in enumerate(train_x) if train_labels[i] > 0]
)
neg_examples = np.array(
    [[t[0], t[1], 1] for i, t in enumerate(train_x) if train_labels[i] < 0]
)
print("positive examples", pos_examples[0:3], "negtive examples", neg_examples[0:3])

weights_history = []


def train(positive_examples, negative_examples, num_iterations=100):
    num_dims = positive_examples.shape[1]
    print("num_dims", num_dims)
    # Initialize weights.
    # We initialize with 0 for simplicity, but random initialization is also a good idea
    weights = np.zeros((num_dims, 1))
    print("num_dims weights", weights)

    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]

    print("pos_count", pos_count, "neg_count", neg_count)

    report_frequency = 10

    for i in range(num_iterations):
        # Pick one positive and one negative example
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)
        if z < 0:  # positive example was classified as negative
            tmp = weights + pos.reshape(weights.shape)
            print(
                "update weights:",
                weights,
                " pos:",
                pos.reshape(weights.shape),
                " update:",
                tmp,
            )
            weights = tmp

        z = np.dot(neg, weights)
        if z >= 0:  # negative example was classified as positive
            weights = weights - neg.reshape(weights.shape)

        weights_history.append([weights, pos, neg])

        # Periodically, print out the current accuracy on all examples
        if i % report_frequency == 0:
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            print(
                "Iteration={}, pos correct={}, neg correct={}".format(
                    i, pos_correct, neg_correct
                )
            )

    return weights


wts = train(pos_examples, neg_examples)
print("wts", wts, "transpose", wts.transpose(), "weights_history", len(weights_history))

def accuracy(weights, test_x, test_labels):
    res = np.dot(np.c_[test_x,np.ones(len(test_x))],weights)
    return (res.reshape(test_labels.shape)*test_labels>=0).sum()/float(len(test_labels))

accuracy(wts, test_x, test_labels)

def plot_boundary(positive_examples, negative_examples, weights, pos, neg):
    if np.isclose(weights[1], 0):
        if np.isclose(weights[0], 0):
            x = y = np.array([-6, 6], dtype="float32")
        else:
            y = np.array([-6, 6], dtype="float32")
            x = -(weights[1] * y + weights[2]) / weights[0]
    else:
        x = np.array([-6, 6], dtype="float32")
        y = -(weights[0] * x + weights[2]) / weights[1]

    plt.clf()
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.plot(positive_examples[:, 0], positive_examples[:, 1], "bo")
    plt.plot(negative_examples[:, 0], negative_examples[:, 1], "ro")

    if pos is not None:
        plt.plot(pos[0], pos[1], "*", color="lightblue", markersize=10)  # 其他点
        plt.plot(neg[0], neg[1], "*", color="pink", markersize=10)  # 其他点

    plt.plot(x, y, "g", linewidth=2.0)
    plt.draw()


count = 0


def on_key(event):
    global count
    if event.key == "right":
        count += 1
        if count >= len(weights_history):
            count = 0
    elif event.key == "left":
        count -= 1
        if count < 0:
            count = len(weights_history) - 1

    his = weights_history[count]
    wts = his[0]
    pos = his[1]
    neg = his[2]

    print("count", count, "wts", wts)
    plot_boundary(pos_examples, neg_examples, wts, pos, neg)


fig, ax = plt.subplots()
fig.canvas.mpl_connect("key_press_event", on_key)

plot_boundary(pos_examples, neg_examples, wts, None, None)
plt.show()
