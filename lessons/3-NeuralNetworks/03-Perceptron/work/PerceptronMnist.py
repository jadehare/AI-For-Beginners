import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import pickle
import os
import gzip
from sklearn.decomposition import PCA

# pick the seed for reproducability - change it to explore the effects of random variations
np.random.seed(1)
import random


def train_graph(positive_examples, negative_examples, num_iterations=100):
    num_dims = positive_examples.shape[1]
    weights = np.zeros((num_dims, 1))  # initialize weights

    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]

    report_frequency = 15
    snapshots = []

    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)
        if z < 0:
            weights = weights + pos.reshape(weights.shape)

        z = np.dot(neg, weights)
        if z >= 0:
            weights = weights - neg.reshape(weights.shape)

        if i % report_frequency == 0:
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            # make correction a list so it is homogeneous to weights list then numpy array accepts
            accuracy = (pos_correct + neg_correct) / 2.0
            shot = (
                np.concatenate(weights),
                accuracy,
            )
            snapshots.append(shot)

    da = np.array(snapshots)

    return da


# https://github.com/microsoft/AI-For-Beginners/raw/c639951043c67fe7862f5c236d1e4f0cdf68202c/data/mnist.pkl.gz
with gzip.open("./mnist.pkl.gz", "rb") as mnist_pickle:
    MNIST = pickle.load(mnist_pickle)

print("Features:", MNIST["Train"]["Features"][0])
print("Train:", MNIST["Train"]["Labels"][0])
features = MNIST["Train"]["Features"].astype(np.float32) / 256.0
labels = MNIST["Train"]["Labels"]
# fig = plt.figure(figsize=(10, 5))
# for i in range(10):
#     ax = fig.add_subplot(1, 10, i + 1)
#     plt.imshow(features[i].reshape(28, 28))
# plt.show()


def set_mnist_pos_neg(positive_label, negative_label):
    positive_indices = [
        i for i, j in enumerate(MNIST["Train"]["Labels"]) if j == positive_label
    ]
    negative_indices = [
        i for i, j in enumerate(MNIST["Train"]["Labels"]) if j == negative_label
    ]

    positive_images = MNIST["Train"]["Features"][positive_indices]
    negative_images = MNIST["Train"]["Features"][negative_indices]

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # plt.imshow(positive_images[0].reshape(28, 28), cmap="gray", interpolation="nearest")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax = fig.add_subplot(1, 2, 2)
    # plt.imshow(negative_images[0].reshape(28, 28), cmap="gray", interpolation="nearest")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.show()

    return positive_images, negative_images


pos1, neg1 = set_mnist_pos_neg(1, 0)
fig = plt.figure(figsize=(10, 4))


def plotit2(snapshots_mn, step):
    global fig
    plt.clf()

    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(snapshots_mn[step][0].reshape(28, 28), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylim([0, 1])

    ssl = len(snapshots_mn[:, 1])
    arange = np.arange(ssl)
    print("arange", arange)

    data = snapshots_mn[:, 1]
    print(
        "data",
        data,
        type(data),
        data.ndim,
        data.shape,
        data.size,
        data.dtype,
        data.itemsize,
    )

    plt.plot(arange, data)
    plt.plot(step, data[step], "bo")


def pl3(step):
    plotit2(snapshots_mn, step)
    plt.draw()


# def pl4(step): plotit2(snapshots_mn2,step)

snapshots_mn = train_graph(pos1, neg1, 1000)

count = 0


def on_key(event):
    global count
    print("event", event)
    if event.key == "right":
        count += 1
        if count >= len(snapshots_mn):
            count = 0
    elif event.key == "left":
        count -= 1
        if count < 0:
            count = len(snapshots_mn) - 1
    pl3(count)


fig.canvas.mpl_connect("key_press_event", on_key)
pl3(0)
plt.show()


def pca_analysis(positive_label, negative_label):
    positive_images, negative_images = set_mnist_pos_neg(positive_label, negative_label)
    M = np.append(positive_images, negative_images, 0)

    mypca = PCA(n_components=2)
    mypca.fit(M)

    pos_points = mypca.transform(positive_images[:200])
    neg_points = mypca.transform(negative_images[:200])

    plt.plot(pos_points[:, 0], pos_points[:, 1], "bo")
    plt.plot(neg_points[:, 0], neg_points[:, 1], "ro")


pca_analysis(1, 0)
plt.show()
