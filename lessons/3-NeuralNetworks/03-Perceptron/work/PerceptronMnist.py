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

def train_graph(positive_examples, negative_examples, num_iterations = 100):
    num_dims = positive_examples.shape[1]
    weights = np.zeros((num_dims,1)) # initialize weights
    
    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]
    
    report_frequency = 15;
    snapshots = []
    
    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)   
        if z < 0:
            weights = weights + pos.reshape(weights.shape)

        z  = np.dot(neg, weights)
        if z >= 0:
            weights = weights - neg.reshape(weights.shape)
            
        if i % report_frequency == 0:             
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)        
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            # make correction a list so it is homogeneous to weights list then numpy array accepts
            snapshots.append((np.concatenate(weights),[(pos_correct+neg_correct)/2.0,0,0]))

    return np.array(snapshots, dtype=object)

# https://github.com/microsoft/AI-For-Beginners/raw/c639951043c67fe7862f5c236d1e4f0cdf68202c/data/mnist.pkl.gz
with gzip.open("./mnist.pkl.gz", "rb") as mnist_pickle:
    MNIST = pickle.load(mnist_pickle)

print("Features:", MNIST["Train"]["Features"][0])
print("Train:", MNIST["Train"]["Labels"][0])
features = MNIST["Train"]["Features"].astype(np.float32) / 256.0
labels = MNIST["Train"]["Labels"]
fig = plt.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(1, 10, i + 1)
    plt.imshow(features[i].reshape(28, 28))
plt.show()


def set_mnist_pos_neg(positive_label, negative_label):
    positive_indices = [
        i for i, j in enumerate(MNIST["Train"]["Labels"]) if j == positive_label
    ]
    negative_indices = [
        i for i, j in enumerate(MNIST["Train"]["Labels"]) if j == negative_label
    ]

    positive_images = MNIST["Train"]["Features"][positive_indices]
    negative_images = MNIST["Train"]["Features"][negative_indices]

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(positive_images[0].reshape(28, 28), cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(negative_images[0].reshape(28, 28), cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return positive_images, negative_images

pos1,neg1 = set_mnist_pos_neg(1,0)

def plotit2(snapshots_mn,step):
    fig = plt.figure(figsize=(10,4))
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(snapshots_mn[step][0].reshape(28, 28), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar()
    ax = fig.add_subplot(1, 2, 2)
    ax.set_ylim([0,1])
    plt.plot(np.arange(len(snapshots_mn[:,1])), snapshots_mn[:,1])
    plt.plot(step, snapshots_mn[step,1], "bo")
    plt.show()

def pl3(step): plotit2(snapshots_mn,step)
# def pl4(step): plotit2(snapshots_mn2,step) 

snapshots_mn = train_graph(pos1,neg1,1000)    
interact(pl3, step=widgets.IntSlider(value=0, min=0, max=len(snapshots_mn) - 1))