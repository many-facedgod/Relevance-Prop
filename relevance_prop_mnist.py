import gzip
import pickle
import sys

import imageio

from os import mkdir
from os.path import isdir

from tqdm import tqdm, trange

from relevance_network.relevance_network import *


def load_data():
    """Load the MNIST data and normalize it."""
    (trainx, trainy), (valx, valy), (testx, testy) = pickle.load(gzip.open("data/mnist_one_hot.pkl.gz"),
                                                                 encoding="latin1")
    trainy = np.argmax(trainy, axis=1)
    valy = np.argmax(valy, axis=1)
    testy = np.argmax(testy, axis=1)
    trainx = trainx * 2 - 1
    valx = valx * 2 - 1
    testx = testx * 2 - 1
    return (trainx, trainy), (valx, valy), (testx, testy)


def train_classifier(data, n_iters=4, batch_size=100):
    """
    Train a feed-forward classifier on the data
    :param data: The MNIST data loaded
    :param n_iters: The number of iterations to train for
    :param batch_size: The batch size to use
    """
    tqdm.write(f'Training a dilated CNN classifier for {n_iters} iterations.')
    (trainx, trainy), (valx, valy), (testx, testy) = data
    train_size, val_size, test_size = trainx.shape[0], valx.shape[0], testx.shape[0]
    train_batches = (train_size - 1) // batch_size + 1
    val_batches = (val_size - 1) // batch_size + 1
    test_batches = (test_size - 1) // batch_size + 1

    model = RelPropNetwork()
    model.add_layer(RelFCLayer(400)) \
        .add_layer(RelReluLayer()) \
        .add_layer(RelFCLayer(100)) \
        .add_layer(RelReluLayer()) \
        .add_layer(RelFCLayer(50)) \
        .add_layer(RelReluLayer()) \
        .add_layer(RelFCLayer(10)) \
        .add_layer(RelSoftmaxCELayer())
    for i in range(1, n_iters + 1):
        train_order = np.random.permutation(train_size)
        bar = trange(train_batches, file=sys.stdout)
        for j in bar:
            cost = model.forward(trainx[train_order[j * batch_size: (j + 1) * batch_size]],
                                 trainy[train_order[j * batch_size: (j + 1) * batch_size]])
            bar.set_description(f'Curr loss: {cost}')
            model.backward()
            model.adam_trainstep()
        correct = []
        for j in range(val_batches):
            res = model.run(valx[j * batch_size:(j + 1) * batch_size])
            correct.append(np.argmax(res, axis=1) == valy[j * batch_size:(j + 1) * batch_size])
        tqdm.write(f'Validation accuracy: {np.mean(correct)}')
        tqdm.write('-------------------------------------------------------')

    correct = []
    for i in range(test_batches):
        res = model.run(testx[i * batch_size:(i + 1) * batch_size])
        correct.append(np.argmax(res, axis=1) == testy[i * batch_size:(i + 1) * batch_size])
    tqdm.write(f'Test accuracy: {np.mean(correct)}')
    tqdm.write('-------------------------------------------------------')
    return model


def generate_maps(model, data):
    """
    Generate the relevance maps for a data point for each digit and returns the maps corresponding to the correct class.
    """
    tqdm.write('Generating maps...')
    (trainx, trainy), _, _ = data
    indices = [np.where(trainy == i)[0][0] for i in range(10)]
    batchx = trainx[indices]
    batchy = trainy[indices]
    model.forward(batchx, batchy)
    maps = model.backward_relevance()
    tqdm.write('Done.')
    return np.array([maps[i][i].reshape(28, 28) for i in range(10)])


def main():
    if not isdir('./generated_maps'):
        mkdir('./generated_maps')
    data = load_data()
    maps = generate_maps(train_classifier(data), data)
    for i, map_ in enumerate(maps):
        map_ = 255 * (map_ - map_.min()) / (map_.max() - map_.min())
        imageio.imwrite(f'./generated_maps/{i}.png', map_.astype(np.uint8))


if __name__ == '__main__':
    main()
