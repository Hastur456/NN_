import numpy as np
import random
from sklearn import datasets
import matplotlib.pyplot as plt


class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

mnist = datasets.load_digits()
dataset = [(mnist.data[i][None, ...], mnist.target[i]) for i in range(len(mnist.target))]
dataset_validation = dataset[0:400]
dataset = dataset[401:1600]
np.seterr(divide='ignore')

#hyperparameters
INPUT_DIM = 64
H1_DIM = 32
H2_DIM = 32
OUT_DIM = 10
BATCH_SIZE = 10
NUM_EPOCHS = 400
LEARNING_RATE = 0.00002


#weights initial
W1 = np.random.rand(INPUT_DIM, H1_DIM)
B1 = np.random.rand(1, H1_DIM)
W2 = np.random.rand(H1_DIM, H2_DIM)
B2 = np.random.rand(1, H2_DIM)
W3 = np.random.rand(H2_DIM, OUT_DIM)
B3 = np.random.rand(1, OUT_DIM)

W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
B1 = (B1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H1_DIM)
B2 = (B2 - 0.5) * 2 * np.sqrt(1/H1_DIM)
W3 = (W3 - 0.5) * 2 * np.sqrt(1/OUT_DIM)
B3 = (B3 - 0.5) * 2 * np.sqrt(1/OUT_DIM)

loss_arr = []
loss_validation_arr = []


def relu(t):
    return np.maximum(t, 0)


def deriv_relu(t):
    return (t >= 0).astype(float)


def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x)


def softmax_batch(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=1, keepdims=True)


def space_cross_entropy(z, y):
    return -np.log(z[0, y])


def space_cross_entropy_butch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, jy in enumerate(y):
        y_full[j, jy] = 1
    return y_full


for _ in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):
        x_batch, y_batch = zip(*dataset[BATCH_SIZE*i : BATCH_SIZE*i+BATCH_SIZE])

        x = np.concatenate(x_batch, axis=0)
        y = np.array(y_batch)

        #forward
        T1 = x @ W1 + B1
        H1 = relu(T1)
        T2 = H1 @ W2 + B2
        H2 = relu(T2)
        T3 = H2 @ W3 + B3
        z = softmax_batch(T3)
        E = sum(space_cross_entropy_butch(z, y))

        loss_arr.append(E)

        #backprop
        y_full = to_full_batch(y, OUT_DIM)
        dE_dT3 = z - y_full
        dE_dB3 = np.sum(dE_dT3, axis=0, keepdims=True)
        dE_dW3 = H2.T @ dE_dT3
        dE_dH2 = dE_dT3 @ W3.T
        dE_dT2 = dE_dH2 * deriv_relu(T2)
        dE_dB2 = np.sum(dE_dT2, axis=0, keepdims=True)
        dE_dW2 = H1.T @ dE_dT2
        dE_dH1 = dE_dT2 @ W2.T
        dE_dT1 = dE_dH1 * deriv_relu(T1)
        dE_dB1 = np.sum(dE_dT1, axis=0, keepdims=True)
        dE_dW1 = x.T @ dE_dT1

        #update
        W1 = W1 - LEARNING_RATE * dE_dW1
        B1 = B1 - LEARNING_RATE * dE_dB1
        W2 = W2 - LEARNING_RATE * dE_dW2
        B2 = B2 - LEARNING_RATE * dE_dB2
        W3 = W3 - LEARNING_RATE * dE_dW3
        B3 = B3 - LEARNING_RATE * dE_dB3


def predict(x):
    T1 = x @ W1 + B1
    H1 = relu(T1)
    T2 = H1 @ W2 + B2
    H2 = relu(T2)
    T3 = H2 @ W3 + B3
    z = softmax_batch(T3)
    return z


def calc_accurasy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    return correct / len(dataset)


accurasy = calc_accurasy()
print(f"Accurasy: {accurasy}")

for x, y in dataset_validation:
    T1 = x @ W1 + B1
    H1 = relu(T1)
    T2 = H1 @ W2 + B2
    H2 = relu(T2)
    T3 = H2 @ W3 + B3
    z = softmax_batch(T3)
    E = space_cross_entropy(z, y)
    loss_validation_arr.append(E)


plt.plot(loss_arr, c="k", label="train loss-epochs")
plt.plot(loss_validation_arr, c="b", label="validation loss-epochs")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(loc="upper right")
plt.show()


