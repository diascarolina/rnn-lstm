import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    return 1.0 - x ** 2


class LSTM:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size

        # Input gate weights and biases
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bi = np.zeros((hidden_size, 1))

        # Forget gate weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bf = np.zeros((hidden_size, 1))

        # Output gate weights and biases
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bo = np.zeros((hidden_size, 1))

        # Cell state weights and biases
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.bc = np.zeros((hidden_size, 1))

        # Output weights and biases
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = {}
        self.i = {}
        self.f = {}
        self.o = {}
        self.c = {}
        self.h = {}
        self.y = {}

        self.h[-1] = np.zeros((self.hidden_size, 1))
        self.c[-1] = np.zeros((self.hidden_size, 1))

        for t in range(len(inputs)):
            self.z[t] = np.row_stack((self.h[t - 1], inputs[t]))

            # Input gate
            self.i[t] = sigmoid(np.dot(self.Wi, self.z[t]) + self.bi)

            # Forget gate
            self.f[t] = sigmoid(np.dot(self.Wf, self.z[t]) + self.bf)

            # Output gate
            self.o[t] = sigmoid(np.dot(self.Wo, self.z[t]) + self.bo)

            # New cell information
            self.c[t] = self.f[t] * self.c[t - 1] + self.i[t] * tanh(
                np.dot(self.Wc, self.z[t]) + self.bc)

            # Hidden state
            self.h[t] = self.o[t] * tanh(self.c[t])

            # Output
            self.y[t] = np.dot(self.Wy, self.h[t]) + self.by

        return self.y

    def backward(self, targets, learning_rate=0.01):
        # Gradients initialization
        dWi, dWf, dWo, dWc = np.zeros_like(self.Wi), np.zeros_like(self.Wf), np.zeros_like(
            self.Wo), np.zeros_like(self.Wc)
        dbi, dbf, dbo, dbc = np.zeros_like(self.bi), np.zeros_like(self.bf), np.zeros_like(
            self.bo), np.zeros_like(self.bc)
        dWy, dby = np.zeros_like(self.Wy), np.zeros_like(self.by)

        dhnext, dcnext = np.zeros_like(self.h[0]), np.zeros_like(self.c[0])

        for t in reversed(range(len(targets))):
            dy = self.y[t] - targets[t]
            dWy += np.dot(dy, self.h[t].T)
            dby += dy

            dh = np.dot(self.Wy.T, dy) + dhnext
            dc = dcnext + (dh * self.o[t] * dtanh(tanh(self.c[t])))

            do = dh * tanh(self.c[t])
            dWo += np.dot(do * dsigmoid(self.o[t]), self.z[t].T)
            dbo += do * dsigmoid(self.o[t])

            df = dc * self.c[t - 1]
            dWf += np.dot(df * dsigmoid(self.f[t]), self.z[t].T)
            dbf += df * dsigmoid(self.f[t])

            di = dc * tanh(np.dot(self.Wc, self.z[t]) + self.bc)
            dWi += np.dot(di * dsigmoid(self.i[t]), self.z[t].T)
            dbi += di * dsigmoid(self.i[t])

            dc_bar = dc * self.i[t]
            dWc += np.dot(dc_bar * dtanh(np.dot(self.Wc, self.z[t]) + self.bc), self.z[t].T)
            dbc += dc_bar * dtanh(np.dot(self.Wc, self.z[t]) + self.bc)

            dz = (np.dot(self.Wi.T, di * dsigmoid(self.i[t])) +
                  np.dot(self.Wf.T, df * dsigmoid(self.f[t])) +
                  np.dot(self.Wo.T, do * dsigmoid(self.o[t])) +
                  np.dot(self.Wc.T, dc_bar * dtanh(np.dot(self.Wc, self.z[t]) + self.bc)))

            dhnext = dz[:self.hidden_size, :]
            dcnext = self.f[t] * dc

        for dparam in [dWi, dWf, dWo, dWc, dbi, dbf, dbo, dbc, dWy, dby]:
            np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

        # Update weights and biases using gradient descent
        self.Wi -= learning_rate * dWi
        self.Wf -= learning_rate * dWf
        self.Wo -= learning_rate * dWo
        self.Wc -= learning_rate * dWc
        self.bi -= learning_rate * dbi
        self.bf -= learning_rate * dbf
        self.bo -= learning_rate * dbo
        self.bc -= learning_rate * dbc
        self.Wy -= learning_rate * dWy
        self.by -= learning_rate * dby


data = [0.5, 0.6, 0.7, 0.8, 0.85]

inputs = [data[i] for i in range(len(data)-1)]
targets = [data[i+1] for i in range(len(data)-1)]

hidden_size = 5
input_size = output_size = 1  # We are predicting a single value based on a single value

lstm = LSTM(input_size, hidden_size, output_size)

iterations = 10000
learning_rate = 0.01

for i in range(iterations):
    total_loss = 0
    for inp, target in zip(inputs, targets):
        lstm.forward(np.array([[inp]]))  # forward pass
        lstm.backward(np.array([[target]]), learning_rate)  # backward pass
        total_loss += np.square(lstm.y[0] - target).item()  # MSE Loss for the last
        # output
    if i % 1000 == 0:
        print(f"Iteration {i}, Loss: {total_loss/len(inputs)}")

predicted = lstm.forward(np.array([[val] for val in data[:-1]]))
print(f"Predicted next value: {predicted[len(data)-2]}")
