import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Pesos e bias
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        outputs = []

        for i in range(len(inputs)):
            h = np.tanh(np.dot(self.Wxh, inputs[i]) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            outputs.append(y)
        return outputs, h

    def backward(self, inputs, outputs, targets, h_last):
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dhnext = np.zeros_like(h_last)

        for t in reversed(range(len(inputs))):
            dy = outputs[t] - targets[t]
            dWhy += np.dot(dy, h_last.T)
            dby += dy
            dh = np.dot(self.Why.T, dy) + dhnext
            dhraw = (1 - h_last * h_last) * dh
            dbh += dhraw
            dWxh += np.dot(dhraw, inputs[t].T)
            dWhh += np.dot(dhraw, h_last.T)
            dhnext = np.dot(self.Whh.T, dhraw)
        return dWxh, dWhh, dWhy, dbh, dby

    def update(self, dWxh, dWhh, dWhy, dbh, dby):
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby

    def train(self, data, epochs):
        for epoch in range(epochs):
            inputs = [np.array([[i]]) for i in data[:-1]]
            targets = [np.array([[i]]) for i in data[1:]]
            outputs, h_last = self.forward(inputs)
            dWxh, dWhh, dWhy, dbh, dby = self.backward(inputs, outputs, targets, h_last)
            self.update(dWxh, dWhh, dWhy, dbh, dby)

    def predict_next(self, data):
        inputs = [np.array([[i]]) for i in data]
        outputs, _ = self.forward(inputs)
        return outputs[-1][0, 0]


if __name__ == "__main__":
    data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    print(data)
    rnn = RNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
    rnn.train(data, epochs=100)
    prediction = rnn.predict_next(data[-10:])
    print("Valor previsto:", prediction)

