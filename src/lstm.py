import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(y):
    return y * (1 - y)


def tanh_derivative(y):
    return 1 - y ** 2


class LSTM:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.Wf = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wi = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wc = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wo = np.random.randn(hidden_size, hidden_size + input_size) * 0.01
        self.Wy = np.random.randn(output_size, hidden_size) * 0.01

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h_prev = np.zeros((self.hidden_size, 1))
        c_prev = np.zeros((self.hidden_size, 1))

        self.hiddens = [h_prev]
        self.cells = [c_prev]

        for x in inputs:
            z = np.row_stack((h_prev, x))
            f = sigmoid(np.dot(self.Wf, z) + self.bf)
            i = sigmoid(np.dot(self.Wi, z) + self.bi)
            c_bar = np.tanh(np.dot(self.Wc, z) + self.bc)
            c = f * c_prev + i * c_bar
            o = sigmoid(np.dot(self.Wo, z) + self.bo)
            h = o * np.tanh(c)

            h_prev, c_prev = h, c
            self.hiddens.append(h)
            self.cells.append(c)

        y = np.dot(self.Wy, h) + self.by
        return y

    def backward(self, inputs, output, target):
        dh_next = np.zeros_like(self.hiddens[0])
        dc_next = np.zeros_like(self.cells[0])
        dWy = np.zeros_like(self.Wy)
        dby = np.zeros_like(self.by)
        dWf = np.zeros_like(self.Wf)
        dWi = np.zeros_like(self.Wi)
        dWc = np.zeros_like(self.Wc)
        dWo = np.zeros_like(self.Wo)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbc = np.zeros_like(self.bc)
        dbo = np.zeros_like(self.bo)

        dy = output - target
        dWy += np.dot(dy, self.hiddens[-1].T)
        dby += dy

        dh = np.dot(self.Wy.T, dy)
        dh_next += dh

        for t in reversed(range(len(inputs))):
            z = np.row_stack((self.hiddens[t], inputs[t]))
            c_prev = self.cells[t]
            c = self.cells[t + 1]
            h = self.hiddens[t + 1]

            do = dh_next * np.tanh(c)
            dc = (dh_next * self.hiddens[-1] * tanh_derivative(np.tanh(c))) + dc_next
            c_bar = dc * self.hiddens[t]
            di = dc * c_prev
            df = dc * c_prev

            dWo += np.dot(do * dsigmoid(self.hiddens[-1]), z.T)
            dbo += do * dsigmoid(self.hiddens[-1])

            dWc += np.dot(c_bar * tanh_derivative(c_bar), z.T)
            dbc += c_bar * tanh_derivative(c_bar)

            dWi += np.dot(di * dsigmoid(self.hiddens[t]), z.T)
            dbi += di * dsigmoid(self.hiddens[t])

            dWf += np.dot(df * dsigmoid(self.hiddens[t]), z.T)
            dbf += df * dsigmoid(self.hiddens[t])

            dz = (np.dot(self.Wf.T, df * dsigmoid(self.hiddens[t])) +
                  np.dot(self.Wi.T, di * dsigmoid(self.hiddens[t])) +
                  np.dot(self.Wc.T, c_bar * tanh_derivative(c_bar)) +
                  np.dot(self.Wo.T, do * dsigmoid(self.hiddens[-1])))

            dh_next = dz[:self.hidden_size, :]
            dc_next = self.hiddens[t] * dc

        gradients = (dWy, dby, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo)
        return gradients

    def update(self, gradients):
        dWy, dby, dWf, dbf, dWi, dbi, dWc, dbc, dWo, dbo = gradients

        self.Wy -= self.learning_rate * dWy
        self.by -= self.learning_rate * dby

        self.Wf -= self.learning_rate * dWf
        self.bf -= self.learning_rate * dbf

        self.Wi -= self.learning_rate * dWi
        self.bi -= self.learning_rate * dbi

        self.Wc -= self.learning_rate * dWc
        self.bc -= self.learning_rate * dbc

        self.Wo -= self.learning_rate * dWo
        self.bo -= self.learning_rate * dbo

    def train(self, data, epochs=10):
        for epoch in range(epochs):
            inputs = [np.array([[i]]) for i in data[:-1]]
            target = np.array([[data[-1]]])
            output = self.forward(inputs)
            gradients = self.backward(inputs, output, target)
            self.update(gradients)

    def predict_next(self, data):
        inputs = [np.array([[i]]) for i in data]
        prediction = self.forward(inputs)
        return prediction[0, 0]


if __name__ == "__main__":
    data = [0.4, 0.5, 0.6, 0.7, 0.8]
    print(data)
    lstm = LSTM(input_size=1, hidden_size=10, output_size=1, learning_rate=0.01)
    lstm.train(data, epochs=100)
    prediction = lstm.predict_next(data[-10:])
    print("Next predicted value:", prediction)
