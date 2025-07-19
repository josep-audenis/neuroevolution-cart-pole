import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, genome):
        idx = 0

        self.w1 = genome[idx:idx + input_size * hidden_size].reshape((input_size, hidden_size))
        idx += input_size * hidden_size

        self.b1 = genome[idx:idx + hidden_size]
        idx += hidden_size

        self.w2 = genome[idx:idx + hidden_size * output_size].reshape((hidden_size, output_size))
        idx += hidden_size * output_size

        self.b2 = genome[idx:idx + output_size]

    def forward(self, x):
        x = np.array(x)
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        output = np.tanh(z2)
        
        return output
