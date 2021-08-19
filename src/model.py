import torch.nn as nn


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()

        self.linear_l1 = nn.Linear(input_size, hidden_size)
        self.linear_l2 = nn.Linear(hidden_size, hidden_size)
        self.linear_l3 = nn.Linear(hidden_size, num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.linear_l1(x)
        out = self.relu(out)
        out = self.linear_l2(out)
        out = self.relu(out)
        out = self.linear_l3(out)

        return out
