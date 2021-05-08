import sys

import pandas as pd
import torch
import torch.optim as optim

from capture import readCommand, runGames


class LinearRegression(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


def model(x_input):
    return A.mm(x_input) + b


def loss(y_predicted, y_target):
    return ((y_predicted - y_target)**2).sum()


# number of input columns
n = 2

# Load the entire CSV file
D = torch.tensor(pd.read_csv(".csv", header=None).values, dtype=torch.float)

# Extract the input columns
x_dataset = D[:, 0:n].t()

# Extract the output column
y_dataset = D[:, n].t()

A = torch.randn((1, n), requires_grad=True)
b = torch.randn(1, requires_grad=True)

optimizer = optim.Adam([A, b], lr=0.1)

for t in xrange(2000):
    # Set the gradients to 0
    optimizer.zero_grad()
    # Compute the current predicted y's from x_dataset
    y_predicted = model(x_dataset)
    # See how far off the prediction is
    current_loss = loss(y_predicted, y_dataset)
    # Compute the gradient of the loss with respect to A and b
    current_loss.backward()
    # Update A and b accordingly
    optimizer.step()
    print "t = %s, loss = %s, A = %s, b = %s" % (
        t, current_loss, A.detach().numpy(), b.item())

if __name__ == '__main__':
    """
    The main function called when pacman.py is run
    from the command line:

    > python capture.py

    See the usage string for more details.

    > python capture.py --help
    """
    options = readCommand(sys.argv[1:]) # Get game components based on input
    games = runGames(**options)
