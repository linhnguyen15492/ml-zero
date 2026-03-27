from time import sleep

import numpy as np
from tqdm import tqdm


def plot_regression_line(X, y, x_label, y_label, title):
    import matplotlib.pyplot as plt

    plt.scatter(X, y, marker="x", c="r")

    # Set the title
    plt.title(title)
    # Set the y-axis label
    plt.ylabel(y_label)
    # Set the x-axis label
    plt.xlabel(x_label)
    plt.show()


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]
    total_cost = 0

    for i in tqdm(range(m)):
        f_wb = w * x[i] + b
        cost = (y[i] - f_wb) ** 2
        total_cost += cost

    total_cost = total_cost / (2 * m)

    return total_cost


def linear_regression(X, y):
    # Number of training examples
    m = len(X)

    # Calculate the mean of X and y
    mean_X = np.mean(X)
    mean_y = np.mean(y)

    # Calculate the slope (b1) and intercept (b0)
    b1 = np.sum((X - mean_X) * (y - mean_y)) / np.sum((X - mean_X) ** 2)
    b0 = mean_y - b1 * mean_X

    return b0, b1


def load_data(path):
    data = np.loadtxt(path, delimiter=",")
    X = data[:, 0]
    y = data[:, 1]
    return X, y


X, y = load_data("../data/ex1data1.txt")
# plot_regression_line(X, y, x_label="Population of City in 10,000s", y_label='Profit in $10,000',
#                      title="Profits vs. Population per city")

for i in tqdm(range(100)):
    print(f"training model... iter {i}")
