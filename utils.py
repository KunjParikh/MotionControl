import numpy as np
import matplotlib.pyplot as plt

class PlotVariable:
    def __init__(self, name):
        self.name = name
        self.value = np.array([])

    def push(self, value):
        self.value = np.append(self.value, value)

    def print(self):
        print(self.value)

    def plot(self):
        plt.plot(self.value, 'b')
        plt.title(self.name)
        plt.show()


if __name__ == "__main__":
    var = PlotVariable("testVar")
    var.push(1)
    var.push(2)
    var.plot()

    var = PlotVariable("testVar1")
    var.push(2)
    var.push(2)
    var.plot()