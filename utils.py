import numpy as np
import matplotlib.pyplot as plt

class PlotVariable:
    def __init__(self, name):
        self.name = name
        self.value = np.array([])
        self.value2 = np.array([])
        self.value3 = np.array([])
        self.value4 = np.array([])

    def push(self, value, value2=False, value3=False, value4=False):
        self.value = np.append(self.value, value)
        if value2:
            self.value2 = np.append(self.value2, value2)
        if value3:
            self.value3 = np.append(self.value3, value3)
        if value4:
            self.value4 = np.append(self.value4, value4)

    def print(self):
        print(self.value)

    def plot(self):
        plt.plot(self.value, 'r')
        if self.value2.shape[0]>0:
            plt.plot(self.value2, 'g')
        if self.value3.shape[0] > 0:
            plt.plot(self.value3, 'b')
        if self.value4.shape[0] > 0:
            plt.plot(self.value4, 'y')
        plt.title(self.name)
        plt.show()


if __name__ == "__main__":
    var = PlotVariable("testVar")
    var.push(1, 3)
    var.push(2, 4)
    var.plot()

    var = PlotVariable("testVar1")
    var.push(2)
    var.push(2)
    var.plot()