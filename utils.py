import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

class PlotVariables:
    def __init__(self, id, names):
        self.vars = {name:PlotVariable(name) for name in names}
        self.id = id

    def push(self, name, value, value2=False, value3=False, value4=False):
        self.vars[name].push(value, value2, value3, value4)

    def set(self, name, value, value2=False, value3=False, value4=False):
        self.vars[name].set(value, value2, value3, value4)

    def plot(self):
        plt.figure()
        numVars = len(self.vars)
        i = 1
        for k in self.vars:
            plt.subplot(numVars, 1, i)
            self.vars[k].plot()
            i = i + 1


        plt.savefig("utils_plot_{}.pdf".format(self.id), bbox_inches='tight')
        plt.close()


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

    def set(self, value, value2=False, value3=False, value4=False):
        self.value = value
        if type(value2) is np.ndarray:
            self.value2 = value2
        if type(value3) is np.ndarray:
            self.value3 = value3
        if type(value4) is np.ndarray:
            self.value4 = value4

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

        # plt.show()
        score = sqrt(mean_squared_error(self.value, self.value2))
        print("RMSE for shape {}: {} RMSE, Max: {}, Min: {}".format(self.name, score, np.max(self.value), np.min(self.value)))


if __name__ == "__main__":

    var = PlotVariables('elipse', ["testVar", "testVar1"])
    var.push('testVar', 1, 3)
    var.push('testVar', 2, 4)

    var.push('testVar1', 2)
    var.push('testVar1', 2)
    var.plot()