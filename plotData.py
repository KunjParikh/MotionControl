import pickle as pkl
import matplotlib.pyplot as plt
import params
from pandas.plotting import lag_plot
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np

def plotFunction(collectedData, shapeName):

    # Plot shape
    fg = params.FunctionGenerator()
    fn = fg.getFunction(shapeName)
    x = np.linspace(-10, 10, 200)
    y = np.linspace(-10, 10, 200)
    X, Y = np.meshgrid(x, y)
    Z = fn.f(X, Y)
    plt.contour(x, y, Z, [fn.z_desired])
    plt.show()

    # Get field value, dz_x, dz_y
    d = collectedData[shapeName]
    z = d['s'].apply(lambda x: x[0])
    dz_x = d['s'].apply(lambda x: x[1])
    dz_y = d['s'].apply(lambda x: x[2])

    # data
    plt.figure()
    plt.title("data")
    i = 1
    for x in [z, dz_x, dz_y]:
        ax = plt.subplot(310+i)
        i = i + 1
        ax.set_title("data-{}".format(i))
        ax.plot(x)
    plt.show()

    # Histogram
    plt.figure()
    plt.title("Histogram")
    for x in [z, dz_x, dz_y]:
        x.hist()
    plt.show()

    # Density plots
    plt.figure()
    plt.title("Density plots")
    for x in [z, dz_x, dz_y]:
        x.plot(kind='kde')
    plt.show()

    # Box-whisker and heatmaps dont make sense

    # Lag scatter plots
    for x in [z, dz_x, dz_y]:
        values = pd.DataFrame(x.values)
        lags = 8
        mult = 20
        columns = [values]
        for i in range(1,(lags + 1)):
            columns.append(values.shift(i*mult))
        dataframe = pd.concat(columns, axis=1)
        columns = ['t']
        for i in range(1,(lags + 1)):
            columns.append('t-' + str(i*mult))
        dataframe.columns = columns
        plt.figure()
        plt.title("Lag scatter plots")
        for i in range(1, (lags + 1)):
            ax = plt.subplot(240 + i)
            ax.set_title('t vs t-' + str(i*mult))
            plt.scatter(x=dataframe['t'].values, y=dataframe['t-' + str(i*mult)].values)
        plt.show()

    # # Auto Correlation plot
    # for x in [z, dz_x, dz_y]:
    #     autocorrelation_plot(x)
    # plt.show()

    plt.figure()
    i = 1
    for x in [z, dz_x, dz_y]:
        ax = plt.subplot(310+i)
        i = i + 1
        plot_acf(x, ax=ax, lags= 9000)
    plt.show()

    i = 1
    for x in [z, dz_x, dz_y]:
        plot_pacf(x, title="pacf-{}".format(i))
        i = i + 1
    plt.show()

shapesData = pkl.load(open("shapesData.p", "rb"))
plotFunction(shapesData, "circle_1")
plotFunction(shapesData, "elipse_1")
plotFunction(shapesData, "irregular1_1")