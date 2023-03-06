import numpy as np
import matplotlib.pyplot as plt


def time_line_plotter(data, person_id='10.130.2.1', day=21):
    data = data.loc[data['person_id'] == person_id]
    arr = data.loc[data['day'] == day]['timeSerious'].to_numpy()
    unique, counts = np.unique(arr, return_counts=True)
    mass = np.arange(0, 1440)
    line_space = np.zeros(1440)
    line_space[unique] = counts
    plt.bar(mass, line_space, color='red', width=0.4)
    plt.show()


def general_time_line_plotter(data):
    plot_data = data['timeSerious'].value_counts()
    xPlot = np.array(plot_data.index)
    yPlot = np.array(plot_data.values)
    plt.bar(xPlot, yPlot)
    plt.show()


def works_plotter_NTB_ipynb(data):  # as always NTB means NO Time Base
    ids = data['person_id'].unique()
    daysOfIndividualId = data.loc[data['person_id'] == ids[0]]['day'].unique()
    # ---------------------------------------------------------------------------------
    days_len = len(daysOfIndividualId)
    figure, axis = plt.subplots(days_len)
    figure.tight_layout(pad=0.5)
    figure.set_figheight(90)
    figure.set_figwidth(30)
    # ---------------------------------------------------------------------------------
    ID = ids[0]
    for i in range(len(daysOfIndividualId)):
        day = daysOfIndividualId[i]
        y = data.loc[(data['person_id'] == ID) & (data['day'] == day)]['work_to_number'].to_numpy()
        x = np.arange(len(y))
        axis[i].scatter(x, y, c='orange')
        name = f"person_id:{str(ID)}day:{str(day)}"
        axis[i].set_title(name)
    plt.show()


def works_plotter_NTB(data, person_id='10.131.2.1'):  # as always NTB means NO Time Base
    # ids = data['person_id'].unique()
    daysOfIndividualId = data.loc[data['person_id'] == person_id]['day'].unique()
    for i in range(len(daysOfIndividualId)):
        f = plt.figure()
        f.set_figheight(10)
        f.set_figwidth(30)
        day = daysOfIndividualId[i]
        y = data.loc[(data['person_id'] == person_id) & (data['day'] == day)]['work_to_number'].to_numpy()
        x = np.arange(len(y))
        plt.plot(x, y, marker="o", markersize=10, color='orange')
        name = f"id_{str(person_id)}day_{str(day)}"
        # plt.suptitle('suptitle', fontsize=14, fontweight='bold')
        for j in range(len(x)):  # labels each point
            plt.annotate(y[j], (x[j], y[j] + 0.2))
        plt.title(name)
        plt.savefig(f"matplotlib/{name}.png")


def X_plotter_2D(day, X):
    # --------------------------
    X = np.argmax(X, axis=-1)
    Tx = X.shape[1]
    # ---------------------------
    f = plt.figure()
    f.set_figheight(5)
    f.set_figwidth(30)
    plt.plot(np.arange(Tx), X[day, :], marker="o", markersize=10, color='orange')
    for i in range(Tx):  # labels each point
        plt.annotate(X[day, :][i], (np.arange(Tx)[i], X[day, :][i] + 0.2))
    plt.show()


def Y_plotter_2D(day, Y):
    # ---------------------------
    Y = np.swapaxes(Y, 0, 1)
    Y = np.argmax(Y, axis=-1)
    Tx = Y.shape[1]
    # ---------------------------
    f = plt.figure()
    f.set_figheight(5)
    f.set_figwidth(30)
    axis1 = np.arange(Tx)
    axis2 = Y[day, :]
    plt.plot(axis1, axis2, marker="o", markersize=10, color='orange')
    for i in range(Tx):  # labels each point
        plt.annotate(axis2[i], (axis1[i], axis2[i] + 0.2))
    plt.show()


def array_plotter(array):
    f = plt.figure()
    f.set_figheight(5)
    f.set_figwidth(30)
    axis1 = np.arange(len(array))
    axis2 = array
    plt.plot(axis1, axis2, marker="o", markersize=10, color='orange')
    for i in range(len(array)):  # labels each point
        plt.annotate(axis2[i], (axis1[i], axis2[i] + 0.7))
    plt.show()
