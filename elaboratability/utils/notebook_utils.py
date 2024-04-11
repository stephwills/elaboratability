import json
import numpy as np

def dump_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)

def load_json(fname):
    with open(fname, "r") as f:
        data = json.load(f)
    return data

def disable_rdlogger():
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')


def plot(values, xaxis=None, yaxis='Frequency', title=None, bins=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,3))
    plt.hist(values, bins=bins)

    if xaxis:
        plt.xlabel(xaxis, fontsize=20)
    if yaxis:
        plt.ylabel(yaxis, fontsize=20)
    if title:
        plt.title(title)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    return plt.show()


def save_npy(arr, fname):
    with open(fname, 'wb') as f:
        np.save(f, arr)


def plot_3d(coordinates, colour='blue', alpha=1,
            xlim=[-10,10],
            ylim=[-10,10],
            zlim=[-10,10],):
    import matplotlib.pyplot as plt
    print(coordinates.shape)
    X = coordinates[:,0]
    Y = coordinates[:,1]
    Z = coordinates[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=colour, marker='o', alpha=alpha)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    return plt.show()





def get_bool_intersect(lst1, lst2):
    return [a and b for a, b in zip(lst1, lst2)]