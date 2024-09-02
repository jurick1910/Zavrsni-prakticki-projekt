import numpy as np
import matplotlib.pyplot as plt
import scipy 
import time
import sklearn
import sklearn.manifold
import scipy.io
from . import data_files
import random
import gurobipy as gp
from gurobipy import Model, GRB

def get_markers(data, labels, num_markers, method='pairwise', epsilon=1, sampling_rate=1, n_neighbors=3, verbose=True):
    """marker selection algorithm
    data: Nxd numpy array with point coordinates, N: number of points, d: dimension
    labels: list with labels (N labels, one per point)
    num_markers: target number of markers to select. num_markers<d
    method: 'centers', 'pairwise', or 'pairwise_centers'
    epsilon: constraints will be of the form expr>Delta, where Delta is chosen to be epsilon times the norm of the smallest constraint (default 1)
    (This is the most important parameter in this problem, it determines the scale of the constraints, 
    the rest the rest of the parameters only determine the size of the LP)
    sampling_rate: (if method=='pairwise') selects constraints from a random sample of proportion sampling_rate (default 1)
    n_neighbors: (if method=='pairwise') chooses the constraints from n_neighbors nearest neighbors (default 3)
    """
    d = data.shape[1]
    t = time.time()
    samples, samples_labels, idx = __sample(data, labels, sampling_rate)

    if method == 'pairwise':
        constraints, smallest_norm = __select_constraints_pairwise(
            data, labels, samples, samples_labels, n_neighbors)
    
    random_max_constraints = random.randint(1, constraints.shape[0])


    num_cons = constraints.shape[0]
    if num_cons > random_max_constraints:
        p = np.random.permutation(num_cons)[0:random_max_constraints]
        constraints = constraints[p, :]
    if verbose:
        print('Solving a linear program with {} variables and {} constraints'.format(
            constraints.shape[1], constraints.shape[0]))
    sol = __lp_markers(constraints, num_markers, smallest_norm * epsilon)
    if verbose:
        print('Time elapsed: {} seconds'.format(time.time() - t))
    x = sol['x'][0:d]
    markers = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[
        : num_markers]
    return markers


def __sample(data, labels, sampling_rate):
    """subsample data"""
    indices = []
    for i in set(labels):
        idxs = [x for x in range(len(labels)) if labels[x] == i]
        n = len(idxs)
        s = int(np.ceil(len(idxs) * sampling_rate))
        aux = np.random.permutation(n)[0:s]
        indices += [idxs[x] for x in aux]
    return [data[i] for i in indices], [labels[i] for i in indices], indices


def __select_constraints_pairwise(data, labels, samples, samples_labels, n_neighbors):
    """select constraints of the form x-y where x,y have different labels"""
    constraints = []
    # nearest neighbors are selected from the entire set
    neighbors = {}
    data_by_label = {}
    smallest_norm = np.inf
    for i in set(labels):
        X = [data[x, :] for x in range(len(labels)) if labels[x] == i]
        data_by_label[i] = X
        neighbors[i] = sklearn.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors).fit(np.array(X))
    # compute nearest neighbor for samples
    for i in neighbors.keys():
        Y = [samples[x]
             for x in range(len(samples_labels)) if samples_labels[x] == i]
        for j in neighbors.keys():
            if i != j:
                idx = neighbors[j].kneighbors(Y)[1]
                for s in range(len(Y)):
                    for t in idx[s]:
                        v = Y[s] - data_by_label[j][t]
                        constraints += [v]
                        if np.linalg.norm(v) ** 2 < smallest_norm:
                            smallest_norm = np.linalg.norm(v) ** 2
    constraints = np.array(constraints)
    return -constraints * constraints, smallest_norm


def __lp_markers(constraints, num_markers, epsilon):
    m, d = constraints.shape
    c = np.concatenate((np.zeros(d), np.ones(m)))
    l = np.zeros(d + m)
    u = np.concatenate((np.ones(d), np.array([float('inf') for i in range(m)])))
    aux1 = np.concatenate((constraints, -np.identity(m)), axis=1)
    aux2 = np.concatenate((np.ones((1, d)), np.zeros((1, m))), axis=1)
    A = np.concatenate((aux1, aux2), axis=0)
    b = np.concatenate((-epsilon * np.ones(m), np.array([num_markers])))
    
    model = gp.Model()
    x = model.addVars(d + m, lb = l, ub = u, name = "x")
    model.addConstrs((gp.quicksum(A[i, j] * x[j] for j in range(d + m)) == b[i] for i in range(m + 1)), name = "constraints")
    model.setObjective(gp.quicksum(c[i] * x[i] for i in range(d + m)), gp.GRB.MINIMIZE)
    model.optimize()
    return {"x": [x[i].x for i in range(d + m)]}


def circles_example(N=30, d=5):
    num_markers = 2
    X = np.concatenate((np.array([[np.sin(2 * np.pi * i / N), np.cos(2 * np.pi * i / N)] for i in range(N)]),
                        np.random.random((N, d - 2))), axis=1)
    Y = np.concatenate((np.array([[2 * np.sin(2 * np.pi * i / N), 2 * np.cos(2 * np.pi * i / N)] for i in range(N)]),
                        np.random.random((N, d - 2))), axis=1)
    data = np.concatenate((X, Y), axis=0)
    labels = np.concatenate((np.zeros(10), np.ones(10)))
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(data[0:N, 0], data[0:N, 1], data[0:N, 2], c='r', marker='o')
    ax.scatter(data[N + 1:2 * N, 0], data[N + 1:2 * N, 1],
               data[N + 1:2 * N, 2], c='g', marker='x')
    plt.show()
    sol = get_markers(data, labels, num_markers, 1, 3, 10)
    x = sol['x'][0:d]
    markers = sorted(range(len(x)), key=lambda i: x[i], reverse=True)[
        :num_markers]
    for i in range(d):
        if i not in markers:
            data[:, i] = np.zeros(2 * N)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(data[0:N, 0], data[0:N, 1], data[0:N, 2], c='r', marker='o')
    ax2.scatter(data[N + 1:2 * N, 0], data[N + 1:2 * N, 1],
                data[N + 1:2 * N, 2], c='g', marker='x')
    plt.show()


def plot_marker_selection(data, markers, names, perplexity=40):
    print('Computing TSNE embedding')
    # code fix to deal with exceptions if there is a particular cell class type with n < 40
    # automatically re-scales perplexity in these cases
    if len(data) < 40:
        perplexity = len(data) - 1
    else:
        perplexity = 40
    t = time.time()
    X_original = sklearn.manifold.TSNE(
        n_components=2, perplexity=perplexity).fit_transform(data)
    X_embedded = sklearn.manifold.TSNE(n_components=2, perplexity=perplexity).fit_transform(
        data[:, markers])
    print('Elapsed time: {} seconds'.format(time.time() - t))
    cmap = plt.cm.jet
    unique_names = list(set(names))
    num_labels = len(unique_names)
    colors = [cmap(int(i * 256 / num_labels)) for i in range(num_labels)]
    aux = [colors[unique_names.index(name)] for name in names]

    fig = plt.figure()
    ax = fig.add_subplot(121)
    for g in unique_names:
        i = [s for s in range(len(names)) if names[s] == g]
        ax.scatter(X_original[i, 0], X_original[i, 1],
                   c=[aux[i[0]]], s=5, label=names[i[0]])
    ax.set_title('Original data')
    ax2 = fig.add_subplot(122)
    for g in np.unique(names):
        i = [s for s in range(len(names)) if names[s] == g]
        ax2.scatter(X_embedded[i, 0], X_embedded[i, 1],
                    c=[aux[i[0]]], s=5, label=names[i[0]])
    ax2.set_title('{} markers'.format(len(markers)))
    plt.legend(bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.7)
    return fig


def one_vs_all_selection(data, labels, num_bins=20):
    data_by_label = {}
    unique_labels = list(set(labels))
    number_classes = len(unique_labels)
    [N, d] = data.shape
    for lab in unique_labels:
        X = [data[x, :] for x in range(len(labels)) if labels[x] == lab]
        data_by_label[lab] = X
    markers = [None for i in range(number_classes)]
    bins = data.max() / num_bins * range(num_bins + 1)
    for idx in range(number_classes):
        c = unique_labels[idx]
        current_class = np.array(data_by_label[c])
        others = np.concatenate([data_by_label[lab]
                                 for lab in unique_labels if lab != c])
        big_dist = 0
        for gene in range(d):
            if gene not in markers[0:idx]:
                [h1, b1] = np.histogram(current_class[:, gene], bins)
                h1 = np.array(h1).reshape(1, -1) / current_class.shape[0]
                [h2, b2] = np.histogram(others[:, gene], bins)
                h2 = np.array(h2).reshape(1, -1) / others.shape[0]
                dist = -sklearn.metrics.pairwise.additive_chi2_kernel(h1, h2)
                if dist > big_dist:
                    markers[idx] = gene
                    big_dist = dist
    return markers


def optimize_epsilon(data_train, labels_train, data_test, labels_test, num_markers, method='centers', fixed_parameters={}, bounds=[(0.2 , 10)], x0=[1], max_fun_evaluations=20, n_experiments=5, clf=None, hierarchy=False, verbose=True):
    """
    Finds the optimal value of epsilon using scipy.optimize.dual_annealing
    """
    if clf==None:
        clf=sklearn.neighbors.NearestCentroid()
    Instance=__ScGeneInstance(data_train, labels_train, data_test, labels_test, clf, num_markers, method, fixed_parameters, n_experiments)
    print('Optimizing epsilon for', num_markers, 'markers and', method, 'method.')
    res = scipy.optimize.dual_annealing(Instance.error_epsilon, bounds=bounds, x0=x0,  maxfun=max_fun_evaluations, no_local_search=True)
    return [res.x, 1-res.fun]    

class __ScGeneInstance:
    def __init__(self, X_train, y_train, X_test, y_test, clf, num_markers, method, fixed_parameters, n_experiments):
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        self.clf=clf
        self.num_markers=num_markers
        self.method=method
        self.fixed_parameters=fixed_parameters
        self.n_experiments=n_experiments

    def error_epsilon(self, epsilon):
        return 1-self.accuracy(epsilon)

    def accuracy(self, epsilon):
        #compute avg over n_experiments random samples for stability

        markers=[get_markers(self.X_train, self.y_train, self.num_markers, self.method, epsilon=epsilon, verbose=False, **self.fixed_parameters) for i in range(self.n_experiments)]
        val=[self.performance( markers[i] ) for i in range(self.n_experiments)]
        return np.mean(val)
    
    def performance(self, markers):
        self.clf.fit(self.X_train[:,markers], self.y_train)
        return self.clf.score(self.X_test[:,markers], self.y_test)

def load_example_data(name):
    if name=="CITEseq":
        a = scipy.io.loadmat(data_files.get_data("CITEseq.mat"))
        data= a['G'].T
        N,d=data.shape
        #transformation from integer entries 
        data=np.log(data+np.ones(data.shape))
        for i in range(N):
            data[i,:]=data[i,:]/np.linalg.norm(data[i,:])
        #load labels from file
        a = scipy.io.loadmat(data_files.get_data("CITEseq-labels.mat"))
        l_aux = a['labels']
        labels = np.array([i for [i] in l_aux])
        #load names from file
        a = scipy.io.loadmat(data_files.get_data("CITEseq_names.mat"))
        names=[a['citeseq_names'][i][0][0] for i in range(N)]
        return [data, labels, names]
    else:
        print("currently available options is only 'CITEseq'")



