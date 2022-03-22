import networkx as nx
import numpy as np
from scipy.sparse import bmat
import math
import matplotlib.pyplot as plt
import time
from functools import partial
import sys
# from project.py
from numpy.random import binomial
import heapq
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from matplotlib.animation import FuncAnimation
from assembly_functions import *
#Default Params
n = 10000
k = 100
p = 0.00
beta = 0.00
T = 12
interval=200
func='gaussian'
sigma=0.01
weight=0.0005
def update(t):
    print(t)
    
    # Get x and y of support at time t
    x = [hst[t][i][0] for i in range(len(hst[t]))]
    y = [hst[t][i][1] for i in range(len(hst[t]))]
    #Clear and re-plot
    ax.clear()
    points, = ax.plot(x, y, 'x')
    plt.ylim(0,1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    plt.xlim(0,1)
    ax.set_title('Hidden variable of winners at time t = %d'%t, fontsize=20)
    return points, ax

if __name__ == '__main__':
    save_gif=False
    name=''
    for i in range(len(sys.argv)):
        if sys.argv[i]=='-save' or sys.argv[i]=='-s':
            name=sys.argv[i+1]
            if not name.endswith('.gif'):
                print("save name must end with .gif")
                exit(0)
            save_gif=True
        if sys.argv[i]=='-T':
            T=int(sys.argv[i+1])
        if sys.argv[i]=='-k':
            k=int(sys.argv[i+1])
        if sys.argv[i]=='-speed' or sys.argv[i]=='-i':
            interval=int(sys.argv[i+1])
        if sys.argv[i]=='-func' or sys.argv[i]=='-f':
            if sys.argv[i+1] == 'gaussian':
                func='gaussian'
            elif sys.argv[i+1] == 'dist':
                func='dist'
            elif sys.argv[i+1]=='exp':
                func='exp'
            else:
                print('supported functions: gaussian and dist')
                exit(0)
        if sys.argv[i]=='-sigma':
            sigma=float(sys.argv[i+1])
        if sys.argv[i]=='-weight' or sys.argv[i]=='-w':
            weight=float(sys.argv[i+1])
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    # Query the figure's on-screen size and DPI. Note that when saving the figure to
    # a file, we need to provide a DPI for that separately.
    print('fig size: {0} DPI, size in inches {1}'.format(
        fig.get_dpi(), fig.get_size_inches()))

    # Create the random graph and project
    if func=='gaussian':
        print("Params: n={}, k={}, T={}, sigma={}".format(n,k,T,sigma))
        G2 = create_graph(n, gen_attributes_uniform_2, partial(edge_probability_gaussian, sigma=sigma))
    elif func=='dist':
        print("Params: n={}, k={}, T={}, weight={}".format(n,k,T,weight))
        G2 = create_graph(n, gen_attributes_uniform_2, partial(edge_probability_invsquare, w=weight, order=2))
    elif func=='exp':
        print("Params: n={}, k={}, T={}, sigma={}".format(n,k,T,sigma))
        G2 = create_graph(n, gen_attributes_uniform_2, partial(edge_probability_exp, sigma=sigma))
    print("Graph edge density", get_edge_density(G2))
    #G2 = create_graph(n, gen_attributes_2, edge_probability_sharp_2)
    support, support_size_at_t, winners_at_t = project(G2, T, 0.0, random_topk=True, verbose=False, n=n, k=k, beta=beta)

    assembly = G2.subgraph(winners_at_t[T-1])
    print("Num nodes: ", len(assembly.nodes))
    print("Assembly Edge density: ", get_edge_density(assembly))
    print("Convergence time: ", np.where(np.array(support_size_at_t)==np.max(support_size_at_t))[0][0])
    print("Support size: ", np.max(support_size_at_t))

    hst = []
    #get hidden variables of support at all times
    for t in range(T):
        hs_t = [G2.nodes[n]['h'] for n in winners_at_t[t]]
        #hst.append(math.sqrt(np.var(hs_t)))
        #hst.append(max(hs_t)-min(hs_t))
        hst.append(hs_t)
    max_x = np.max([np.max(hst), 1])

    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, T), interval=interval)
    if save_gif:
        anim.save(name, dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()