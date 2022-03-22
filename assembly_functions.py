### Move all assembly functions to the same file
### In order to standardize across files
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from functools import partial
import time 
# from project.py
from numpy.random import binomial
import heapq
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import bmat
##### Graph creation functions
def create_graph(n, attr_function, edge_prob_function):
    #Create nodes
    G = nx.DiGraph()
    attr = attr_function(n) #generate a 1xn list of attributeszz
    G.add_nodes_from([(i, {"h":attr[i]}) for i in range(n)]) #add n nodes with attr stored in node.h
    #Create random edges
    #split up due to memory issues
    adj_matrix = np.zeros((n,n))
    for i in range(0, n, n//10):
        for j in range(0, n, n//10): #fix end bounds for n not divisible by 10
            
            xx, yy = np.meshgrid(range(i, i+n//10), range(j, j+n//10))
            hx = attr[xx]
            hy = attr[yy]
            edge_prob = edge_prob_function(hx, hy).squeeze() #calculate pairwise edge probabilities based on attributes
            adj_matrix[j:j+n//10, i:i+n//10] = edge_prob > np.random.rand(n//10,n//10)
    adj_matrix[np.identity(n, dtype=bool)] = False #Remove self-edges
    #add random edges to graph
    G_edges = nx.from_numpy_array(adj_matrix, parallel_edges=False, create_using=nx.DiGraph)
    G.add_edges_from(G_edges.edges, weight=1) #Separate so it keeps node attributes
    return G
def create_ROC_graph(n, d, s, q):
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for i in range(d):
        subset = np.random.choice(range(n), size=s)
        community = nx.gnp_random_graph(s, q, directed=True)
        community = nx.relabel_nodes(community, {i:subset[i] for i in range(s)})
        G.add_edges_from(community.edges(data=True))
        for node in subset:
            G.nodes[node]['community']=i
        for edge in community.edges:
            G.edges[edge]['weight']=1
    return G
#Create an adjacency matrix with edge connections between G1 and G2
#For each node in G1, connect to each node in G2 with a certain probability
def create_bipartite_graph(G1, G2, edge_prob_function):
    attr_1 = np.array([x[1] for x in list(G1.nodes(data='h'))])
    attr_2 = np.array([x[1] for x in list(G2.nodes(data='h'))]) #retrieve all node attributes from graph
    #Create random edges
    xx, yy = np.meshgrid(range(len(attr_1)), range((len(attr_2))))
    hx = attr_1[xx]
    hy = attr_2[yy]
    edge_prob = edge_prob_function(hx, hy) #calculate pairwise edge probabilities based on attributes
    adj_matrix = edge_prob > np.random.rand(edge_prob.shape[0], edge_prob.shape[1])
    #add edges using sparse graph
    biadj_matrix=bmat([[None, adj_matrix], [None, None]])
    U = nx.reverse(from_biadjacency_matrix(biadj_matrix, create_using=nx.DiGraph()))
    #relabel G1 nodes with k+number
    U = nx.relabel_nodes(U, {k+1000:'k'+str(k) for k in range(100)})
    G1 = nx.relabel_nodes(G1, lambda x: 'k'+str(x))
    #Transfer node attributes from G1 and G2
    U.add_nodes_from(G1.nodes(data=True))
    U.add_nodes_from(G2.nodes(data=True))
    return U

#Project 
#A: the graph to project onto 
#T: the number of rounds
#p: the probability of connection to the initial stimulus, or the n-dimensional stimulus vector
#random_topk: whether to choose random values in the case of ties(not repeatable)
def project(A, T, p, random_topk=True, verbose=False, n=10000, beta=0.00, k=100):
#     if plot_cap:
#         dims=len(A.nodes[0]['h'])
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         fig.show()
#         fig.canvas.draw()
    winners = []
    support = set()
    support_size_at_t = []
    winners_at_t = []
    new_winners_at_t = []
    if np.isscalar(p):
        stimulus_inputs = np.random.binomial(k, p, size=n) # random binomial inputs
    else:
        if len(p) != n:
            print("Stimulus length must be {}".format(n))
            return
        stimulus_inputs = p
    for t in range(T):
        #Add inputs from outside stimulus
        inputs = stimulus_inputs.copy()
        #Add inputs from previous winners
        for w in winners:
            for e in A.edges(w, data=True): #iterate over all edges from w
                inputs[e[1]] += e[2]['weight'] #add weight of this edge
        if random_topk:
            #identify top k winners with randomization
            shuffle_idx = np.random.permutation(n) #generate random permutation of list
            inputs_shuffled = inputs[shuffle_idx] 
            new_winners_shuffled = heapq.nlargest(k, range(len(inputs_shuffled)), inputs_shuffled.__getitem__)
            new_winners = shuffle_idx[new_winners_shuffled]#same winners except in case of ties
        else:
            #identify top k winners
            new_winners = heapq.nlargest(k, range(len(inputs)), inputs.__getitem__)
        
        if verbose:
            max_k = np.max([inputs[i] for i in new_winners])
            min_k = np.min([inputs[i] for i in new_winners])
            print("Time {}:, Max weight: {}, Min weight: {}".format(t, max_k, min_k))
        for i in new_winners:
            stimulus_inputs[i] *= (1+beta) #strengthen stimulus inputs to new winners
        # plasticity: for winners, for previous winners, update edge weight
        for i in winners:
            for j in new_winners:
                if j in A[i]:
                    A[i][j]['weight']*=(1+beta)
        # update winners
        for i in new_winners:
            support.add(i) #add all winners to the 
        winners = new_winners
        support_size_at_t.append(len(support))
        winners_at_t.append(winners)
        if t > 0:
            new_winners_at_t.append(support_size_at_t[-1]-support_size_at_t[-2])
#         if plot_cap:
#             if t>0:
#                 cb.remove()
#             hst=[A.nodes[n]['h'] for n in winners]
#             s_inputs = [inputs[n] for n in winners] #proportional
# #             min_k = np.min([inputs[i] for i in new_winners])
# #             s_inputs = [inputs[n]==min_k for n in new_winners] #plot tie values
#             ax.clear()
#             #plot 1d hidden var on line
#             if dims==1:
#                 im= ax.scatter(hst[t], np.zeros_like(hst[t]), 'x',c=s_inputs)
#             #plot 2d hidden var in box
#             else:
#                 x = [hst[i][0] for i in range(len(hst))]
#                 y = [hst[i][1] for i in range(len(hst))]
#                 im = ax.scatter(x, y, c=s_inputs)
#                 plt.ylim(0,1)
#                 plt.ylabel('h2')
#             plt.xlim(0,1)
#             plt.ylim(0, 1)
#             cb = fig.colorbar(im, ax=ax)
#             plt.xlabel('h1')
#             ax.set_title('Hidden variable of winners at time t = %d'%t)
#             fig.canvas.draw()
    return support, support_size_at_t, winners_at_t


#Functions for generating hidden attributes
def gen_attributes(n): #h is random value between 0 and 1
    return np.expand_dims(np.random.rand(n), 1)
def gen_attributes_2(n): #h is tuple of two random values (r1, r2) between (0,0) and (1,1)
    return np.random.rand(n, 2)
def gen_attributes_uniform(n):
    return np.expand_dims(np.array([k/n for k in range(n)]), 1)
def gen_attributes_uniform_2(n):
    sn = int(math.sqrt(n))
    return np.array([(x/sn,y/sn) for x in range(sn) for y in range(sn)])

def gen_attributes_powerlaw(n, beta):
    return gen_attributes(n)**(-beta)
def gen_attributes_gaussian(n):
    return np.random.normal(size=n)


### Edge probability functions
###Probabilities below assume h comes from Uniform(0,1)

#Edge probability drops off sharply at given distance between h1 and h2
#for 1d only, because in 2d case the modifier is different
def edge_probability_sharp(h1, h2, dist = 0.1):
    #Pr(|h1-h2|<c) = c*(2-c) for 0<c<1
    #1/36 if h1, h2 less than 0.2 apart. 0 otherwise
    modifier = dist*(2-dist)
    return p*np.less(abs(h1-h2), dist)/modifier
def edge_constant(h1, h2, dist=0.005):
    return np.less(np.linalg.norm(h1-h2, axis=2), dist)
#Edge probability drops off proportional to 1-dist(h1, h2)
#For 1d only, because in 2d case max distance is different
def edge_probability_graded(h1, h2):
    return 1.5*p*(1-abs(h1-h2))
#Uniform edge probability p
def edge_probability_er(h1, h2, p = 0.05):
    return np.full((h1.shape[0], h1.shape[1]), p)
#Edge probability drops off proportional to 1/distance(h1, h2)
def edge_probability_decay(h1, h2, dist=0.1, modifier = 0.01):
    return modifier/(np.linalg.norm(h1-h2, axis=h1.ndim-1)+dist)

#Edge probability drops off sharply at a given distance
#Note that exact modifier is somewhat difficult to compute
#https://math.stackexchange.com/questions/1294800/average-distance-between-two-randomly-chosen-points-in-unit-square-without-calc
def edge_probability_sharp_2(h1, h2):
    dist = np.linalg.norm(h1-h2, axis=2)
    return 0.095 * np.less(dist, 0.2) 

#Edge probability drops off according to w*e^-|dist(x,y)|^2/2c^2
def edge_probability_gaussian(h1, h2, sigma=0.1, w=1):
    dist = np.linalg.norm(h1-h2, axis=h1.ndim-1)**2 #squared
    prob = w*np.exp(-1*dist/(2*sigma**2))
    return prob 
#edge probability drops off according to e^-|x-y|/c
def edge_probability_exp(h1, h2, sigma=0.01):
    dist=np.linalg.norm((h1-h2), ord=1, axis=h1.ndim-1)
    prob=np.exp(-1*dist/sigma)
    return prob
def edge_probability_invsquare(h1, h2, w=0.0005, order=2):
    dist=np.linalg.norm(h1-h2, axis=h1.ndim-1, ord=order)**2
    return w/(1+dist)


#Using the method described in Barthelemy 2011
#Should result in the creation of "hub neurons" with higher fitness
def gen_attributes_exp(n):
    return np.random.exponential(size=n)
def edge_probability_heavyside(h1, h2):
    return np.greater(h1+h2-6.5, 0)

###Helper/Visualization Functions
def draw_support(G2, winners_at_t, T, var_range=[(0,1), (0, 1)]):
    hst = []
    for t in range(T):
        hs_t = [G2.nodes[n]['h'] for n in winners_at_t[t]]
        #hst.append(math.sqrt(np.var(hs_t)))
        #hst.append(max(hs_t)-min(hs_t))
        hst.append(hs_t)
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.show()
    fig.canvas.draw()
    max_x = np.max([np.max(hst), 1])
    for t in range(T):
        
        ax.clear()
        #plot 1d hidden var on line
        if hst[0][0].size==1:
            ax.plot(hst[t], np.zeros_like(hst[t]), 'x')
        #plot 2d hidden var in box
        else:
            x = [hst[t][i][0] for i in range(len(hst[t]))]
            y = [hst[t][i][1] for i in range(len(hst[t]))]
            ax.plot(x, y, 'x')
            plt.ylim(var_range[1][0], var_range[1][1])
            plt.ylabel('h2')
        plt.xlim(var_range[0][0],var_range[0][1])
        plt.xlabel('h1')
        ax.set_title('Hidden variable of winners at time t = %d'%t)
        time.sleep(0.5)
        fig.canvas.draw()
#returns edge density of digraph with no self-edges possible
def get_edge_density(graph):
    nnodes = len(graph.nodes)
    return len(graph.edges)/(nnodes*(nnodes-1))