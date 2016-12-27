__author__ = 'ckomurlu'

import numpy as np
import networkx as nx
import cPickle as cpk
import json
import matplotlib.pyplot as plt

# dag_file_location = r'C:\Users\ckomurlu\Documents\workbench\experiments\20150911\dag3.txt'
# parentChild_file_location = r'C:\Users\ckomurlu\Documents\workbench\experiments\20150911\parentChildDicts_k2_bin10.pkl'
# dag_file_location = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151102/humidity_dag_bin5.txt'
# parentChild_file_location = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151102/' + \
#                             'humidity_parentChildDicts_k2_bin5.pkl'
dag_file_location = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/temp_humid_dag_bin5.txt'
parentChild_file_location = 'C:/Users/ckomurlu/Documents/workbench/experiments/20151103/' + \
                            'temp_humid_parentChildDicts_k2_bin5.pkl'

dagMat = np.loadtxt(dag_file_location, delimiter=',')
edgeCoords = np.where(dagMat)
edges = zip(edgeCoords[0], edgeCoords[1])
print len(edges)

G = nx.DiGraph()
G.add_nodes_from(range(100))
G.add_edges_from(edges)
# print G.node
# print G.edge
# try:
#     print nx.find_cycle(G)
# except nx.exception.NetworkXNoCycle:
#     print 'no cycles found'
# print nx.is_weakly_connected(G)
childCounts = np.zeros(shape=(100,),dtype=int)
parentCounts = np.zeros(shape=(100,),dtype=int)
parentDict = dict()
childDict = dict()
for node in G.node:
    # print node, G[node], G.predecessors(node)
    parentDict[node] = G.predecessors(node)
    childDict[node] = G.successors(node)

cpk.dump((parentDict,childDict),
         open(parentChild_file_location,'wb'))

# print 'parent dict'
# # print parentDict
# json.dumps(parentDict,indent=2,separators=(',', ': '))
# print
# print 'child dict'
# # print childDict
# json.dumps(childDict,indent=2,separators=(',', ': '))

    # childCounts[node] = len(G[node])
#     parentCounts[node] = len(G.predecessors(node))
# print np.histogram(childCounts,bins=[0,1,2,3,4,5])
# print np.histogram(parentCounts,bins=[0,1,2,3,4,5])
# print nx.is_weakly_connected(G)
# labels = dict()
# for i in range(50):
#     labels[i] = str(i)
# nx.draw(G,labels=labels)
# plt.show()
