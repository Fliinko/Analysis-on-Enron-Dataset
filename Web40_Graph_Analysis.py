#!/usr/bin/env python
# coding: utf-8

# In[3]:


import csv
import re
import collections
import operator
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# In[2]:


class email:
    def __init__(self, t):
        self.sender = t[0]
        self.subject = t[1]
        self.receiver = t[2]
        self.content = t[3]


# In[ ]:


with open("emails.json", encoding ="utf8") as fd:
    emails = []
    rd = json.load(fd)
    
    for row in rd:
        emails.append(email(row))
        
    fd.close()


# In[4]:


edges = {}

for e in emails:
    if "@" in t.receiver:
        sending = re.findall("@(/w+)", e.receiver)
        for s in sending:
            key = tuple([t.sender, s])
            if key in edges.keys():
                edges[key] += 1
            else:
                edges[key] = 1


# In[5]:


G = nx.DiGraph()

for k,v in edges:
    G.add_edge(k,v, weight=edges[(k,v)])


# In[11]:


inDegree = G.in_degree()
outDegree = G.out_degree()


# In[12]:


def Diameter(emails):
    root = random.choice(emails)
    email, distance = find_furthest_node(root, emails)
    _, distance = find_furthest_node(email, emails)
    
    return distance


# In[7]:


largest_cc = max(nx.weakly_connected_components(G), key=len)
SG = G.subgraph(largest_cc)


# In[6]:


#Average Path Length
averagePathLength = nx.average_shortest_path_length(SG)


# In[ ]:


#Global Clustering Coefficient
clusteringCoefficient = nx.average_clustering(G)


# In[9]:


def sortDiscDesc(d):
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse = True))
    return sorted_d


# In[10]:


#Betweeness Centrality
betweennessCentrality = nx.betweenness_centrality(G, normalized=True)
betweennessCentrality = sortDiscDesc(betweennessCentrality)


# In[13]:


#Page Rank
pr = nx.pagerank(G, 0.4)


# In[ ]:




