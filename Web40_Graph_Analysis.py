#!/usr/bin/env python
# coding: utf-8

# In[4]:


import csv
import re
import collections
import operator
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Email object containing the properties in the JSON file

# In[5]:


class Email:
    def __init__(self, sender, recipients, body):
        self.sender = sender
        self.recipients = recipients
        self.body = body


# Using the inbuilt functions in the JSON library, the .json file was opened and a dataset was created

# In[6]:


with open("keyword_cloud_dict.json", encoding ="utf8") as fd:
    emails = []
    rd = json.load(fd)
    
    for row in rd:
        emails.append(email(row))
        
    fd.close()


# Creating a dictionary of edges. This is to set a key pair to a corresponding value. 
# We can use the sender of the email and the recipients as the key with the number of times sent as the value.

# In[7]:


edges = {}

for e in emails:
    if "@" in e.sender:
        sending = re.findall("@(/w+)", e.receiver)
        for s in sending:
            key = tuple([t.sender, s])
            if key in edges.keys():
                edges[key] += 1
            else:
                edges[key] = 1


# Creating a Directed Graph using the edges found. 
# A directed graph entails the nature of the email; ie. the sender and the recipients

# In[8]:


G = nx.DiGraph()

for k,v in edges:
    G.add_edge(k,v, weight=edges[(k,v)])


# In Degree - Getting the number of vertices coming in to the node, from other connected nodes
# Out Degree - Getting the number of vertices going out of the node, to other connected nodes

# In[9]:


inDegree = G.in_degree()
outDegree = G.out_degree()


# Degree Distribution is the probability distribution of the in and out degrees over the whole network

# In[10]:


plt.title("Degree Histogram")
plt.xlabel("Count")
plt.ylabel("Degree")

degrees = [G.degree(n) for n in G.nodes()]

plt.hist(degrees, log='true')
plt.show()


# Scatter plot of the degree distribution

# In[11]:


plt.title("Degree Scatter")
plt.xlabel("Rank")
plt.ylabel("Degree")

#creates a distribution of degrees
degree=nx.degree_histogram(G)
degree.sort(reverse=True)

#Generate normalised values for y according to degree
y=[z/float(sum(degree))for z in degree]

#Generate X axis sequence, from 1 to maximum degree
x=range(len(degree))
plt.scatter(x,y,s=1,color=(1,0,0))

plt.show()


# The Graph Diameter is the length of the longest shortest path between any two graph vertices.
# For arbitrary graphs, we need to compute the shortest path between any two vertices and take the length of the greatest of these paths.
# The algorithm starts by calculating the diameter of a graph made of variables and relations. A random node is picked in the tree and a breath first search is used to find the furthest node in the graph.

# In[12]:


def Diameter(emails):
    root = random.choice(emails)
    email, distance = find_furthest_node(root, emails)
    _, distance = find_furthest_node(email, emails)
    
    return distance


# Creates a subgraph of the largest weakly connected component

# In[13]:


largest_cc = max(nx.weakly_connected_components(G), key=len)
SG = G.subgraph(largest_cc)


# Average Path Length is the sum of the path lengths between all pairs of nodes normalized by n*(n-1) where n is the number of nodes in the Graph (G).
# Since it can only be executed on a connected graph, the AVP has to be calculated on the largest subgraph

# In[14]:


#Average Path Length
averagePathLength = nx.average_shortest_path_length(SG)


# The Global Clustering Coefficient is the number of closed trianges over the total number of open or closed triplets.
# A triplet contain three nodes with 2 edges. A triangle contain three closed overlapping triplets, one centered on each of the nodes

# In[15]:


#Global Clustering Coefficient
clusteringCoefficient = nx.average_clustering(G)


# Printing Graph Analytics 

# In[16]:


print("Average Shortest Path Length: ", averagePathLength, "\n")
print("Clustering Coefficient: ", clusteringCoefficient, "\n")


# Sorting the Dictionary according to value in descending order

# In[17]:


def sortDiscDesc(d):
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse = True))
    return sorted_d


# Betweenness Centrality of a node is the sum of the fraction of all pairs shortest paths that pass through the node in question

# In[18]:


#Betweeness Centrality
betweennessCentrality = nx.betweenness_centrality(G, normalized=True)
betweennessCentrality = sortDiscDesc(betweennessCentrality)


# Page Rank determines the popularity of a node in the whole dataset, implying how important the node is in the dataset. Therefore, this implies the more active senders and receivers of emails in the dataset

# In[19]:


#Page Rank
pr = nx.pagerank(G, 0.4)


# Force Directed Graph using NetworkX on the JSON File

# In[45]:


# fixing the size of the figure 
plt.figure(figsize =(10, 7)) 
  
node_color = [G.degree(v) for v in G] 
# node colour is a list of degrees of nodes 
  
node_size = [0.0005 * nx.get_node_attributes(G, 'population')[v] for v in G] 
# size of node is a list of population of cities 
  
edge_width = [0.0015 * G[u][v]['weight'] for u, v in G.edges()] 
# width of edge is a list of weight of edges 
  
nx.draw_networkx(G, node_size = node_size,  
                 node_color = node_color, alpha = 0.7, 
                 with_labels = True, width = edge_width, 
                 edge_color ='.4', cmap = plt.cm.Blues) 
  
plt.axis('off') 
plt.tight_layout(); 


# In[ ]:




