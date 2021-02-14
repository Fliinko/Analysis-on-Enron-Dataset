#!/usr/bin/env python
# coding: utf-8

# In[37]:


import csv
import re
import os
import collections
import operator
from email.parser import Parser
import json
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Email object containing the properties in the JSON file

# In[38]:


class email:
    def __init__(self, sender, recipients, body):
        self.sender = sender
        self.recipients = recipients
        self.body = body


# In[39]:



"""
# # Reading data from the dataset
"""

emails = []
email_addresses = []

rootdir = "maildirtest"
for directory, subdirectory, filenames in  os.walk(rootdir):
    for filename in filenames:

        #Reading data from file
        with open(os.path.join(directory, filename), "r") as f:
            data = f.read()

        #Creating instance of the email.parser object
        emailParser = Parser().parsestr(data)

        #reading the from section of the email
        sender = emailParser['from']

        #reading the to section of the email, which can contain multiple recipients
        if emailParser['to']:
            recipients = emailParser['to']
            recipients = "".join(recipients.split())
            recipients = recipients.split(",")
        else:
            recipients = ['None']

        if emailParser['cc']:
            cc = emailParser['cc']
            cc = "".join(cc.split())
            cc = cc.split(",")
            recipients.extend(cc)

        if emailParser['bcc']:
            bcc = emailParser['bcc']
            bcc = "".join(bcc.split())
            bcc = bcc.split(",")
            recipients.extend(bcc)

        recipients = list(set(recipients))

        #reading the body section of the email
        body = emailParser.get_payload()

        #Creating an email object and appending it to the list of all emails
        email = Email(sender, recipients, body)
        emails.append(email)

        #Adding all users (email adresses) to a list
        email_addresses.extend(sender)
        email_addresses.extend(recipients)

email_addresses = list(set(email_addresses))


# In[41]:


# Creating a dictionary of all the combined documents between two distinct people
dataset = {}
for email in emails:
    for recipient in email.recipients:
        key1 = (email.sender, recipient)
        key2 = (recipient, email.sender)
        
        if(key1 in dataset.keys()):
            dataset[email.sender, recipient] += 1
        elif (key2 in dataset.keys()):
            dataset[recipient, email.sender] += 1
        else:
            dataset[email.sender, recipient] = 1
            
print(dataset)


# Creating a Directed Graph using the edges found. 
# A directed graph entails the nature of the email; ie. the sender and the recipients

# In[42]:


G = nx.DiGraph()

for k,v in dataset:
    G.add_edge(k,v, weight=dataset[(k,v)])


# In Degree - Getting the number of vertices coming in to the node, from other connected nodes
# Out Degree - Getting the number of vertices going out of the node, to other connected nodes

# In[43]:


inDegree = G.in_degree()
outDegree = G.out_degree()


# Degree Distribution is the probability distribution of the in and out degrees over the whole network

# In[44]:


plt.title("Degree Histogram")
plt.xlabel("Count")
plt.ylabel("Degree")

degrees = [G.degree(n) for n in G.nodes()]

plt.hist(degrees, log='true')
plt.show()


# Scatter plot of the degree distribution

# In[45]:


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

# In[46]:


def Diameter(emails):
    root = random.choice(emails)
    email, distance = find_furthest_node(root, emails)
    _, distance = find_furthest_node(email, emails)
    
    return distance


# Creates a subgraph of the largest weakly connected component

# In[47]:


largest_cc = max(nx.weakly_connected_components(G), key=len)
SG = G.subgraph(largest_cc)


# Average Path Length is the sum of the path lengths between all pairs of nodes normalized by n*(n-1) where n is the number of nodes in the Graph (G).
# Since it can only be executed on a connected graph, the AVP has to be calculated on the largest subgraph

# In[48]:


#Average Path Length
averagePathLength = nx.average_shortest_path_length(SG)


# The Global Clustering Coefficient is the number of closed trianges over the total number of open or closed triplets.
# A triplet contain three nodes with 2 edges. A triangle contain three closed overlapping triplets, one centered on each of the nodes

# In[49]:


#Global Clustering Coefficient
clusteringCoefficient = nx.average_clustering(G)


# Printing Graph Analytics 

# In[50]:


print("Average Shortest Path Length: ", averagePathLength, "\n")
print("Clustering Coefficient: ", clusteringCoefficient, "\n")


# Sorting the Dictionary according to value in descending order

# In[51]:


def sortDiscDesc(d):
    sorted_d = dict(sorted(d.items(), key=operator.itemgetter(1), reverse = True))
    return sorted_d


# Betweenness Centrality of a node is the sum of the fraction of all pairs shortest paths that pass through the node in question

# In[55]:


#Betweeness Centrality
betweennessCentrality = nx.betweenness_centrality(G, normalized=True)
betweennessCentrality = sortDiscDesc(betweennessCentrality)

print(betweennessCentrality)


# Page Rank determines the popularity of a node in the whole dataset, implying how important the node is in the dataset. Therefore, this implies the more active senders and receivers of emails in the dataset

# In[56]:


#Page Rank
pr = nx.pagerank(G, 0.4)
print(pr)


# Force Directed Graph using NetworkX on the JSON File

# In[66]:


weights = [G[u][v]['weight'] for u,v in dataset]

node_sizes = []
for n in SG.nodes():
    node_sizes.append(betweennessCentrality[n])

nx.draw_kamada_kawai(SG, node_size=node_sizes, with_labels=True, font_size=8, width=weights)
plt.show()


# In[ ]:




