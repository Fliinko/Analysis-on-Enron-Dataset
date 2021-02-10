#Libraries for File-Reading and Pre-Processing
import os
from email.parser import Parser
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Libraries for TF-IDF, Cosine Similarity and K-means
import math
import numpy as np
from collections import Counter
import random
import json


#Object Email
class Email:
    def __init__(self, sender, recipients, body):
        self.sender = sender
        self.recipients = recipients
        self.body = body

    def __str__(self):
        return "From: %s \nTo: %s \n\n%s" %(self.sender, self.recipients, self.body)

"""
# # Reading data from the dataset
"""

emails = []
email_addresses = []

rootdir = "maildir"
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

# Creating a dictionary of all the combined documents between two distinct people
dataset = {}
for email in emails:
    for recipient in email.recipients:
        key1 = (email.sender, recipient)
        key2 = (recipient, email.sender)
        if(key1 in dataset.keys()):
            dataset[email.sender, recipient] += email.body
        elif (key2 in dataset.keys()):
            dataset[recipient, email.sender] += email.body
        else:
            dataset[email.sender, recipient] = email.body



"""
# # Pre-processing
"""
# Tokenizing the Data
def tokenize(data):
    
    tokens = nltk.word_tokenize(data)

    return tokens


# Changing all the words to lowercase
def change_case(data):

    for i in range(len(data)):
        data[i] = data[i].casefold()

    return data


# Removing all 'extra' words, such as “the”, “is” and “and”. These do not give us any meaning and so can be ignored.
def remove_stop_words(data):
    
    stop_words = set(stopwords.words('english'))

    filtered_list = []

    for word in data: 
        if word not in stop_words:
            filtered_list.append(word)

    return filtered_list


# Stemming reduces the word to its essence, rather than grammatical correctnes. For example "waiting" and "waits" would both be reduced to "wait".
def stemming(data):
    
    ps = PorterStemmer() 

    for i in range(len(data)):
        data[i] = ps.stem(data[i])

    return data


# Removing Symbols
def remove_symbols(data):
    
    temp = []

    symbols = "“”‘’!\"#$€%&()*'+-,./:;<=>?@[\]^_`{|}~\n"
    for t in data:
        if t not in symbols:
            if t != '--':
                temp.append(t)
            
    return temp

# # Pre-processing the dataset

#does not pay attention to email replies. Therfore it will include "original"
#also does not take into consideration the "None" emails
for data in dataset:
    dataset[data] = tokenize(dataset[data])
    dataset[data] = change_case(dataset[data])
    dataset[data] = remove_stop_words(dataset[data])
    dataset[data] = stemming(dataset[data])
    dataset[data] = remove_symbols(dataset[data])


"""
# # Calculating TFIDF
"""

# Returns a list of all the unique words in a dictionary
def get_doc_words(dfDict):
    doc_words = [x for x in dfDict]
    return doc_words


#Calculate DF
def calculate_DF(dataset):
    
    dfDict = {}

    for i in range(len(dataset)):
        tokens = dataset[i]
        for w in tokens:
            try:
                dfDict[w].add(i)
            except:
                dfDict[w] = {i}

    for i in dfDict:
        dfDict[i] = len(dfDict[i])

    return dfDict


#returning all unique words in the dataset
def get_unique_words(dataset):
    
    unique_terms = []
    
    for f in dataset:
        words = get_doc_words(f)
        unique_terms.extend(words)
    
    unique_terms = set(unique_terms)
    return unique_terms



list_of_documents = list(dataset.values())
list_of_keys = list(dataset.keys())

unique_words = list(get_unique_words(list_of_documents))

N = len(dataset)
df_dict = calculate_DF(list_of_documents)
    
tf_idf = {}

for i in range(N):
        
    tokens = list_of_documents[i]
    counter = Counter(tokens)
        
    for token in np.unique(tokens):

        tf = counter[token]/len(tokens)
        df = df_dict[token]
        idf = np.log(N/(df))
        tf_idf[list_of_keys[i], token] = tf*idf

""" 
# Printing tfidf values for all terms in all documents
counter = 0
for i in range(0, len(tf_idf)):
    counter += 1
    print("\n")
    print(list(tf_idf.keys())[i])
    print(list(tf_idf.values())[i])
    if(counter == 50):
        break
"""


""" 
Get all documents which user X participates in (recieves or sends)
Get all the TF.IDF values of said documents (tfidf of each term in each doc user X is found), and store their average
Link the average as the TF.IDF associated with User X
Repeat for all users
"""

user_tfidf = {}

number_of_words = 20
keyword_cloud = {}

counter = 0
for user in email_addresses:
    counter += 1

    # Get dictionary of all words a particular user has in all their documents
    dict_of_user = {k:v for k,v in tf_idf.items() if (k[0][0]==user or k[0][1]==user)}

    # Calculating the average values for each user
    list_of_values = dict_of_user.values()
    if (len(list_of_values) > 0):
        avg_value = sum(list_of_values)/len(list_of_values)
    else:
        avg_value = 0

    user_tfidf[user] = avg_value

    # Getting word cloud
    user_cloud = sorted(dict_of_user.items(), key=lambda x:x[1], reverse=True)
    
    temp_list = []

    if len(user_cloud) < number_of_words:
        n = len(user_cloud)
    else:
        n = number_of_words

    for i in range(0,n):
        temp_list.append(user_cloud[i])
            

    keyword_cloud[user] = temp_list


# Changing the dictionary to Vectors
items = list(user_tfidf.items())
user_Vectors = np.array(items)
#print(user_Vectors)


""" 
Use the correspondent vectors to cluster the users using the k-
means algorithm. The choice of k is up to you. Note that you
only need to do a single level of clustering, that is, no hierarchies
are being requested. The clusters need to be visualised using an
interactive bubble chart (or equivalent), and when a cluster bub-
ble is clicked, the keyword-cloud representing that cluster should
be displayed.
(need to use cosine similarity)
For visualisations and clustering we can ignore the users which had very little participation

# # K-means

# Setting random initial centroids
init_centroids = random.sample(range(0, len(user_Vectors)), 3)
print("init_centroids")
print(init_centroids)

#getting the data assigned to the random centroids
centroids = []
for i in init_centroids:
    centroids.append(user_Vectors[i])
print("centroids")
print(centroids)

# Claculating distance using cosine similarity
def get_dist(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

# Assigning each item of data to a centroid
def findClosestCentroids(ic, X):    #ic is a list of centroids, X is the np array of data
    assigned_centroid = []
    for i in X:
        distance=[]
        for j in ic:
            distance.append(get_dist(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid
"""


# # Outputting results to JSON files

#keyword cloud json
with open('keyword_cloud.json', 'w') as outfile:
    json.dump(keyword_cloud, outfile)


""" 
# prints the most active users
ranked = user_Vectors[user_Vectors[:,1].argsort()[::-1]]
print(ranked)

 """