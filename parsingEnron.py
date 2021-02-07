#Libraries for File-Reading and Pre-Processing
import os
from email.parser import Parser
import nltk
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#Libraries for TF-IDF, Cosine Similarity and K-means
import math
import numpy as np
from collections import Counter 


#Object Email
class Email:
    def __init__(self, sender, recipients, body):
        self.sender = sender
        self.recipients = recipients
        self.body = body

    def __str__(self):
        return "From: %s \nTo: %s \n\n%s" %(self.sender, self.recipients, self.body)


emails = []
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

        #reading the body section of the email
        body = emailParser.get_payload()

        #Creating an email object and appending it to the list of all emails
        email = Email(sender, recipients, body)
        emails.append(email)


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


# # Pre-Processing

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
for data in dataset:
    dataset[data] = tokenize(dataset[data])
    dataset[data] = change_case(dataset[data])
    dataset[data] = remove_stop_words(dataset[data])
    dataset[data] = stemming(dataset[data])
    dataset[data] = remove_symbols(dataset[data])

def printDatasetData():
    counter = 0
    for data in dataset:
        counter += 1
        print("\n\n\n")
        print(dataset[data])
        if(counter == 2):
            break

def printDatasetKeys():
    counter = 0
    for data in dataset:
        counter += 1
        print("\n\n\n")
        print(data)
        if(counter == 2):
            break

printDatasetData()


# # Calculating TFIDF

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
#print(unique_words)

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

counter = 0
for i in range(0, len(tf_idf)):
    counter += 1
    print("\n")
    print(list(tf_idf.keys())[i])
    print(list(tf_idf.values())[i])
    if(counter == 50):
        break
