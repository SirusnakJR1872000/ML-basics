# Tokenizing a Sentence and Converting it into Word Embeddings

# import all the required files

import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize
import numpy as np

# download necessary NLTK data
nltk.download('punkt')

# Example sentence
sentence = "This is an example sentence for word embeddings."

# tokenize and convert to lower case
tokens = word_tokenize(sentence.lower())

# load pre-trained word embedding 
word_vector = api.load("glove-wiki-gigaword-50")

# convert token to word embeddings
embed_matrix = []
for token in tokens:
    if token in word_vector:
        embed_matrix.append(word_vector[token])
    else:
        embed_matrix.append(np.zeroes(50))

# convert list to numpy array
embed_matrix = np.array(embed_matrix)

print("Word Embedding: ", embed_matrix)
