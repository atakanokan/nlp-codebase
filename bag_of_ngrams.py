
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[1]:

get_ipython().run_cell_magic('javascript', '', "\n$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# This script uses bag-of-ngrams approach to sentiment classification using the IMDB review dataset.

# # PyTorch

# ## Data Loading

# The dataset was downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/

# In[4]:

import os


# In[5]:

data_loc = "data/imdb_reviews/"


# In[6]:

def read_txt_files(folder_path):
    """Reads all .txt files in a folder to a list"""
    
    file_list = os.listdir(folder_path)
    # for debugging, printing out the folder path and some files in it
    print(folder_path)
    print(file_list[:10])
    
    all_reviews = []
    for file_path in file_list:
        f = open(folder_path + file_path,"r")
        all_reviews.append(f.readline())
        
    return all_reviews


# In[7]:

import numpy as np


# In[8]:

train_pos = read_txt_files(folder_path=data_loc+"train/pos/")
print(len(train_pos))
train_neg = read_txt_files(folder_path=data_loc+"train/neg/")
print(len(train_neg))
test_pos = read_txt_files(folder_path=data_loc+"test/pos/")
print(len(test_pos))
test_neg = read_txt_files(folder_path=data_loc+"test/neg/")
print(len(test_neg))


# In[9]:

random_text = np.random.randint(1, high=len(train_pos)-1)
print(random_text)
train_pos[random_text]


# In[10]:

print("Train Positive examples = " + str(len(train_pos)))
print("Train Negative examples = " + str(len(train_neg)))
print("Test Positive examples = " + str(len(test_pos)))
print("Test Negative examples = " + str(len(test_neg)))


# ## Data Preparation

# ### Labeling the training dataset

# In[11]:

train_pos_labels = np.ones((len(train_pos),), dtype=int)
train_pos_labels


# In[12]:

train_neg_labels = np.zeros((len(train_neg),), dtype=int)
train_neg_labels


# In[13]:

train_data_labels = np.concatenate((train_pos_labels,train_neg_labels))
train_data_labels


# ### Storing the labels of the test set for Test Error Measuring

# In[14]:

test_pos_labels = np.ones((len(test_pos),), dtype=int)
test_neg_labels = np.zeros((len(test_neg),), dtype=int)
test_data_labels = np.concatenate((test_pos_labels,test_neg_labels))
print(len(test_data_labels))
test_data_labels


# ## Data Cleaning

# ### Removing HTML tags

# In[15]:

import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


# In[16]:

train_pos[random_text]


# In[17]:

train_pos_clean = [cleanhtml(x) for x in train_pos]
train_neg_clean = [cleanhtml(x) for x in train_neg]

test_pos_clean = [cleanhtml(x) for x in test_pos]
test_neg_clean = [cleanhtml(x) for x in test_neg]


# In[18]:

train_pos_clean[random_text]


# ### Replacing dots & question marks & paranthesis with space
# 
# It seems that punctuations 

# In[19]:

#"asdasdasds.asdasda".replace("."," ")


# In[ ]:

# def remove_dqmp(review):
    
#     review = review.replace("."," ")
#     review = review.replace("?"," ")
#     review = review.replace(")"," ")
#     review = review.replace("("," ")
    
#     return review


# In[ ]:

# remove_dqmp(train_pos_clean[random_text])


# In[ ]:

# train_pos_clean = [remove_dqmp(x) for x in train_pos_clean]
# train_neg_clean = [remove_dqmp(x) for x in train_neg_clean]


# ## Tokenization

# In[340]:

import spacy
import string

# Load English tokenizer, tagger, parser, NER and word vectors
tokenizer = spacy.load('en_core_web_sm')
punctuations = string.punctuation

# This is word tokenizer
# # lowercase and remove punctuation
# def tokenize(sent):
#     tokens = tokenizer(sent)
#     return [token.text.lower() for token in tokens if (token.text not in punctuations)]
#     #return [token.text.lower() for token in tokens]
    
# Modified for n-grams
def tokenize(sent, n_gram = 0):
    
    tokens = tokenizer(sent)
    
    # unigrams
    #unigrams = [token.text.lower() for token in tokens if (token.text not in punctuations)]
    unigrams = [token.lemma_.lower() for token in tokens if (token.text not in punctuations)]
    output = []
    output.extend(unigrams)
    
    n = 2
    while n <= n_gram:
        ngram_tokens = [" ".join(unigrams[x:x+n])                             for x in range(len(unigrams)-n+1)]
        output.extend(ngram_tokens)
        n = n + 1
        
    return output


# In[341]:

random_text = np.random.randint(1, high=len(train_pos)-1)
print(random_text)


# In[342]:

train_pos_clean[random_text]


# In[343]:

# Example
tokens = tokenize(train_pos_clean[random_text], n_gram = 4)
#tokens = tokenize(train_pos_clean[random_text])
print(tokens)


# ### Merging neg and pos examples - Training

# In[83]:

# to check the order of concatenation
train_data_labels


# In[84]:

train_all_clean = train_pos_clean + train_neg_clean
len(train_all_clean)


# ### Merging neg and pos examples - Test

# In[85]:

# to check the order of concatenation
test_data_labels


# In[86]:

test_all_clean = test_pos_clean + test_neg_clean
len(test_all_clean)


# ### Training -> Training + Validation

# In[87]:

# should be smaller than 25000
training_size = 20000

assert training_size < 25000


# In[88]:

shuffled_index = np.random.permutation(len(train_all_clean))
print(len(shuffled_index))
print(shuffled_index)


# In[89]:

shuffled_index[:training_size]


# In[90]:

training_all_clean = [train_all_clean[i] for i in shuffled_index[:training_size]]
training_labels = [train_data_labels[i] for i in shuffled_index[:training_size]]
print(len(training_all_clean))
print(len(training_labels))


# In[91]:

validation_all_clean = [train_all_clean[i] for i in shuffled_index[training_size:]]
validation_labels = [train_data_labels[i] for i in shuffled_index[training_size:]]
print(len(validation_all_clean))
print(len(validation_labels))


# ### Tokenizing the whole dataset

# In[344]:

def lower_case_remove_punc(parsed):
    return [token.text.lower() for token in parsed if (token.text not in punctuations)]

def tokenize_dataset(dataset, n_gram):
    token_dataset = []
    # we are keeping track of all tokens in dataset
    # in order to create vocabulary later
    all_tokens = []

#     for sample in tqdm_notebook(tokenizer.pipe(dataset, 
#                                                disable=['parser', 'tagger', 'ner'], 
#                                                batch_size=512, 
#                                                n_threads=4)):

    itr = 0
    for sample in dataset:
        
        if itr % 50 == 0:
            print(str(itr) + " / " + str(len(dataset)))
        # unigram version
        #tokens = lower_case_remove_punc(sample)
        
        # n-gram version
        tokens = tokenize(sample,n_gram)
        
        token_dataset.append(tokens)
        all_tokens += tokens
        
        itr = itr + 1

    return token_dataset, all_tokens


# In[113]:

from tqdm import tqdm_notebook


# In[114]:

import pickle as pkl


# In[115]:

# train set tokens
print ("Tokenizing train data")
train_data_tokens, all_train_tokens = tokenize_dataset(training_all_clean,
                                                       n_gram = 2)
pkl.dump(train_data_tokens, open("train_data_tokens.p", "wb"))
pkl.dump(all_train_tokens, open("all_train_tokens.p", "wb"))


# In[116]:

# val set tokens
print ("Tokenizing val data")
val_data_tokens, _ = tokenize_dataset(validation_all_clean,
                                     n_gram = 2)
pkl.dump(val_data_tokens, open("val_data_tokens.p", "wb"))


# In[117]:

# test set tokens
print ("Tokenizing test data")
test_data_tokens, _ = tokenize_dataset(test_all_clean,
                                      n_gram = 2)
pkl.dump(test_data_tokens, open("test_data_tokens.p", "wb"))


# In[118]:

print(train_data_tokens[:2])


# In[119]:

print(all_train_tokens[0:5])


# ### Remove blank space tokens
# 
# In the above tokenization, some blankspace strings were observed, thus this section adresses that by deleting them from the token list.

# In[ ]:

# blankspaces = [" ","  ","   "]


# In[ ]:

# def remove_blankspaces(review):
    
#     review = [x for x in review if x not in blankspaces] 
    
#     return review


# In[ ]:

# print(remove_blankspaces(tokens))


# In[ ]:

# train_data_tokens_clean = [remove_blankspaces(token) for token in train_data_tokens]
# len(train_data_tokens_clean)


# In[ ]:

# all_train_tokens_clean = remove_blankspaces(all_train_tokens)


# ## Building Vocabulary

# In[120]:

len(all_train_tokens)


# In[121]:

len(list(set(all_train_tokens)))


# we are going to create the vocabulary of most common 10,000 tokens in the training set.

# In[122]:

import random


# In[177]:

from collections import Counter

max_vocab_size = 10000
# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens,vocab_size=max_vocab_size):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token

token2id, id2token = build_vocab(all_train_tokens,vocab_size=10000)


# In[126]:

# Lets check the dictionary by loading random token from it

random_token_id = random.randint(0, len(id2token)-1)
random_token = id2token[random_token_id]

print ("Token id {} ; token {}".format(random_token_id, id2token[random_token_id]))
print ("Token {}; token id {}".format(random_token, token2id[random_token]))


# In[127]:

# convert token to id in the dataset
def token2index_dataset(tokens_data):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data

train_data_indices = token2index_dataset(train_data_tokens)
val_data_indices = token2index_dataset(val_data_tokens)
test_data_indices = token2index_dataset(test_data_tokens)

# double checking
print ("Train dataset size is {}".format(len(train_data_indices)))
print ("Val dataset size is {}".format(len(val_data_indices)))
print ("Test dataset size is {}".format(len(test_data_indices)))


# ## Dataset

# In[128]:

MAX_SENTENCE_LENGTH = 200


# In[129]:

import numpy as np
import torch
from torch.utils.data import Dataset


# In[130]:

class IMDBDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's 
    readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    
    def __init__(self, data_list, target_list):
        """
        @param data_list: list of newsgroup tokens 
        @param target_list: list of newsgroup targets 

        """
        self.data_list = data_list
        self.target_list = target_list
        assert (len(self.data_list) == len(self.target_list))

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        
        token_idx = self.data_list[key][:MAX_SENTENCE_LENGTH]
        label = self.target_list[key]
        return [token_idx, len(token_idx), label]


# In[131]:

def imdb_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    data_list = []
    label_list = []
    length_list = []
    #print("collate batch: ", batch[0][0])
    #batch[0][0] = batch[0][0][:MAX_SENTENCE_LENGTH]
    for datum in batch:
        label_list.append(datum[2])
        length_list.append(datum[1])
    # padding
    for datum in batch:
        padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        data_list.append(padded_vec)
    return [torch.from_numpy(np.array(data_list)), 
            torch.LongTensor(length_list), 
            torch.LongTensor(label_list)]


# In[132]:

BATCH_SIZE = 32
train_dataset = IMDBDataset(train_data_indices, training_labels)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=imdb_func,
                                           shuffle=True)

val_dataset = IMDBDataset(val_data_indices, validation_labels)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=imdb_func,
                                           shuffle=True)

test_dataset = IMDBDataset(test_data_indices, test_data_labels)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                           batch_size=BATCH_SIZE,
                                           collate_fn=imdb_func,
                                           shuffle=False)


# ## Bag of N-grams

# ### Training

# In[133]:

import torch
import torch.nn as nn
import torch.nn.functional as F


# In[134]:

class BagOfNgrams(nn.Module):
    """
    BagOfNgrams classification model
    """
    def __init__(self, vocab_size, emb_dim):
        """
        @param vocab_size: size of the vocabulary. 
        @param emb_dim: size of the word embedding
        """
        super(BagOfNgrams, self).__init__()
        # pay attention to padding_idx 
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.linear = nn.Linear(emb_dim,20)
    
    def forward(self, data, length):
        """
        
        @param data: matrix of size (batch_size, max_sentence_length). Each row in data represents a 
            review that is represented using n-gram index. Note that they are padded to have same length.
        @param length: an int tensor of size (batch_size), which represents the non-trivial (excludes padding)
            length of each sentences in the data.
        """
        out = self.embed(data)
        out = torch.sum(out, dim=1)
        out /= length.view(length.size()[0],1).expand_as(out).float()
     
        # return logits
        out = self.linear(out.float())
        return out


# In[135]:

emb_dim = 100
model = BagOfNgrams(len(id2token), emb_dim)


# In[136]:

learning_rate = 0.01
num_epochs = 10 # number epoch to train

# Criterion and Optimizer
criterion = torch.nn.CrossEntropyLoss()  
## try both sgd and adam
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# In[137]:

# Function for testing the model
def test_model(loader, model):
    """
    Help function that tests the model's performance on a dataset
    @param: loader - data loader for the dataset to test against
    """
    correct = 0
    total = 0
    model.eval()
    for data, lengths, labels in loader:
        data_batch, length_batch, label_batch = data, lengths, labels
        outputs = F.softmax(model(data_batch, length_batch), dim=1)
        predicted = outputs.max(1, keepdim=True)[1]
        
        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total)


# In[138]:

for epoch in range(num_epochs):
    for i, (data, lengths, labels) in enumerate(train_loader):
        model.train()
        data_batch, length_batch, label_batch = data, lengths, labels
        optimizer.zero_grad()
        outputs = model(data_batch, length_batch)
        loss = criterion(outputs, label_batch)
        loss.backward()
        optimizer.step()
        # check training score every 100 iterations
        ## validate every 100 iterations
        if i > 0 and i % 50 == 0:
            # validate
            val_acc = test_model(val_loader, model)
            train_acc = test_model(train_loader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Training Acc: {},Validation Acc: {}'.format( 
                       epoch+1, num_epochs, i+1, 
                len(train_loader), train_acc, val_acc))


# ## Hyperparameter Search

# The hyperparameters we are going to try to optimize are the following:
# 
# * n-gram max length
# * optimizer choice
# * embedding size
# * vocab size
# * learning rate of the optimizer
# 
# And maybe increase the batch size to speed up the optimization process.

# In[245]:

import itertools


# In[191]:

optimizers = [torch.optim.Adam(model.parameters(), 
                               lr=learning_rate),             
              torch.optim.SGD(model.parameters(), 
                              lr=learning_rate)]


# In[192]:

shuffled_index = np.random.permutation(len(train_all_clean))
print(len(shuffled_index))
print(shuffled_index)

shuffled_index[:training_size]

training_all_clean = [train_all_clean[i] for i in shuffled_index[:training_size]]
training_labels = [train_data_labels[i] for i in shuffled_index[:training_size]]
print(len(training_all_clean))
print(len(training_labels))

validation_all_clean = [train_all_clean[i] for i in shuffled_index[training_size:]]
validation_labels = [train_data_labels[i] for i in shuffled_index[training_size:]]
print(len(validation_all_clean))
print(len(validation_labels))


# In[273]:

from collections import Counter

# save index 0 for unk and 1 for pad
PAD_IDX = 0
UNK_IDX = 1

def build_vocab(all_tokens, max_vocab_size = 10000):
    # Returns:
    # id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    # token2id: dictionary where keys represent tokens and corresponding values represent indices
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2,2+len(vocab)))) 
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX 
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token


# #### Save all ngram tokens for easy use

# In[204]:

grams = params[1]
grams


# In[351]:

grams = 4

train_data_tokens, all_train_tokens = tokenize_dataset(training_all_clean,
                                                       n_gram=grams)

# Tokenize Validation
val_data_tokens, _ = tokenize_dataset(validation_all_clean,
                                      n_gram=grams)


# In[352]:

grams = "lemma_4"
print(grams)

# val set tokens
print ("Tokenizing val data")
pkl.dump(val_data_tokens, open("val_data_tokens_"+str(grams)+".p", "wb"))

# train set tokens
print ("Tokenizing train data")
pkl.dump(train_data_tokens, open("train_data_tokens_"+str(grams)+".p", "wb"))
pkl.dump(all_train_tokens, open("all_train_tokens_"+str(grams)+".p", "wb"))


# In[226]:

# print(train_data_tokens[0:2])


# In[227]:

# all_train_tokens[:2]


# In[228]:

# all_train_tokens[-3:]


# In[229]:

# print(val_data_tokens[:2])


# In[356]:

def hyperparameter_search(hyperparameter_space=params,
                          epochs=5,
                          optimizer_name = "Adam",
                          lemmatize = False):

    # returns all the permutations of the parameter search space
    param_space = [*itertools.product(*params)]
    
    # validation loss dictionary
    val_losses = {}
    
    # counter for progress
    count = 0
    
    for param_comb in param_space:
        print("-----------------------------------------------------------")
        print("Parameter Combination = " + str(count+1) + " / " + str(len(param_space)))
        count = count + 1      
        
        NUM_EPOCHS = epochs
        lr_rate = param_comb[0]             # learning rate
        grams = param_comb[1]               # n-grams
        max_vocab_size = int(param_comb[2]) # vocabulary size
        embed_dimension = param_comb[3]     # embedding vector size
        MAX_SENTENCE_LENGTH = param_comb[4] # max sentence length of data loader
        BATCH_SIZE = param_comb[5]
        
        print("Learning Rate = " + str(lr_rate))
        print("Ngram = " + str(grams))
        print("Vocab Size = " + str(max_vocab_size))
        print("Embedding Dimension = " + str(embed_dimension))
        print("Max Sentence Length = " + str(MAX_SENTENCE_LENGTH))
        print("Batch Size = " + str(BATCH_SIZE))

        # Tokenization
        # All tokens are created before the hyperparameter search loop
        # Load the tokens here
        if lemmatize == True:
            grams = "lemma_" + str(grams)
        
        train_data_tokens = pkl.load(open("train_data_tokens_"+str(grams)+".p", "rb"))
        all_train_tokens = pkl.load(open("all_train_tokens_"+str(grams)+".p", "rb"))

        val_data_tokens = pkl.load(open("val_data_tokens_"+str(grams)+".p", "rb"))
        
        print ("Train dataset size is {}".format(len(train_data_tokens)))
        print ("Val dataset size is {}".format(len(val_data_tokens)))
        print ("Total number of tokens in train dataset is {}".format(len(all_train_tokens)))
        
        # Building Vocabulary
        # implicitly gets the max_vocab_size parameter
        token2id, id2token = build_vocab(all_train_tokens,
                                         max_vocab_size=max_vocab_size)
        
        # Lets check the dictionary by loading random token from it
        random_token_id = random.randint(0, len(id2token)-1)
        random_token = id2token[random_token_id]
        print ("Token id {} -> token {}".format(random_token_id, id2token[random_token_id]))
        print ("Token {} -> token id {}".format(random_token, token2id[random_token]))
        
        train_data_indices = token2index_dataset(train_data_tokens)
        val_data_indices = token2index_dataset(val_data_tokens)
        # double checking
        print ("Train dataset size is {}".format(len(train_data_indices)))
        print ("Val dataset size is {}".format(len(val_data_indices)))
        
        

        # Load training and validation data
        train_dataset = IMDBDataset(train_data_indices, 
                                    training_labels)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=imdb_func,
                                                   shuffle=True)

        val_dataset = IMDBDataset(val_data_indices, 
                                  validation_labels)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
                                                   batch_size=BATCH_SIZE,
                                                   collate_fn=imdb_func,
                                                   shuffle=True)  

        # Initialize the N-gram Model
        model = BagOfNgrams(len(id2token), embed_dimension)
        
        # Both Adam and SGD will be tried
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)
        else:
            print("this optimizer is not implemented yet")
        
        # Cross Entropy Loss will be used
        criterion = torch.nn.CrossEntropyLoss()  
        
        # Validation Losses will be stored in a list
        # Caution: Two different optimizers
        val_losses[param_comb] = []
        
    #for optimizer in optimizers:
        print("Optimization Start")
        print(optimizer)

        for epoch in range(NUM_EPOCHS):
            for i, (data, lengths, labels) in enumerate(train_loader):
                model.train()
                data_batch, length_batch, label_batch = data, lengths, labels
                optimizer.zero_grad()
                outputs = model(data_batch, length_batch)
                loss = criterion(outputs, label_batch)
                loss.backward()
                optimizer.step()
                # Validate every 100 iterations
                # Adjust it to accustom changing batch sizes
                if i > 0 and i % (50 * (64 / BATCH_SIZE)) == 0:

                    # Accuracy Calculations
                    train_acc = test_model(train_loader, model)
                    val_acc = test_model(val_loader, model)
                    val_losses[param_comb].append(val_acc)

                    # Logging
                    print('Epoch:[{}/{}],Step:[{}/{}],Training Acc:{},Validation Acc:{}'.format( 
                               epoch+1, NUM_EPOCHS, 
                                i+1, len(train_loader), 
                                train_acc, val_acc))
                      
    return val_losses


# ### Setting the Search Space

# In[359]:

params = [[1e-2,1e-1,1,2], ## learning rates
          list(range(1,4)), ## ngrams
          [1e5,1e6], ## vocab size
          [100,150,200], ## embedding size
          [100,200], ## max sentence length
          [64,128] ## batch size
         ]

# params = [[1e-1,1,2,5], ## learning rates
#           list(range(1,2)), ## ngrams
#           [1e5], ## vocab size
#           [100], ## embedding size
#           [100], ## max sentence length
#           [64] ## batch size
#          ]

print(len([*itertools.product(*params)]))
[*itertools.product(*params)]


# ### Running the Grid Search

# #### Adam

# In[361]:

param_val_losses_adam = hyperparameter_search(hyperparameter_space = params,
                                         epochs = 5,
                                         optimizer_name = "Adam",
                                          lemmatize = False)


# #### SGD

# In[360]:

param_val_losses_sgd = hyperparameter_search(hyperparameter_space = params,
                                         epochs = 5,
                                         optimizer_name = "SGD",
                                          lemmatize = True)


# ### Analyzing the Results

# In[329]:

for key, value in param_val_losses_adam.items():
    print (key, value)


# In[ ]:

for key, value in param_val_losses_sgd.items():
    print (key, value)


# ### Validation Accuracy Plots

# In[ ]:



