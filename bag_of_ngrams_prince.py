
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[1]:

get_ipython().run_cell_magic('javascript', '', "\n$.getScript('https://kmahelona.github.io/ipython_notebook_goodies/ipython_notebook_toc.js')")


# ## import modules

# In[2]:

import itertools
import os
import random
import pickle as pkl
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import spacy
import string
import torch
import torch.nn as nn
import torch.nn.functional as F


# ## Data Loading
# 
# The dataset was downloaded from: http://ai.stanford.edu/~amaas/data/sentiment/

# In[3]:

data_loc = "data/imdb_reviews/"


# In[4]:

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


# In[5]:

train_pos = read_txt_files(folder_path=data_loc+"train/pos/")
print(len(train_pos))
train_neg = read_txt_files(folder_path=data_loc+"train/neg/")
print(len(train_neg))
test_pos = read_txt_files(folder_path=data_loc+"test/pos/")
print(len(test_pos))
test_neg = read_txt_files(folder_path=data_loc+"test/neg/")
print(len(test_neg))


# In[6]:

print("Train Positive examples = " + str(len(train_pos)))
print("Train Negative examples = " + str(len(train_neg)))
print("Test Positive examples = " + str(len(test_pos)))
print("Test Negative examples = " + str(len(test_neg)))


# ## Label Generation

# In[7]:

train_pos_labels = np.ones((len(train_pos),), dtype=int)
train_pos_labels

train_neg_labels = np.zeros((len(train_neg),), dtype=int)
train_neg_labels

train_data_labels = np.concatenate((train_pos_labels,train_neg_labels))
print(len(train_data_labels))
print(train_data_labels)

test_pos_labels = np.ones((len(test_pos),), dtype=int)
test_neg_labels = np.zeros((len(test_neg),), dtype=int)
test_data_labels = np.concatenate((test_pos_labels,test_neg_labels))
print(len(test_data_labels))
print(test_data_labels)


# ## Data Cleaning

# ### Removing HTML Tags

# In[8]:

import re

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


# In[9]:

train_pos_clean = [cleanhtml(x) for x in train_pos]
train_neg_clean = [cleanhtml(x) for x in train_neg]

test_pos_clean = [cleanhtml(x) for x in test_pos]
test_neg_clean = [cleanhtml(x) for x in test_neg]


# ## Merging Negatives and Positives

# In[10]:

train_all_clean = train_pos_clean + train_neg_clean
len(train_all_clean)

test_all_clean = test_pos_clean + test_neg_clean
len(test_all_clean)


# ## Creating the Validation Set 

# In[11]:

# should be smaller than 25000
training_size = 20000

assert training_size < 25000


# In[12]:

shuffled_index = np.random.permutation(len(train_all_clean))
print(len(shuffled_index))
print(shuffled_index)


# In[13]:

training_all_clean = [train_all_clean[i] for i in shuffled_index[:training_size]]
training_labels = [train_data_labels[i] for i in shuffled_index[:training_size]]
print(len(training_all_clean))
print(len(training_labels))


# In[14]:

validation_all_clean = [train_all_clean[i] for i in shuffled_index[training_size:]]
validation_labels = [train_data_labels[i] for i in shuffled_index[training_size:]]
print(len(validation_all_clean))
print(len(validation_labels))


# In[54]:

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
def tokenize(sent, n_gram = 0, lemmatize = False):
    
    tokens = tokenizer(sent)
    
    # unigrams
    if lemmatize == False:
        unigrams = [token.text.lower() for token in tokens if (token.text not in punctuations)]
    else:
        #LEMMATIZED
        unigrams = [token.lemma_.lower() for token in tokens if (token.text not in punctuations)]
    
    
    output = []
    output.extend(unigrams)
    
    n = 2
    while n <= n_gram:
        ngram_tokens = [" ".join(unigrams[x:x+n])                             for x in range(len(unigrams)-n+1)]
        output.extend(ngram_tokens)
        n = n + 1
        
    return output


# In[55]:

def lower_case_remove_punc(parsed):
    return [token.text.lower() for token in parsed if (token.text not in punctuations)]

def tokenize_dataset(dataset, n_gram, lemmatize = False):
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
        tokens = tokenize(sample,n_gram, lemmatize = False)
        
        token_dataset.append(tokens)
        all_tokens += tokens
        
        itr = itr + 1

    return token_dataset, all_tokens


# ## Tokenization

# In[56]:

# convert token to id in the dataset
def token2index_dataset(tokens_data, token2id):
    indices_data = []
    for tokens in tokens_data:
        index_list = [token2id[token] if token in token2id else UNK_IDX for token in tokens]
        indices_data.append(index_list)
    return indices_data


# In[57]:

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


# ## Create All Ngrams for future use

# In[58]:

grams = [1,2,3]
lemmatize_list = [True,False]


# In[60]:

for lemmatize_arg in lemmatize_list:
    for gram_no in grams:
        print(str(gram_no))

        train_data_tokens, all_train_tokens = tokenize_dataset(training_all_clean,
                                                               n_gram=gram_no, 
                                                               lemmatize = lemmatize_arg)

        # Tokenize Validation
        val_data_tokens, _ = tokenize_dataset(validation_all_clean,
                                              n_gram=gram_no, 
                                              lemmatize = lemmatize_arg)

        if lemmatize_arg == True:
            gram_no = str(gram_no) + "_lemma"
        else:
            gram_no = str(gram_no)
        print(gram_no)

        # val set tokens
        print ("Tokenizing val data")
        pkl.dump(val_data_tokens, open("val_data_tokens_"+str(gram_no)+".p", "wb"))

        # train set tokens
        print ("Tokenizing train data")
        pkl.dump(train_data_tokens, open("train_data_tokens_"+str(gram_no)+".p", "wb"))
        pkl.dump(all_train_tokens, open("all_train_tokens_"+str(gram_no)+".p", "wb"))


# In[47]:

MAX_SENTENCE_LENGTH = 200

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


# In[48]:

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


# In[49]:

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


# In[50]:

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


# In[51]:

params = [[1e-2,1e-1,1,2], ## learning rates
          list(range(1,4)), ## ngrams
          [1e5,1e6], ## vocab size
          [100,150,200], ## embedding size
#          [100,200], ## max sentence length
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


# In[52]:

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
        #max_sentence_length = int(param_comb[4]) # max sentence length of data loader
        BATCH_SIZE = param_comb[4]
        
        print("Learning Rate = " + str(lr_rate))
        print("Ngram = " + str(grams))
        print("Vocab Size = " + str(max_vocab_size))
        print("Embedding Dimension = " + str(embed_dimension))
        #print("Max Sentence Length = " + str(max_sentence_length))
        print("Batch Size = " + str(BATCH_SIZE))

        # Tokenization
        # All tokens are created before the hyperparameter search loop
        # Load the tokens here
        if lemmatize == True:
            grams = "lemma_" + str(grams)
        
        train_data_tokens = pkl.load(open("train_data_tokens_"+str(grams)+".p", "rb"))
        all_train_tokens = pkl.load(open("all_train_tokens_"+str(grams)+".p", "rb"))

        val_data_tokens = pkl.load(open("val_data_tokens_"+str(grams)+".p", "rb"))
        
        print("Train dataset size is {}".format(len(train_data_tokens)))
        print("Val dataset size is {}".format(len(val_data_tokens)))
        print("Total number of tokens in train dataset is {}".format(len(all_train_tokens)))
        
        # Building Vocabulary
        # implicitly gets the max_vocab_size parameter
        token2id, id2token = build_vocab(all_train_tokens,
                                         max_vocab_size=max_vocab_size)
        
        # Lets check the dictionary by loading random token from it
        random_token_id = random.randint(0, len(id2token)-1)
        random_token = id2token[random_token_id]
        print ("Token id {} -> token {}".format(random_token_id, id2token[random_token_id]))
        print ("Token {} -> token id {}".format(random_token, token2id[random_token]))
        
        train_data_indices = token2index_dataset(train_data_tokens, 
                                                 token2id = token2id)
        val_data_indices = token2index_dataset(val_data_tokens, 
                                               token2id = token2id)
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


# In[53]:

param_val_losses_adam = hyperparameter_search(hyperparameter_space = params,
                                         epochs = 5,
                                         optimizer_name = "Adam",
                                          lemmatize = False)


# In[ ]:



