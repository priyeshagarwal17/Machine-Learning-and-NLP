
# coding: utf-8

# In[76]:


import nltk
#natural language processing tool kit


# In[77]:


nltk.download("movie_reviews")


# In[78]:


nltk.download()


# list and download other datasets interactively
# 

# In[79]:


from nltk.corpus import movie_reviews


# In[80]:


#fileids method : gives access to a  list of all the files available


# In[81]:


movie_reviews.fileids()


# In[82]:


len(movie_reviews.fileids())
#counting number of files


# In[83]:


negative_fileids = movie_reviews.fileids('neg')
positive_fileids = movie_reviews.fileids('pos')
len(negative_fileids),len(positive_fileids)


# In[84]:


#raw: each file is split into sentences


# In[85]:


print(movie_reviews.raw(fileids=positive_fileids[0]))


# In[86]:


#tokenizing


# In[87]:


romeo_text = """Why then, O brawling love! O loving hate!
O any thing, of nothing first create!
O heavy lightness, serious vanity,
Misshapen chaos of well-seeming forms,
Feather of lead, bright smoke, cold fire, sick health,
Still-waking sleep, that is not what it is! 
This love feel I, that feel no love in this."""

#issues:
#hyphenated words (well-seeming)
# inconsistant  use of punctuation (love!) , (is!)


# In[88]:


romeo_text.split()


# In[89]:


#punkt : sophisticated word tokenizer


# In[90]:


nltk.download("punkt")


# In[91]:


romeo_words= nltk.word_tokenize(romeo_text)


# In[92]:


romeo_words
#now the punctiuations are seperated nicely


# In[93]:


movie_reviews.words(fileids=positive_fileids[0])


# In[94]:


#simplest way tfor analysing text is to thing about words as an unordered collection of words
#dictionary
{word: True for word in romeo_words}


# In[95]:


type(_)

#    '_'    is the output from last code i.e. the line above


# In[96]:


def build_bag_of_words_features(words):
    return{word: True for word in words}

#created a funcytion which accepts a set fo words and return a dictionary


# In[97]:


build_bag_of_words_features(romeo_words)


# In[98]:


nltk.download("stopwords")
#for stopwords like 'the' , 'is'


# In[99]:


import string


# In[100]:


string.punctuation


# In[101]:


useless_words= nltk.corpus.stopwords.words("english") + list (string.punctuation)


# In[102]:


useless_words


# In[103]:


type(_)


# In[104]:


def build_bag_of_words_features_filtered(words):
    return{
        word: 1 for word in words \
    if not word in useless_words}

#mean is the word is in not in useless words and in our dictionary


# In[105]:


build_bag_of_words_features_filtered(romeo_words)


# In[116]:


# Count frequencies of words

all_words = movie_reviews.words()
len(all_words)/1e6

#.words() function with no argment can extract all words form entire dataset

#almost 1.36 million words


# In[109]:


#filter out useless words


# In[111]:


filtered_words=[word for word in movie_reviews.words() if not word in useless_words]


# In[112]:


type(filtered_words)


# In[118]:


len(filtered_words)/1e6


# In[115]:


len(all_words)-_
#this is the total number of useless words in the dataset


# In[119]:


#create a counter object by importing counter from collections

from collections import Counter
word_counter = Counter(filtered_words)


# In[120]:


#most_common() to accss the most common words 
most_common_words = word_counter.most_common()[:10]


# In[121]:


most_common_words


# In[122]:


#Lets visualize the most common words


# In[124]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[128]:


sorted_word_counts = sorted(list(word_counter.values()),reverse= True)
#sorted word counter and values in it
plt.loglog(sorted_word_counts)

#create logarithmic plot
plt.ylabel("Frequency")
plt.xlabel("Word Rank")

#logarithmic plot


# In[136]:


plt.hist(sorted_word_counts,bins=50);
plt.ylabel("Frequency")
plt.xlabel("Word Rank")
#as we have many words that come lessers times like 0/1/2/3 we see that the graph peaks near 0


# In[139]:


plt.hist(sorted_word_counts,bins=50,log= True);
plt.ylabel("Frequency")
plt.xlabel("Word Rank")
#as we have many words that come lessers times like 0/1/2/3 we see that the graph peaks near 0
#Log scales allow a large range of values that are compressed into one point of the graph to be shown


# In[191]:


#Sentiment Analysis


# In[205]:


# Our Database is already devided into positive and negative reviews

negative_features = [
    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'neg') \
    for f in negative_fileids
]
    
positive_features = [
    (build_bag_of_words_features_filtered(movie_reviews.words(fileids=[f])), 'pos') \
    for f in positive_fileids
]   


# In[193]:


print(negative_features[3])


# In[194]:


print(positive_features[6])


# In[206]:


from nltk.classify import NaiveBayesClassifier


# In[184]:


#there are 1000 positive and 1000 negatiove reviews
# we can use 80% of data as training data  and rest asa test data


# In[207]:


split = 800


# In[208]:


sentiment_classifier =NaiveBayesClassifier.train(positive_features[:split]+negative_features[:split])


# In[209]:


nltk.classify.util.accuracy(sentiment_classifier, positive_features[:split]+negative_features[:split])*100
#checking accuracy of the model


# In[210]:


nltk.classify.util.accuracy(sentiment_classifier, positive_features[split:]+negative_features[split:])*100
#checking accuracy of the model


# In[213]:


sentiment_classifier.show_most_informative_features()
#showing the most informative features of the model.

