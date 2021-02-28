# =============================================================================
# 
# 
# DATA EXTRACTION FROM THE INTERNET
# 
# 
# =============================================================================
# Web scraping, pickle imports
import requests
from bs4 import BeautifulSoup
import pickle

# =============================================================================
# # Scrapes transcript data from scrapsfromtheloft.com
# def url_to_transcript(url):
#     '''Returns transcript data specifically from scrapsfromtheloft.com.'''
#     page = requests.get(url).text
#     soup = BeautifulSoup(page, "lxml")
#     text = [p.text for p in soup.find(class_="elementor-widget-container").find_all('p')]
#     print(url)
#     return text
# 
# # URLs of transcripts in scope
# urls = ['http://scrapsfromtheloft.com/2017/05/06/louis-ck-oh-my-god-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/04/11/dave-chappelle-age-spin-2017-full-transcript/',
#         'http://scrapsfromtheloft.com/2018/03/15/ricky-gervais-humanity-transcript/',
#         'http://scrapsfromtheloft.com/2017/08/07/bo-burnham-2013-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/05/24/bill-burr-im-sorry-feel-way-2014-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/04/21/jim-jefferies-bare-2014-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/08/02/john-mulaney-comeback-kid-2015-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/10/21/hasan-minhaj-homecoming-king-2017-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/09/19/ali-wong-baby-cobra-2016-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/08/03/anthony-jeselnik-thoughts-prayers-2015-full-transcript/',
#         'http://scrapsfromtheloft.com/2018/03/03/mike-birbiglia-my-girlfriends-boyfriend-2013-full-transcript/',
#         'http://scrapsfromtheloft.com/2017/08/19/joe-rogan-triggered-2016-full-transcript/']
# 
# # Comedian names
# comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']
# 
# 
# transcripts = [url_to_transcript(u) for u in urls]
# for i, c in enumerate(comedians):
#     with open("Transcripts/" + c + ".txt", "wb") as file:
#         pickle.dump(transcripts[i], file)
# =============================================================================
comedians = ['louis', 'dave', 'ricky', 'bo', 'bill', 'jim', 'john', 'hasan', 'ali', 'anthony', 'mike', 'joe']
data =  {}
for i,c in enumerate(comedians):
    with open("Transcripts/" + c + ".txt", "rb") as file:
        data[c] = pickle.load(file)

# print(data.keys())

# print(data['louis'][:1])

def combineText (list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text
data_combined = {key:[combineText(value)] for (key,value) in data.items()}
# =============================================================================
# 
# 
# DATA CLEANING BEGINS HERE
# 
# 
# =============================================================================
import pandas as pd
pd.set_option('max_colwidth', 150)
data_df = pd.DataFrame.from_dict(data_combined).transpose()
data_df.columns = ['Transcript']
data_df = data_df.sort_index()
#data_df.Transcript.loc['ali']

import re
import string
def cleanData1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text) # Removes all the values in the square brackets
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # removes all the punctuation marks
    text = re.sub('\w*\d\w*', '', text) # removes alpha numeric number alphanumeric pattern characters
    return text
round1 = lambda x: cleanData1(x)
data_clean = pd.DataFrame(data_df.Transcript.apply(round1))
def cleanData2(text):
    text = re.sub('[''""...]', '', text)
    text = re.sub('\n', '', text)
    return text
round2 = lambda x : cleanData2(x)

full_names = ['Ali Wong', 'Anthony Jeselnik', 'Bill Burr', 'Bo Burnham', 'Dave Chappelle', 'Hasan Minhaj',
              'Jim Jefferies', 'Joe Rogan', 'John Mulaney', 'Louis C.K.', 'Mike Birbiglia', 'Ricky Gervais']

data_df['full_name'] = full_names
data_df.to_pickle("corpus.pkl")

data_clean = pd.DataFrame(data_clean.Transcript.apply(round2))

# Creating a document term matrix using count vectorizer and exclude common english words like "the", "a"
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(stop_words ='english')
data_cv = cv.fit_transform(data_clean.Transcript)
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
data_dtm.index = data_clean.index
data_dtm.to_pickle('dtm.pkl')

data_clean.to_pickle('data_clean.pkl')
pickle.dump(cv, open("cv.pkl", "wb"))
# =============================================================================
# 
# 
# EXPLORATORY DATA ANALYSIS
# 
# STEPS : 
#     DATA  - Determine the format of the raw data you'll need to start
#     AGGREGATE - Figure out how to aggregate the data
#     VISUALIZE - Find the best way to Visualize the data
#     INSIGHTS - Extract some key takeaways from the visualizations 
# 
# 
# =============================================================================
# =============================================================================
# 
# PREPARING A WORD CLOUD FOR ALL THE COMEDIANS (Most Used Words)
# 
# =============================================================================
data = pd.read_pickle('dtm.pkl')  # SAME as using data = data_dtm
data = data.transpose()
data.head()

top_dict = {}
for c in data.columns:
    top = data[c].sort_values(ascending = False).head(30)
    top_dict[c] = list(zip(top.index, top.values))
print(top_dict)
# =============================================================================
# 
# 
# CLEANING UP ANOTHER SET OF MOST USED WORDS BY ALL COMEDIANS TO GET A UNIQUE
# SET OF WORDS SPOKEN BY EACH PERSON
# 
# 
# =============================================================================
for comedian, top_words in top_dict.items():
    print(comedian)
    print(','.join([word for word, count in top_words[0:14]]))
    print('---')
    
from collections import Counter
words = []
for comedian in data.columns:
    top = [word for (word, count) in top_dict[comedian]]
    for t in top : 
        words.append(t)
print(words)
print(Counter(words).most_common())

add_stop_words = [word for word, count in Counter(words).most_common() if count > 6]
print(add_stop_words)

from sklearn.feature_extraction import text
data_clean = pd.read_pickle("data_clean.pkl")
stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)

cv = CountVectorizer(stop_words=stop_words)
data_cv = cv.fit_transform(data_clean.Transcript)
data_stop = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())
data_stop.index = data_clean.index

pickle.dump(cv, open("cv_stop.pkl", "wb"))
data_stop.to_pickle("dtm_stop.pkl")

from wordcloud import WordCloud
wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=140, random_state=100)
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [18,8]

for index, comedian in enumerate(data.columns):
    wc.generate(data_clean.Transcript[comedian])
    plt.subplot(3, 4, index+1)
    plt.imshow(wc, interpolation = 'bilinear')
    plt.axis("off")
    plt.title(full_names[index])
plt.show()
# =============================================================================
# 
# 
# CHECKING THE VOCABULARY OF ALL THE COMEDIANS
# 
# 
# =============================================================================

# Identify the words that occur the least for every comedian (non zero items in the document)

unique_list = []
for comedian in data.columns:
    uniques = data[comedian].to_numpy().nonzero()[0].size
    unique_list.append(uniques)
# New data frame for unique words by each comedian
data_words = pd.DataFrame(list(zip(full_names, unique_list)), columns=['comedian', 'unique_words'])
data_unique_sort = data_words.sort_values(by = 'unique_words')
print(data_unique_sort)



# =============================================================================
# 
# getting the words per minute
# source : IMDB (runtime of each comedian)
# we got the total runtime of the show for each comedian and the total number of words he spoke in the whole show
# dividing that we get the words per minute
# 
# NOTE : This is NOT the number of unique words per minute
# 
# 
# =============================================================================
# Calculate the words per minute of each comedian

# Find the total number of words that a comedian uses
total_list = []
for comedian in data.columns:
    totals = sum(data[comedian])
    total_list.append(totals)
    
# Comedy special run times from IMDB, in minutes
run_times = [60, 59, 80, 60, 67, 73, 77, 63, 62, 58, 76, 79]

# Let's add some columns to our dataframe
data_words['total_words'] = total_list
data_words['run_times'] = run_times
data_words['words_per_minute'] = data_words['total_words'] / data_words['run_times']

# Sort the dataframe by words per minute to see who talks the slowest and fastest
data_wpm_sort = data_words.sort_values(by='words_per_minute')
print(data_wpm_sort)
# PLOTTING

import numpy as np
y_pos = np.arange(len(data_words))
plt.subplot(1,2,1)
plt.barh(y_pos, data_unique_sort.unique_words, align='center')
plt.yticks(y_pos,  data_unique_sort.comedian)
plt.title("Number of unique words", fontsize = 15)

plt.subplot(1,2,2)
plt.barh(y_pos, data_wpm_sort.words_per_minute, align='center')
plt.yticks(y_pos,  data_wpm_sort.comedian)
plt.title("Number of words per minute", fontsize = 15)

plt.tight_layout()
plt.show()
# =============================================================================
# 
# 
# AMOUNT OF PROFANITY
# 
# NUMBER OF SHEETS AND FCUKS SAID BY EVERY COMEDIANS (OBSERVATION FROM THE WORD 
# CLOUD)
# 
# 
# =============================================================================
Counter(words).most_common()
# print(Counter(words).most_common())

data_bad_words = data.transpose()[['fucking', 'fuck', 'shit']]
data_profanity = pd.concat([data_bad_words.fucking + data_bad_words.fuck, 
                            data_bad_words.shit], axis=1)
data_profanity.columns = ['f_word', 's_word']



# Let's create a scatter plot of our findings
plt.rcParams['figure.figsize'] = [10, 8]

for i, comedian in enumerate(data_profanity.index):
    x = data_profanity.f_word.loc[comedian]
    y = data_profanity.s_word.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+1.5, y+0.5, full_names[i], fontsize=10)
    plt.xlim(-5, 155) 
    
plt.title('Number of Bad Words Used in Routine', fontsize=20)
plt.xlabel('Number of F Bombs', fontsize=15)
plt.ylabel('Number of S Words', fontsize=15)

plt.show()
"""

NEVER WATCH JOE ROGAN, BILL BURR, JIM JEFFEROES ON SPEAKER WHEN YOUR PARENTS ARE NEAR :P

"""

# =============================================================================
# 
# 
# NLP TECHNIQUES
# 
#
# =============================================================================

"""
1. Sentiment Analysis
INPUT : Corpus (We dont use the document term matrix because the words will be
                split not good = negative
                good = postive and in document term matrix we decided to split 
                words and remove stop words so it'll throw off the predictions 
                )
TEXTBLOB : python library built on top of nltk. Provide rules-based sentiment scores

OUTPUT : Give the sentiment score for each comedian(Postive/Negative)
        and a subjecivity score (How opinionated the comedians are)

"""

data = pd.read_pickle("corpus.pkl")

from textblob import TextBlob

pol = lambda x : TextBlob(x).sentiment.polarity
sub = lambda x : TextBlob(x).sentiment.subjectivity
data['Polarity'] = data['Transcript'].apply(pol)
data['Subjectivity'] = data['Transcript'].apply(sub)
print (data)

# Plotting

plt.rcParams['figure.figsize'] = [10,8]

for index, comedian in enumerate(data.index):
    x = data.Polarity.loc[comedian]
    y = data.Subjectivity.loc[comedian]
    plt.scatter(x, y, color='blue')
    plt.text(x+.001, y+.001, data['full_name'][index], fontsize=10)
    plt.xlim(-.01, .12) 
    
plt.title('Sentiment Analysis', fontsize=20)
plt.xlabel('<-- Negative -------- Positive -->', fontsize=15)
plt.ylabel('<-- Facts -------- Opinions -->', fontsize=15)

plt.show()

"""

CONCLUSION IF YOU WANT TO ATTEND A MOTIVATIONAL SPEECH THAN A STAND UP COMEDY GOTO HASAN'S SHOW :P

"""
# SENTIMENT OF ROUTINE OVER TIME

import math

def split_text(text, n=10):
    length = len(text)
    size = math.floor(length/n)
    start = np.arange(0, length, size)
    split_list = []
    for piece in range(n):
        split_list.append(text[start[piece]:start[piece]+size])
    return split_list
# splits the transcript to 10 pieces of text

list_pieces = []
for t in data.Transcript:
    split = split_text(t)
    list_pieces.append(split)

# calculate polarity of each piece of text
polarity_text = []
for lp in list_pieces:
    polarity_piece = []
    for p in lp:
        polarity_piece.append(TextBlob(p).sentiment.polarity)
    polarity_text.append(polarity_piece)
print(polarity_text)


# Show the plot for all comedians
plt.rcParams['figure.figsize'] = [16, 12]

for index, comedian in enumerate(data.index):    
    plt.subplot(3, 4, index+1)
    plt.plot(polarity_text[index])
    plt.plot(np.arange(0,10), np.zeros(10))
    plt.title(data['full_name'][index])
    plt.ylim(ymin=-.2, ymax=.3)
    
plt.show()

"""

TOPIC MODELING
1. INPUT : Document term matrix, NUmber of topics, number of iterations

2. GENSIM : Tool kit for Topic Modelling.

3. Technique : Latent Dirichlet Allocation (LDA)

4. OUTPUT : Themes each comedian speaks about

"""
from gensim import matutils,  models
import scipy.sparse

# =============================================================================
# data = pd.read_pickle("dtm_stop.pkl")
# tdm = data.transpose()
# 
# # Convert Document term Matrix to gensim accepted format
# # df => Sparse Matrix
# sparse_counts = scipy.sparse.csr_matrix(tdm)
# corpus = matutils.Sparse2Corpus(sparse_counts)
# 
# # Map of all terms and the respective position in the DTM
# cv = pickle.load(open('cv_stop.pkl','rb'))
# id2word = dict((v,k) for k,v in cv.vocabulary_.items())
# 
# lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=5, passes = 20)
# print(lda.print_topics())
#
# Colossal Failure because verbs and adverbs do not give a proper 
# topic next attempt to make tpoics from nouns
# =============================================================================
"""
ONLY NOUNS AS TOPICS
"""
from nltk import word_tokenize, pos_tag
import nltk
nltk.download()

def nouns(text):
    isNoun = lambda pos : pos[:2] == 'NN'
    tokenized = word_tokenize(text)
    all_nouns = [word for (word,pos) in pos_tag(tokenized) if isNoun(pos)]
    return ' '.join(all_nouns)

data_clean = pd.read_pickle('data_clean.pkl')

# Apply the nouns function to the transcripts to filter only on nouns
data_nouns = pd.DataFrame(data_clean.Transcript.apply(nouns))


# CREATING NEW DTM WITH ONLY NOUNS THIS TIME
# Re-add the additional stop words since we are recreating the document-term matrix
add_stop_words = ['like', 'im', 'know', 'just', 'dont', 'thats', 'right', 'people',
                  'youre', 'got', 'gonna', 'time', 'think', 'yeah', 'said', 've', 'ahah']
cvn = CountVectorizer(stop_words=stop_words)
data_cvn = cvn.fit_transform(data_nouns.Transcript)
data_dtmn = pd.DataFrame(data_cvn.toarray(), columns=cvn.get_feature_names())
data_dtmn.index = data_nouns.index

# Create the gensim corpus
corpusn = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmn.transpose()))

# Create the vocabulary dictionary
id2wordn = dict((v, k) for k, v in cvn.vocabulary_.items())

# Let's try 4 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=4, id2word=id2wordn, passes=50)
print(ldan.print_topics())


# Let's start with 2 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=2, id2word=id2wordn, passes=50)
print(ldan.print_topics())


# Let's start with 6 topics
ldan = models.LdaModel(corpus=corpusn, num_topics=6, id2word=id2wordn, passes=50)
print(ldan.print_topics())


# TOPIC MODELLING WITH NOUNS AND ADJECTIVES


# Let's create a function to pull out nouns from a string of text
def nouns_adj(text):
    '''Given a string of text, tokenize the text and pull out only the nouns and adjectives.'''
    is_noun_adj = lambda pos: pos[:2] == 'NN' or pos[:2] == 'JJ'
    tokenized = word_tokenize(text)
    nouns_adj = [word for (word, pos) in pos_tag(tokenized) if is_noun_adj(pos)] 
    return ' '.join(nouns_adj)

# Apply the nouns function to the transcripts to filter only on nouns
data_nouns_adj = pd.DataFrame(data_clean.Transcript.apply(nouns_adj))

cvna = CountVectorizer(stop_words=stop_words, max_df=.8)
data_cvna = cvna.fit_transform(data_nouns_adj.Transcript)
data_dtmna = pd.DataFrame(data_cvna.toarray(), columns=cvna.get_feature_names())
data_dtmna.index = data_nouns_adj.index

print ("##############################################################################################################")
# Create the gensim corpus
corpusna = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_dtmna.transpose()))

# Create the vocabulary dictionary
id2wordna = dict((v, k) for k, v in cvna.vocabulary_.items())

ldana = models.LdaModel(corpus=corpusna, num_topics=2, id2word=id2wordna, passes=200)
print(ldana.print_topics())

ldana = models.LdaModel(corpus=corpusna, num_topics=4, id2word=id2wordna, passes=200)
print(ldana.print_topics())

ldana = models.LdaModel(corpus=corpusna, num_topics=6, id2word=id2wordna, passes=200)
print(ldana.print_topics())

corpus_transformed = ldana[corpusna]
print(list(zip([a for [(a,b)] in corpus_transformed], data_dtmna.index)))

"""
TEXT GENERATION

CHECK NOTES

"""

data = pd.read_pickle('corpus.pkl')

# Extract only Ali Wong's text
ali_text = data.Transcript.loc['ali']
ali_text[:200]


from collections import defaultdict

def markov_chain(text):
    '''The input is a string of text and the output will be a dictionary with each word as
       a key and each value as the list of words that come after the key in the text.'''
    
    # Tokenize the text by word, though including punctuation
    words = text.split(' ')
    
    # Initialize a default dictionary to hold all of the words and next words
    m_dict = defaultdict(list)
    
    # Create a zipped list of all of the word pairs and put them in word: list of next words format
    for current_word, next_word in zip(words[0:-1], words[1:]):
        m_dict[current_word].append(next_word)

    # Convert the default dict back into a dictionary
    m_dict = dict(m_dict)
    return m_dict

# Create the dictionary for Ali's routine, take a look at it
ali_dict = markov_chain(ali_text)

import random

def generate_sentence(chain, count=15):
    '''Input a dictionary in the format of key = current word, value = list of next words
       along with the number of words you would like to see in your generated sentence.'''

    # Capitalize the first word
    word1 = random.choice(list(chain.keys()))
    sentence = word1.capitalize()

    # Generate the second word from the value list. Set the new word as the first word. Repeat.
    for i in range(count-1):
        word2 = random.choice(chain[word1])
        word1 = word2
        sentence += ' ' + word2

    # End it with a period
    sentence += '.'
    return(sentence)

print(
generate_sentence(ali_dict))