# Required function for running this assignment
# Written by Mehdi Rezvandehy


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import accuracy_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import PercentFormatter
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

import nltk
#Tokenization
from nltk.tokenize import word_tokenize
#Stop words
from nltk.corpus import stopwords
stopword_set = set(stopwords.words('english'))
#Stemming 
from nltk.stem import PorterStemmer
#Lemmatization 
from nltk.corpus import wordnet
import re
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


##############################################################
def stem(words):
    """Stemming"""
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in words]

def categorize(words):
    tags = nltk.pos_tag(words)
    return [tag for word, tag in tags]

def lemmatize(words, tags):
    """Lemmatization"""
    lemmatizer = WordNetLemmatizer()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    pos = [tag_dict.get(t[0].upper(), wordnet.NOUN) for t in tags]
    return [lemmatizer.lemmatize(w, pos=p) for w, p in zip(words, pos)]

##############################################################

############## Spell Checking ##############
def spell_check(txt,maxword,wordcost,nonuse_words=['BNAME','ACCOUNT','CURRBAL','DELPAY'],Lemmatization=True):
    """
    spell checking for words of a document.
    
    txt          : document to be spell checked
    maxword      : maximum number of words in corpus 
    wordcost     : cost dictionary, assuming Zipf's law
    nonuse_words : list of words should not be spell checked
    
    """

    def infer_spaces(s):
        """Uses dynamic programming to infer the location of spaces in a string
        without spaces."""
    
        # Find the best match for the i first characters, assuming cost has
        # been built for the i-1 first characters.
        # Returns a pair (match_cost, match_length).
        def best_match(i):
            candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
            return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)
    
        # Build the cost array.
        cost = [0]
        for i in range(1,len(s)+1):
            c,k = best_match(i)
            cost.append(c)
            
        # Backtrack to recover the minimal-cost string.
        out = []
        i = len(s)
        while i>0:
            c,k = best_match(i)
            assert c == cost[i]
            out.append(s[i-k:i])
            i -= k
        return " ".join(reversed(out))

    # Convert n't to not
    txt=txt.replace('n’t', ' not')
    
    # Remove all numbers
    txt = re.sub('[0-9]+', '', txt)
    
    # Tokenization
    tokens = word_tokenize(txt)    
    
    # Remove punctuation (e.g. ',','.')
    words_doc = [word.lower() for word in tokens if word.isalnum()]
   
    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos = categorize(words_doc) 
        words_doc  = lemmatize(words_doc, words_pos) 
    
    mis_spell=[]
    corr_spell=[]
    for iwords in words_doc:
        if iwords not in nonuse_words:
            if iwords==infer_spaces(iwords):
                pass
            else:
                mis_spell.append(iwords)
                corr_spell.append(infer_spaces(iwords))
                
    result_df = pd.DataFrame()
    result_df['miss_spell'] = mis_spell  
    result_df['suggestion'] = corr_spell 
                
    return result_df                 

############## Jaccard Similarity ##############

def Jaccard_Similarity(doc1, doc2, stopwords=False, Stemming=False,Lemmatization=True,
                   nonuse_words=None): 
    """Jaccard Similarity""" 
    
    # Tokenization
    tokens_1 = word_tokenize(doc1)
    tokens_2 = word_tokenize(doc2)
    
    # Remove punctuation (e.g. ',','.')
    words_doc1 = [word.lower() for word in tokens_1 if word.isalnum()]
    words_doc2 = [word.lower() for word in tokens_2 if word.isalnum()]
    
    # Remove stop words (e.g. 'The','I','It')
    if stopwords:
        words_doc1 = [word for word in words_doc1 if word not in stopword_set]    
        words_doc2 = [word for word in words_doc2 if word not in stopword_set] 
    
    # Stemming (remove affixes)
    if Stemming:
        words_doc1 = stem(words_doc1)
        words_doc2 = stem(words_doc2)
        
    if nonuse_words:
        nonuse_words = [word.lower() for word in nonuse_words if word.isalnum()]
        words_doc1=[iwords for iwords in words_doc1 if iwords not in nonuse_words] 
        words_doc2=[iwords for iwords in words_doc2 if iwords not in nonuse_words]        
    
    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos1 = categorize(words_doc1) 
        words_doc1  = set(lemmatize(words_doc1, words_pos1)) 
        words_pos2 = categorize(words_doc2) 
        words_doc2  = set(lemmatize(words_doc2, words_pos2)) 
    
    # Find the intersection of words list of doc1 & doc2
    intersection = set(words_doc1).intersection(set(words_doc2))

    # Find the union of words list of doc1 & doc2
    union = set(words_doc1).union(set(words_doc2))
        
    # Calculate Jaccard similarity score 
    # using length of intersection set divided by length of union set
    return float(len(intersection)) / len(union)


############## Cosine_Similarity ##############

#CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

def Cosine_Similarity(doc1, doc2, stopwords=False, Stemming=False,Lemmatization=True,counts=False,
                     nonuse_words=None): 
    """ Cosine Similarity """
    # Tokenization
    tokens_1 = word_tokenize(doc1)
    tokens_2 = word_tokenize(doc2)
    
    # Remove punctuation (e.g. ',','.')
    words_doc1 = [word.lower() for word in tokens_1 if word.isalnum()]
    words_doc2 = [word.lower() for word in tokens_2 if word.isalnum()]
    
    # Remove stop words (e.g. 'The','I','It')
    if stopwords:
        words_doc1 = [word for word in words_doc1 if word not in stopword_set]    
        words_doc2 = [word for word in words_doc2 if word not in stopword_set]     
    
    ## Stemming (remove affixes)
    if Stemming:
        words_doc1 = stem(words_doc1)
        words_doc2 = stem(words_doc2)
    
    if nonuse_words:
        nonuse_words = [word.lower() for word in nonuse_words if word.isalnum()]
        words_doc1=[iwords for iwords in words_doc1 if iwords not in nonuse_words] 
        words_doc2=[iwords for iwords in words_doc2 if iwords not in nonuse_words]    
    
    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos1 = categorize(words_doc1) 
        words_doc1  = lemmatize(words_doc1, words_pos1) 
        words_pos2 = categorize(words_doc2) 
        words_doc2  = lemmatize(words_doc2, words_pos2)     
    
    # CountVectorizer
    words_doc1 = [" ".join(words_doc1)]
    words_doc2 = [" ".join(words_doc2)]
    words_f=words_doc1+words_doc2
    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(words_f)
    doc_term_matrix = sparse_matrix.todense()
 
    if (counts):
        df_counts = pd.DataFrame(doc_term_matrix, 
                          columns=count_vectorizer.get_feature_names(), 
                          index=['document1', 'document2'])
    
    # Consine similarity
    cos=cosine_similarity(doc_term_matrix[0], doc_term_matrix[1])[0][0]   
    
    if (counts):
        return cos, df_counts
    else: 
        return cos


############## Term frequency log-transformed (tf-idf) ##############

from sklearn.feature_extraction.text import TfidfVectorizer

def Cosine_Similarity_tf_idf(doc1, doc2, stopwords=False, Stemming=False, tf_idf=False,Lemmatization=True,
            counts=False,nonuse_words=['svc','payinfourl','FNAME','BNAME','ACCOUNT','CURRBAL',
                                    'DELPAY','SUSPDATE','Rogers','24','us','Msg']): 
    """ Cosine Similarity tf_idf """
    
    # Tokenization
    tokens_1 = word_tokenize(doc1)
    tokens_2 = word_tokenize(doc2)
    
    # Remove punctuation (e.g. ',','.')
    words_doc1 = [word.lower() for word in tokens_1 if word.isalnum()]
    words_doc2 = [word.lower() for word in tokens_2 if word.isalnum()]
    
    # Remove stop words (e.g. 'The','I','It')
    if stopwords:
        words_doc1 = [word for word in words_doc1 if word not in stopword_set]    
        words_doc2 = [word for word in words_doc2 if word not in stopword_set]     
    
    ## Stemming (remove affixes)
    if Stemming:
        words_doc1 = stem(words_doc1)
        words_doc2 = stem(words_doc2)

    nonuse_words = [word.lower() for word in nonuse_words if word.isalnum()]
    words_doc1=[iwords for iwords in words_doc1 if iwords not in nonuse_words] 
    words_doc2=[iwords for iwords in words_doc2 if iwords not in nonuse_words]          
    
    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos1 = categorize(words_doc1) 
        words_doc1  = lemmatize(words_doc1, words_pos1) 
        words_pos2 = categorize(words_doc2) 
        words_doc2  = lemmatize(words_doc2, words_pos2)  
        
    words_doc1 = [" ".join(words_doc1)]
    words_doc2 = [" ".join(words_doc2)]
    words_f=words_doc1+words_doc2        

    if tf_idf:
        # TfidfVectorizer
        vect = TfidfVectorizer(min_df=1)                                                                                                                                                                                                   
        tfidf = vect.fit_transform(words_f)
        doc_term_matrix = tfidf.todense()
    else:
        # CountVectorizer
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(words_f)
        doc_term_matrix = sparse_matrix.todense()        
        
    if (counts):
        df_counts = pd.DataFrame(doc_term_matrix, 
                          columns=vect.get_feature_names(), 
                          index=['document1', 'document2'])
    
    # Consine similarity
    cos=cosine_similarity(doc_term_matrix[0], doc_term_matrix[1])[0][0]   
        
    if (counts):
        return cos, df_counts
    else: 
        return cos 


############## Glove Cosine Embedding ##############

import scipy

def Glove_Cosine_Embedding(doc1, doc2,model,stopwords=False, Stemming=False,Lemmatization=True,
                 nonuse_words=None): 
    """ Glove Cosine Embedding """
    # Tokenization
    tokens_1 = word_tokenize(doc1)
    tokens_2 = word_tokenize(doc2)
    
    # Remove punctuation (e.g. ',','.')
    words_doc1 = [word.lower() for word in tokens_1 if word.isalnum()]
    words_doc2 = [word.lower() for word in tokens_2 if word.isalnum()]
    
    # Remove stop words (e.g. 'The','I','It')
    if stopwords:
        words_doc1 = [word for word in words_doc1 if word not in stopword_set]    
        words_doc2 = [word for word in words_doc2 if word not in stopword_set]     
    
    ## Stemming (remove affixes)
    if Stemming:
        words_doc1 = stem(words_doc1)
        words_doc2 = stem(words_doc2)
    
    if nonuse_words:
        nonuse_words = [word.lower() for word in nonuse_words if word.isalnum()]
        words_doc1=[iwords for iwords in words_doc1 if iwords not in nonuse_words] 
        words_doc2=[iwords for iwords in words_doc2 if iwords not in nonuse_words]            

    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos1 = categorize(words_doc1) 
        words_doc1  = lemmatize(words_doc1, words_pos1) 
        words_pos2 = categorize(words_doc2) 
        words_doc2  = lemmatize(words_doc2, words_pos2)      
       
    vector_1 = np.mean([model[word] for word in words_doc1],axis=0)
    vector_2 = np.mean([model[word] for word in words_doc2],axis=0)     
    
    result_list = [[1-scipy.spatial.distance.cosine(model[word1], model[word2]) 
                    for word2 in words_doc2] for word1 in words_doc1]
    result_df = pd.DataFrame(result_list)
    result_df.columns = words_doc2
    result_df.index = words_doc1
        
    # Aggregated Consine similarity for two documents
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    return np.round((1-cosine),3), result_df


############## Plot correlation matrix ##############

def matrix_occure_prob(df,title,fontsize=11,vmin=-0.1, vmax=0.8,lable1='Sentence 1',pad=55,
                    lable2='Sentence 2',label='Cosine Similarity',rotation_x=90,axt=None,
                    num_ind=False,txtfont=6,lbl_font=9,shrink=0.8,cbar_per=False, 
                       xline=False):  
    
    """Plot correlation matrix"""
    ax = axt or plt.axes()
    colmn1=list(df.columns)
    colmn2=list(df.index)
    corr=np.zeros((len(colmn2),len(colmn1)))
    
    for l in range(len(colmn1)):
        for l1 in range(len(colmn2)):
            cc=df[colmn1[l]][df.index==colmn2[l1]].values[0]
            try:
                if len(cc)>1:
                    corr[l1,l]=cc[0]  
            except TypeError:
                corr[l1,l]=cc            
            if num_ind:
                ax.text(l, l1, str(round(cc,2)), va='center', ha='center',fontsize=txtfont)
    im =ax.matshow(corr, cmap='jet', interpolation='nearest',vmin=vmin, vmax=vmax)
    cbar =plt.colorbar(im,shrink=shrink,label=label) 
    if (cbar_per):
        cbar.ax.set_yticklabels(['{:.0f}%'.format(x) for x in np.arange( 0,110,10)])    

    ax.set_xticks(np.arange(len(colmn1)))
    ax.set_xticklabels(colmn1,fontsize=lbl_font)
    ax.set_yticks(np.arange(len(colmn2)))
    ax.set_yticklabels(colmn2,fontsize=lbl_font)    
    
    # Set ticks on both sides of axes on
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax.xaxis.get_major_ticks()], rotation=rotation_x,
             ha="right", va="center", rotation_mode="anchor")
    
    # Rotate and align bottom ticklabels
    plt.setp([tick.label1 for tick in ax.yaxis.get_major_ticks()], rotation=rotation_x,
             ha="right", va="center", rotation_mode="anchor")
    
    if xline:
        x_labels = list(ax.get_xticklabels())
        x_label_dict = dict([(x.get_text(), x.get_position()[0]) for x in x_labels])
        
        for ix in xline:
            plt.axvline(x=x_label_dict[ix]-0.5,linewidth =1.2,color='k', linestyle='--')
            plt.axhline(y=x_label_dict[ix]-0.5,linewidth =1.2,color='k', linestyle='--')  

    plt.xlabel(lable1)
    plt.ylabel(lable2)    
    ax.grid(color='k', linestyle='-', linewidth=0.05)
    plt.title(f'{title}',fontsize=fontsize, pad=pad)
    plt.show()

    
############## Word Mover Distance ##############

def Word_Mover_Distance(doc1, doc2,model, stopwords=False, Stemming=False,Lemmatization=True,
                      nonuse_words=['svc','payinfourl','FNAME','BNAME','ACCOUNT','CURRBAL',
                                    'DELPAY','SUSPDATE','Rogers','24','us','Msg']): 
    """ Word_Mover_Distance """
    # Tokenization
    tokens_1 = word_tokenize(doc1)
    tokens_2 = word_tokenize(doc2)
    
    # Remove punctuation (e.g. ',','.')
    words_doc1 = [word.lower() for word in tokens_1 if word.isalnum()]
    words_doc2 = [word.lower() for word in tokens_2 if word.isalnum()]
    
    # Remove stop words (e.g. 'The','I','It')
    if stopwords:
        words_doc1 = [word for word in words_doc1 if word not in stopword_set]    
        words_doc2 = [word for word in words_doc2 if word not in stopword_set]     
    
    ## Stemming (remove affixes)
    if Stemming:
        words_doc1 = stem(words_doc1)
        words_doc2 = stem(words_doc2)
    
    nonuse_words = [word.lower() for word in nonuse_words if word.isalnum()]
    words_doc1=[iwords for iwords in words_doc1 if iwords not in nonuse_words] 
    words_doc2=[iwords for iwords in words_doc2 if iwords not in nonuse_words]     
    
    # Lemmatization (irregular verbs,have,has..)
    if Lemmatization:
        words_pos1 = categorize(words_doc1) 
        words_doc1  = lemmatize(words_doc1, words_pos1) 
        words_pos2 = categorize(words_doc2) 
        words_doc2  = lemmatize(words_doc2, words_pos2)      
    

    Distance = model.wmdistance(words_doc1,words_doc2)

    result_list = [[model.wmdistance(word1,word2) for word2 in words_doc2] for word1 in words_doc1]
    result_df = pd.DataFrame(result_list)
    result_df.columns = words_doc2
    result_df.index = words_doc1    

    return Distance, result_df
    