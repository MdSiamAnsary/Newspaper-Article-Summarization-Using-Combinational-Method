import numpy as np
import pandas as pd
from textblob import TextBlob
from nltk.tokenize import sent_tokenize
import networkx as nx
from rake_nltk import *
from rouge import Rouge
import glob
import os
import errno
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
# nltk.download('punkt')
import re
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.simplefilter("ignore")

# article_category = ""


def sentiment_score_neu(sen):
    if len(sen) > 0:
        if TextBlob(sen).sentiment.polarity == 0:
            return 1
        else:
            return 0
    else:
        return 0

def sentiment_score_pos(sen):
    if len(sen) > 0:
        if TextBlob(sen).sentiment.polarity > 0:
            return 1
        else:
            return 0
    else:
        return 0

def sentiment_score_neg(sen):
    if len(sen) > 0:
        if TextBlob(sen).sentiment.polarity < 0:
            return 1
        else:
            return 0
    else:
        return 0


# function to remove stopwords
def remove_stopwords(sen):
    stop_words = stopwords.words('english')
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new


def summarize(lines):

    document_sentiment = TextBlob(lines).sentiment.polarity

    # print(lines)

    '''
    if article_category == "sport":
        with open('sport_phraselist.pkl', 'rb') as f_s:
            phrase_list = pickle.load(f_s)

    elif article_category == "politics":
        with open('politics_phraselist.pkl', 'rb') as f_p:
            phrase_list = pickle.load(f_p)

    elif article_category == "entertainment":
        with open('entertainment_phraselist.pkl', 'rb') as f_e:
            phrase_list = pickle.load(f_e)

    elif article_category == "business":
        with open('business_phraselist.pkl', 'rb') as f_b:
            phrase_list = pickle.load(f_b)

    elif article_category == "tech":
        with open('tech_phraselist.pkl', 'rb') as f_t:
            phrase_list = pickle.load(f_t)
    '''

    with open('entertainment_phraselist.pkl', 'rb') as f_e:
        phrase_list = pickle.load(f_e)

    '''
    sentences = []
    for line in lines:
        sentences.append(sent_tokenize(line))
    sentences = [y for x in sentences for y in x]

    print(sentences)
    '''

    sentences = sent_tokenize(lines)
    # print(sentences)


    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    # print(clean_sentences)

    total_sen_count = len(clean_sentences)
    # print(total_sen_count)

    sentiment_scores = []
    keyphrase_based_scores = []

    for each_sen in sentences:
        # sentiment_scores.append(sentiment_score(each_sen))

        if document_sentiment == 0:
            sentiment_scores.append(sentiment_score_neu(each_sen))
        elif document_sentiment > 0:
            sentiment_scores.append(sentiment_score_pos(each_sen))
        else:
            sentiment_scores.append(sentiment_score_neg(each_sen))

        r = Rake()
        r.extract_keywords_from_text(each_sen)
        phrases_in_each_sen = r.get_ranked_phrases()

        phr_count = 0
        for each_phr in phrases_in_each_sen:
            if each_phr in phrase_list:
                phr_count = phr_count + 1
        if phr_count == 0:
            keyphrase_based_scores.append(0)
        else:
            keyphrase_based_scores.append(phr_count / len(each_sen))

    # for i in range(int(len(clean_sentences))):
        # sentiment_scores.append(sentiment_score(str(clean_sentences[i])))

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    for i in range(int(total_sen_count)):
        # print(scores[i])
        scores[i] = scores[i] + sentiment_scores[i] + keyphrase_based_scores[i]
        # print(scores[i])

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
    # Extract top 10 sentences as the summary
    summary_sentences = []
    for i in range(int((total_sen_count*40)/100)):
        # print(ranked_sentences[i][1])
        summary_sentences.append(str(ranked_sentences[i][1]))

    summary = ""
    for sen in sentences:
        if sen in summary_sentences:
            summary = summary + str(TextBlob(sen).correct()) + " "

    return summary


it_ = 0

input_articles = []
generated_summaries = []

for (root, subdirs, files) in os.walk('input/'):
    if not subdirs:
        for file in files:
            document_text = ''
            heading = True
            with open(root + '/' + file, 'r', encoding='latin1') as f:
                for line in f:
                    if heading:
                        heading = False
                        continue
                    temp_line = line.strip()
                    if len(document_text) == 0:
                        document_text = temp_line
                    else:
                        document_text = document_text + ' ' + temp_line

            # print("Article : " + document_text)
            gen_summ = summarize(document_text)
            # print("Generated summary : " + gen_summ)

            generated_summaries.append(gen_summ)
            print(it_)

            it_ = it_ + 1


it_ = 0
provided_summaries = []
for (root, subdirs, files) in os.walk('summary/'):
    if not subdirs:
        for file in files:
            summary_text = ''
            with open(root + '/' + file, 'r', encoding='latin1') as f:
                for line in f:
                    temp_line = line.strip()
                    if len(summary_text) == 0:
                        summary_text = temp_line
                    else:
                        summary_text = summary_text + ' ' + temp_line
            provided_summaries.append(summary_text)
            it_ = it_ + 1

file_open = open("evaluation_result.txt", "w+")
no_of_doc = len(generated_summaries)
i = 0

while i < no_of_doc:
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries[i], provided_summaries[i])
    scores = str(scores)
    # print(scores)
    file_open.write(scores + "\n")
    i += 1

