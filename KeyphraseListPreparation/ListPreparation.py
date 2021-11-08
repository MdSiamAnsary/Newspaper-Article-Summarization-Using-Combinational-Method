import errno
from rake_nltk import *
from glob import *
import glob
from string import punctuation
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re
nltk.download()


def not_stopword(word):

    if word.lower() in nltk.corpus.stopwords.words('english'):
        return False
    else:
        return True


def process(keyphrase):

    lemmatizer = WordNetLemmatizer()
    result = ''
    word_list = keyphrase.split()

    for words in word_list:
        words = lemmatizer.lemmatize(words)
        words = str(words).strip().lower()
        words = re.sub('[^A-Za-z]+', '', words)

        if not_stopword(words):
            result = result + ' ' + words

    return result.strip()


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def hasPunctuation(text):
    return any(p in text for p in punctuation)


def rake_function(text1):

    no_of_doc = len(text1)
    i = 0

    article = ""
    while i < no_of_doc:
        article = article + text1[i]
        i += 1

    r = Rake()
    r.extract_keywords_from_text(article)
    phr = r.get_ranked_phrases()
    phr_len = len(phr)

    phrase_list = []
    iter = 0

    while iter < phr_len:

        if hasNumbers(phr[iter]) is False:
            if (len(phr[iter].split()) <= 2):
                if hasPunctuation(phr[iter]) is False:

                    phrase_list.append(phr[iter])

        iter = iter+1

    return phrase_list


def key_phrase_extraction(category):

    plain_texts = []

    path_output = 'Input/'+category+'/*.txt'
    files_output = glob.glob(path_output)
    for name_output in files_output:
        try:
            with open(name_output) as f_output:
                for line_output in f_output:
                    plain_texts.append(line_output)
        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise

    phrase_list = rake_function(plain_texts)

    final_phrase_list = []

    for phrase in phrase_list:
        cleaned_phrase = process(phrase)

        if cleaned_phrase not in final_phrase_list:
            final_phrase_list.append(cleaned_phrase)

    pickle_file_name = category + '_phraselist.pkl'

    with open(pickle_file_name, 'wb') as f:
        pickle.dump(final_phrase_list, f)

    print(category + " phrase list prepared")


categories = ['politics', 'entertainment', 'business', 'sport', 'tech']

for each_category in categories:
    key_phrase_extraction(each_category)
