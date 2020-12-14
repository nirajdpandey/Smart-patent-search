#!/usr/bin/env python
# coding: utf-8

__Author__ = 'Niraj Dev Pandey'
__Date__ = '04 June 2020'
__Purpose__ = 'Case study data scientist for Allymatch GmbH'

# Import dependencies
import os
import glob
import nltk
import smart_open
import pandas as pd
from langdetect import detect
from termcolor import colored
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import gensim
from gensim.models import KeyedVectors

import warnings

warnings.filterwarnings("ignore")
nltk.download('popular')


# Check if you are running this project on the same versions of the library as author did
print("Checking your library compatibility....")


def library_check():
    if gensim.__version__ != '3.8.3':
        print("The project is developed on gensim 3.8.3")
        print("you are running on gensim {} version".format(gensim.__version__))
    if pd.__version__ != '1.0.3':
        print("The project is developed on Pandas 1.0.3")
        print("you are running on Panda {} version".format(pd.__version__))
    if nltk.__version__ != '3.4.5':
        print("The project is developed on nltk 3.4.5")
        print("you are running on nltk {} version".format(nltk.__version__))
    else:
        print("congratulations...! you already have all the correct dependencies installed")


library_check()

# Read all the .XML files and extract text from them
# Thereafter writing those text to individual text files (same name as XML files)
print("Loading your files . . . ")
for filename in glob.glob(os.path.join("./data", '*.xml')):
    with open(filename, 'r', encoding="utf-8") as f:
        soup = BeautifulSoup(f, 'xml')
        text = ' '.join(soup.findAll(text=True))
    file = open(filename + ".txt", "w")
    file.write(str(text))
    file.close()

# Now, reading saved text files, which contains the patent information
patent_files = glob.glob("data/*.txt")
print("The number of text patent files are:", len(patent_files))

text_corpus = []


def read_corpus(text_file_folder):
    """
    Read text files within  a folder
    :param text_file_folder: insert text file folder path
    :return list of content text files contains
    """
    for i in text_file_folder:
        with smart_open.smart_open(i, encoding="utf-8") as f:
            text_corpus.append(f.read())
    return text_corpus


def clean_corpus(path_to_text_files):
    """
    This function clean the text files contents to large extent.
    It also uses the function above read_corpus. It cleans the data by their language
    We are only interested in English patents for this prototype.
    :param path_to_text_files: path to the text file folders
    :return: Only English text files out of all data
    """
    lang_en = []  # English patent
    lang_else = []  # Other language patent

    text_files = glob.glob(path_to_text_files)  # Load all text files from data folder
    print("Total text files we cleaned are: ", len(text_files))
    data = read_corpus(text_files)  # Read all the text files
    data = list(map(lambda s: s.strip(), data))  # Remove newline from list
    data = list(filter(None, data))  # remove empty lines
    for i in data:
        if detect(str(i)) == "en":
            lang_en.append(i)
        else:
            lang_else.append(i)
    print("Total number of 'English' written patent are:", len(lang_en))
    print("Number of 'other' languages written patent are:", len(lang_else))
    return lang_en  # We are only interested for En patent


final_data = clean_corpus("data/*.txt")


def preprocess(doc):
    """
    After data is ready. This function takes documents and fine grain them with stop word removal
    and making whol document case insensitive.
    :param doc: list of documents
    :return: Cleaned list of documents
    """
    doc = doc.lower()  # Lower the text.
    doc = nltk.word_tokenize(doc)  # Split into words.
    stop_words = set(stopwords.words('english'))
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    #     doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


question_listperdoc = [' '.join(preprocess(question)) for question in final_data]
# Load word embedding
filename = 'GoogleNews-vectors-negative300.bin'
# Set the limit so that your poor machine can load them.
# Else it will crash
model = KeyedVectors.load_word2vec_format(filename, limit=400000, binary=True)

# Start your search here. Enter your query and find best matched patent.
# It's upto you to decide how many matched patent you want in return
# enter 'stop' once done.
while True:
    print(colored("==" * 55, 'green'))
    query = input(colored('Your Question ->>', 'red'))
    if query == "stop":
        break

    list_distances = []
    stop_words = set(stopwords.words('english'))
    sent1 = [word for word in nltk.word_tokenize(query) if word not in stop_words]
    # print("Words taken as a query", sent1)
    print("Looking for matching patent . . . ")
    for cont in question_listperdoc:
        cont = nltk.sent_tokenize(str(cont))
        para = ''.join(x for x in cont)
        sent2 = [word for word in nltk.word_tokenize(para) if word not in stop_words]
        wmd_distance = model.wmdistance(sent1, sent2)
        list_distances.append(wmd_distance)
    WMD_Dataframe = pd.DataFrame({'Sentence':
                                      final_data,
                                  'WMD_Score': list_distances}).sort_values(by=['WMD_Score'],
                                                                            ascending=True)
    WMD_Dataframe = WMD_Dataframe.reset_index()
    # Change below lines index [0] to any number (limit is No. of total Doc) to find more result.
    print(WMD_Dataframe.Sentence[0])
