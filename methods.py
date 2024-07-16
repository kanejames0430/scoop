import requests
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from textblob import TextBlob
import spacy
import datetime

nlp = spacy.load('en_core_web_trf')

def getArticleData(url):
    ''' 
    Gets the necessary data from a given URL.
    Returns a list of the form ["publication date", "raw text"]
    '''
    # Fetch the web page content
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the main article text
    article_text = []
    for paragraph in soup.find_all('p'):
        article_text.append(paragraph.get_text())

    # Join all the paragraphs into a single string
    full_text = '\n'.join(article_text)

    return full_text

def wordProcessor(text):
    ''' 
    Returns the processed text from a given string. 
    USE: analysis on word by word basis, not overall sentence or text
    - removes punctuation
    - removes extra whitespace
    - removes numbers
    - lower cases
    - removes stop words
    - tokenizes remaining words
    - removes names of speakers
    '''
    # Normalize text
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace

    # Remove named entities (proper nouns)
    doc = nlp(text)
    text = ' '.join([token.text for token in doc if token.ent_type_ != 'PERSON'])

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def sentiment_by_word(text):
    ''' 
    Returns the sentiment of a string on a word by word basis
    '''
    # initialize
    blob = TextBlob(text)
    word_sentiments = []

    # for each word, compute the sentiment and append it to the list
    for word in blob.words:
        word_blob = TextBlob(word)
        word_sentiment = word_blob.sentiment.polarity
        word_sentiments.append(word_sentiment)
    return word_sentiments

def sentiment_by_sentence(text):
    ''' 
    Returns the sentiment of a string on a sentence by sentence basis
    '''
    # initialize
    blob = TextBlob(text)
    sentence_sentiments = []

    # for each sentence, compute the sentiment and append it to the list
    for sentence in blob.sentences:
        sentence_sentiment = sentence.sentiment.polarity
        sentence_sentiments.append(sentence_sentiment)
    return sentence_sentiments

def overall_sentiment(text):
    ''' 
    Returns the overall sentiment of a given string
    '''
    scores = [sum(sentiment_by_word(text))/len(sentiment_by_word(text)), sum(sentiment_by_sentence(text))/len(sentiment_by_sentence(text))]
    return scores

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def verifyDate(date_string):
    date_pattern = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{1,2}, [0-9]{4}$")
    return bool(date_pattern.match(date_string))

def formatDate(date_string):
    month_dict = {
        "January": "01" or "1",
        "February": "02" or "2",
        "March": "03" or "3",
        "April": "04" or "4",
        "May": "05" or "5",
        "June": "06" or "6",
        "July": "07" or "7",
        "August": "08" or "8",
        "September": "09" or "9",
        "October": "10",
        "November": "11",
        "December": "12"
    }
    day, year = date_string.split()[1:]
    day = day.replace(",", "")  # Remove the comma from the day
    day = int(day)
    month = month_dict[date_string.split()[0]]
    return datetime.datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d").strftime("%Y-%m-%d")



def verifyDate(date_string):
    date_pattern = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December) [0-9]{1,2}, [0-9]{4}$")
    return bool(date_pattern.match(date_string))

def merge_dicts(dict1, dict2):
    return {**dict1, **dict2}