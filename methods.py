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

    # Extract the publication date
    # This part will vary depending on the website's structure
    # Here we assume the date is within a <time> tag or a <meta> tag with a specific property

    tempList = response.text.split('\n')
    for line in tempList:
        if "<title>" in line:
            date = getDate(line) 

    return date, full_text

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

def urlExtractor(url):
    ''' 
    Returns a list of hyperlinks on a given page. The returned string lack the prefix neeeded
    to be a complete URL
    '''
    response = requests.get(url)
    
    # If successful response, parse the text and find the hyperlinks. 
    # If no valid response, display status code.
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a')
        
        # Extract the href attributes
        urls = [link.get('href') for link in links if link.get('href')]
        
        return urls
    else:
        print("Failed to retrieve the webpage. Status code:", response.status_code)
        return []

def getDate(text):
    ''' 
    Returns the date extract from a given string. This is hard coded and works
    specifically for urls of the format:
    
    https://www.debates.org/voter-education/debate-transcripts/september-29-2020-debate-transcript/
    
    where the date is embedded into the url and is in the form month-DD-YYYY
    '''
    # Define regex pattern for numbers and months
    pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|\d+)\b'
    
    # find matches
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # join all the matches together
    result = ' '.join(matches)

    return result

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

def formatDate(date):

    months = {
    'January': '01',
    'February': '02',
    'March': '03',
    'April': '04',
    'May': '05',
    'June': '06',
    'July': '07',
    'August': '08',
    'September': '09',
    'October': '10',
    'November': '11',
    'December': '12'}

    month,day,year = date.split(' ')
    
    if months.get(month):
        return year + '-' + months.get(month) + '-' + day
    else:
        print("invalid date")

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