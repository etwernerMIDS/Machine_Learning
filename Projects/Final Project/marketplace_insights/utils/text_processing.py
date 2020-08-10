import re
import html.parser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup

RE_SUBSTITUTIONS = [
    [r'<[^>]*>', r' '],  # Removing extraneous HTML tags
    [r' +', r' '],
]

def apply_regexes(text):
    for re_sub in RE_SUBSTITUTIONS:
        params = [re_sub[0], re_sub[1], text]
        text = re.sub(*params)
    return text.strip()


def unescape_html(text):
    return html.parser.unescape(text)


def remove_html_tags(text, parser='html.parser'):
    return BeautifulSoup(text, parser).text


def remove_line_breaks(text):
    return re.sub(r'< *br *\/?>', ' ', text)


def preprocess(text, lemmatize=True, reg_pattern='[^A-Za-z]+', stopwords=stopwords.words('english')):
    lemmatizer = WordNetLemmatizer()
    regularizer = re.compile(reg_pattern)

    doc_list = text.lower()
    tokens = regularizer.sub(' ', doc_list).split()       # keep only letters OR numbers and tokenize strings

    stopped_tokens = [i for i in tokens if not i in stopwords]  # remove stop words

    long_tokens = [i for i in stopped_tokens if len(i) >= 2]     # remove single letters
    if lemmatize==True:
        lemmatized = [lemmatizer.lemmatize(i) for i in long_tokens] # lemmatize words
    else:
        lemmatized = long_tokens

    # remove 'xxxx'-like tokens
    cleaned = [word for word in lemmatized if word != len(word) * word[0]]
    return ' '.join(cleaned)

