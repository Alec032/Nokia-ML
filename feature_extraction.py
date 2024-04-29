import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

porter_stemmer = PorterStemmer()

def process(text):
    tokens = nltk.word_tokenize(text)
    stopwords_list = set(stopwords.words('english'))
    punctuation_pattern = re.compile(r'[^\w\s]')
    filtered_tokens = [porter_stemmer.stem(token) for token in tokens if token.lower() not in stopwords_list and not punctuation_pattern.match(token)]
    return filtered_tokens

def extract_features(documents):
    features_list = []
    for document in documents:
        title = document.get('title', '')
        description = document.get('description', '')
        
        title_tokens = process(title)
        description_tokens = process(description)
        tokens = description_tokens + title_tokens

        features_list.append(' '.join(tokens))
    
    return features_list

def extract_other(documents):
    other_list = []
    for document in documents:
        build = document.get('build', '')
        feature = document.get('feature', '')
        release = document.get('release', '')
        
        build_tokens = process(build)
        feature_tokens = process(feature)
        release_tokens = process(' '.join(release))
        tokens = build_tokens + feature_tokens + release_tokens

        other_list.append(' '.join(tokens))
    
    return other_list