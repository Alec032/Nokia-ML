from scipy.sparse import hstack, save_npz
from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_extraction import process
vectorizer = TfidfVectorizer()

def prepare(feature_file, other_feature_file, output_file):
    with open(feature_file, 'r') as file:
        features_corpus = [line.strip() for line in file.readlines()]

    with open(other_feature_file, 'r') as file:
        other_features_corpus = [line.strip() for line in file.readlines()]

    tfidf_features = vectorizer.fit_transform(features_corpus)
    tfidf_others = vectorizer.fit_transform(other_features_corpus)

    combined_features = hstack([tfidf_features, tfidf_others])
    
    save_npz(output_file, combined_features)