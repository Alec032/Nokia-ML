from scipy.sparse import save_npz
from vectorizer_load import load_vectorizer

db_url = 'mongodb://localhost:27017/'
db_name = 'vectorizer_db'
collection_name = 'TFIDF_vectorizer'
model_name = 'TFIDF'

vectorizer = load_vectorizer(db_url, db_name, collection_name, model_name)

def prepare(features_corpus, other_features_corpus, output_file):
    combined_corpus = []
    min_length = min(len(features_corpus), len(other_features_corpus))
    
    for i in range(min_length):
        combined_corpus.append(f"{features_corpus[i]} {other_features_corpus[i]}")
        
    combined_corpus.extend(features_corpus[min_length:])
    combined_corpus.extend(other_features_corpus[min_length:])
    combined_features = vectorizer.transform(combined_corpus)
    
    save_npz(output_file, combined_features)