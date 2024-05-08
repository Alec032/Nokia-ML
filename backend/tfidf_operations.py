from scipy.sparse import save_npz
from vectorizer_load import load_vectorizer

db_url = 'mongodb://localhost:27017/'
db_name = 'vectorizer_db'
collection_name = 'TFIDF_vectorizer'
model_name = 'TFIDF'

vectorizer = load_vectorizer(db_url, db_name, collection_name, model_name)

def prepare(features_corpus, other_features_corpus, output_file):
    combined_corpus = [f"{features_corpus[i]} {other_features_corpus[i]}" for i in range(len(features_corpus))]
    combined_features = vectorizer.transform(combined_corpus)
    
    save_npz(output_file, combined_features)