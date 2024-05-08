import pickle
from pymongo import MongoClient

def load_vectorizer(db_url, db_name, collection_name, name):
    client = MongoClient(db_url)
    db = client[db_name]
    collection = db[collection_name]

    vectorizer_doc = collection.find_one({'name': name})

    if vectorizer_doc:
        vectorizer = pickle.loads(vectorizer_doc['vectorizer_blob'])
        return vectorizer
    else:
        return None