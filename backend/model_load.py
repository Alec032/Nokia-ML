import pickle
from pymongo import MongoClient
import gzip

def load_model(client_uri, db_name, collection_name, model_name):
    client = MongoClient(client_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    model_data = collection.find_one({'model_name': model_name})
    
    if model_data is None:
        raise ValueError("Model not found in the collection")
    
    compressed_model = model_data['model_data']
    
    serialized_model = gzip.decompress(compressed_model)
    
    model = pickle.loads(serialized_model)
    
    return model