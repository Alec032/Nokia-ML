import pickle
import gzip
from pymongo import MongoClient

def load_encoder(client_uri, db_name, collection_name, encoder_name):
    client = MongoClient(client_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    encoder_data = collection.find_one({'encoder_name': encoder_name})
    
    if encoder_data is None:
        return None
    
    compressed_encoder = encoder_data['encoder_data']
    serialized_encoder = gzip.decompress(compressed_encoder)
    label_encoder = pickle.loads(serialized_encoder)
    
    return label_encoder