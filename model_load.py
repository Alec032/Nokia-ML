import numpy as np
import pickle
from pymongo import MongoClient

def load_model(client, db_name, collection_name, model_name):
    try:
        client = MongoClient(client)
        db = client[db_name]
        collection = db[collection_name]
        model_data = collection.find_one({"name": model_name})
        model = pickle.loads(model_data["model"])
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None