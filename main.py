from json_operations import get_documents
from user_input import process_input
from feature_extraction import extract_features, extract_other
from tfidf_operations import prepare
from scipy.sparse import load_npz
from model_load import load_model
import numpy as np
import os

client = 'mongodb://localhost:27017/'
db_name = 'model_db'
collection_name = 'MultinomialNB_model_final'
model_name = 'MultinomialNB'

def main():
    choice = input("Enter '1' to input data or '2' to read from a JSON file: ")
    
    if choice == '1':
        process_input()
        prepare("features.txt", "other_features.txt", "output.npz")
    elif choice == '2':
        json_file_path = input("Enter the path to the JSON file: ")
        documents = get_documents(json_file_path)
        features = extract_features(documents)
        other_features = extract_other(documents)
        
        with open("features.txt", "w") as features_file:
            features_file.write("\n".join(features))
        
        with open("other_features.txt", "w") as other_features_file:
            other_features_file.write("\n".join(other_features))
        
        prepare("features.txt", "other_features.txt", "output.npz")
    else:
        print("Invalid choice. Please enter '1' or '2'.")
    
    model = load_model(client, db_name, collection_name, model_name)
    if model:
        data = load_npz("output.npz")
        predictions = model.predict_proba(data)
                
        for prediction in predictions:
            max_index = np.argmax(prediction)
            if max_index == 0:
                print("NOT_BOAM:", "{:.2f}%".format(prediction[max_index] * 100))
            elif max_index == 1:
                print("BOAM:", "{:.2f}%".format(prediction[max_index] * 100))
            else:
                print("Invalid prediction value.")
    else:
        print("Failed to load model.")
        
    os.remove("features.txt")
    os.remove("other_features.txt")
    os.remove("output.npz")

if __name__ == "__main__":
    main()