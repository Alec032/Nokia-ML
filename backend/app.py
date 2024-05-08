from flask import Flask, jsonify, request
from json_operations import get_documents
from user_input import process_input
from feature_extraction import extract_features, extract_other
from tfidf_operations import prepare
from scipy.sparse import load_npz
from model_load import load_model
from flask_cors import CORS, cross_origin
import numpy as np
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

client = 'mongodb://localhost:27017/'
db_name = 'model_db'
collection_name = 'MultinomialNB_model_final'
model_name = 'MultinomialNB'

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    choice = data.get('choice')
    
    if choice == '1':
        title = data.get('title')
        description = data.get('description')
        build = data.get('build')
        feature = data.get('feature')
        release = data.get('release')
        
        combined_feature_tokens, other_feature_tokens = process_input(title, description, build, feature, release)
        if not combined_feature_tokens or not other_feature_tokens:
            return jsonify({'error': 'Failed to process input'}), 400
        
        data = prepare(combined_feature_tokens, other_feature_tokens)
        
    elif choice == '2':
        json_file_path = data.get('json_file_path')
        documents = get_documents(json_file_path)
        if documents is None:
            return jsonify({'error': 'Unable to read JSON file'}), 400
        
        features = extract_features(documents)
        other_features = extract_other(documents)
        
        data = prepare(features, other_features)
        
    else:
        return jsonify({'error': 'Invalid choice'}), 400
    
    model = load_model(client, db_name, collection_name, model_name)
    if model:
        predictions = model.predict_proba(data)
                
        result = []
        for prediction in predictions:
            max_index = np.argmax(prediction)
            if max_index == 0:
                result.append({"label": "NOT_BOAM", "probability": prediction[max_index]})
            elif max_index == 1:
                result.append({"label": "BOAM", "probability": prediction[max_index]})
            else:
                result.append({"error": "Invalid prediction value"})
        
        return jsonify(result)
    else:
        return jsonify({'error': 'Failed to load model'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1')