from json_operations import get_documents
from user_input import process_input
from feature_extraction import extract_features, extract_other
from tfidf_operations import prepare
from scipy.sparse import load_npz
from model_load import load_model
from flask_cors import CORS
import numpy as np
from flask import Flask, jsonify, request
from load_encoder import load_encoder
import os

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

db_name2 = 'encoder_db'
collection_name2 = 'label_encoders'
encoder_name = 'group_in_charge_encoder'

client = 'mongodb://localhost:27017/'
db_name = 'model_db'
collection_name = 'MultinomialNB_model_final2'
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
    
        output_file = "output.npz"
        prepare(combined_feature_tokens, other_feature_tokens, output_file)
        
    elif choice == '2':
        json_file_path = os.path.normpath(data.get('json_file_path'))
        if not os.path.isfile(json_file_path):
            return jsonify({'error': f'File not found: {json_file_path}'}), 400
        documents = get_documents(json_file_path)
        if documents is None:
             return jsonify({'error': 'Unable to read JSON file'}), 400
        
        features = extract_features(documents)
        other_features = extract_other(documents)
            
        output_file = "output.npz" 
        prepare(features, other_features, output_file)
        
    else:
        return jsonify({'error': 'Invalid choice'}), 400

    data = load_npz(output_file)

    model = load_model(client, db_name, collection_name, model_name)
    label_encoder = load_encoder(client, db_name2, collection_name2, encoder_name)
    if model:
        predictions = model.predict_proba(data)

        result = []
        for prediction in predictions:
            max_index = np.argmax(prediction)
            original_label = label_encoder.inverse_transform([max_index])[0]
            probability = round(prediction[max_index] * 100, 2)
            result.append({"label": original_label, "probability": probability})

        result.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify(result[0])
    else:
        return jsonify({'error': 'Failed to load model'})

if __name__ == '__main__':
    app.run(host='127.0.0.1')