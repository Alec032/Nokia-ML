import json

def get_documents(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return [data] if isinstance(data, dict) else data