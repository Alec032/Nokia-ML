import json

def get_documents(json_file_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        return [data] if isinstance(data, dict) else data
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON file: {json_file_path}")
        return None