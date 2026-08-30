import json
import os
from datetime import datetime
import uuid

HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'data', 'history.json')

def init_history_file():
    if not os.path.exists(os.path.dirname(HISTORY_FILE)):
        os.makedirs(os.path.dirname(HISTORY_FILE))
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump({}, f)

def load_all_history():
    init_history_file()
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_all_history(data):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(data, f, indent=4)

def get_user_history(username):
    data = load_all_history()
    if username not in data:
        return []
    
    # Return sorted by date descending (newest first)
    user_history = data[username]
    return sorted(user_history, key=lambda x: x['date'], reverse=True)

def save_analysis(username, headline, article, sentiment, confidence, probabilities, word_count):
    data = load_all_history()
    
    if username not in data:
        data[username] = []
        
    analysis_id = str(uuid.uuid4())
    
    analysis_record = {
        "id": analysis_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "headline": headline,
        "article": article,
        "sentiment": sentiment,
        "confidence": confidence,
        "probabilities": probabilities,
        "word_count": word_count
    }
    
    data[username].append(analysis_record)
    save_all_history(data)
    
    return analysis_id

def delete_analysis(username, analysis_id):
    data = load_all_history()
    if username in data:
        data[username] = [record for record in data[username] if record['id'] != analysis_id]
        save_all_history(data)
