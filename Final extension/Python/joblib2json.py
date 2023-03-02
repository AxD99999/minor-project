import json
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the joblib model
model = joblib.load("Boost.joblib")

# Convert the joblib model to a dictionary
model_dict = model.__dict__

# Convert the dictionary to a JSON string
model_json = json.dumps(model_dict, default=str)

# Write the JSON string to a file
with open("Boost.json", "w") as f:
    f.write(model_json)
