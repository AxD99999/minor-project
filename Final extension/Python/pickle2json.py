import json
import pickle
import numpy as np
from xgboost import XGBClassifier

# Load the pickled model
with open("BoostingClassifier.pkl", "rb") as f:
    model = pickle.load(f)

# Define a custom encoder
class XGBClassifierEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, XGBClassifier):
            obj_dict = obj.__dict__.copy()
            booster = obj_dict.pop('booster')
            obj_dict['booster'] = booster.save_raw()
            return obj_dict
        return json.JSONEncoder.default(self, obj)

# Convert the pickled model to a dictionary with NumPy arrays converted to lists
model_dict = {}
for key, value in model.__dict__.items():
    if isinstance(value, np.ndarray):
        model_dict[key] = value.tolist()
    else:
        model_dict[key] = value

# Convert the dictionary to a JSON string using the custom encoder
model_json = json.dumps(model_dict, cls=XGBClassifierEncoder)

# Write the JSON string to a file
with open("BoostingClassifier.json", "w") as f:
    f.write(model_json)
