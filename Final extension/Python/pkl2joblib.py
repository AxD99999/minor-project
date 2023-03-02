import joblib
import pickle

# Load the pickled model
with open("Boost.pkl", "rb") as f:
    model = pickle.load(f)

# Save the model as a joblib file
joblib.dump(model, "Boost.joblib")
