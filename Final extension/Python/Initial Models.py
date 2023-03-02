import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

import pickle

# Load the dataset
data = pd.read_csv("dataset.csv")

# Replace -1 labels with 0 in the target variable
data["Result"] = data["Result"].replace(-1, 0)

# Take the first 10 rows of the dataset
#data = data.iloc[:10, :]

# Split the data into training and testing sets
X = data.drop("Result", axis=1)
y = data["Result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the base models
xgb_model = xgb.XGBClassifier()
svm_model = SVC()
rf_model = RandomForestClassifier()
lr_model = LogisticRegression()

# Train the individual models
xgb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

# Make predictions on the testing set for each individual model
y_pred_xgb = xgb_model.predict(X_test)
y_pred_svc = svm_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# Evaluate the accuracy of each individual model
accuracy_xgb = (y_pred_xgb == y_test).mean()
accuracy_svc = (y_pred_svc == y_test).mean()
accuracy_rf = (y_pred_rf == y_test).mean()
accuracy_lr = (y_pred_lr == y_test).mean()

# Print the accuracies of each individual model
print(f"Accuracy of XGBoost: {accuracy_xgb:.5f}")
print(f"Accuracy of SVM: {accuracy_svc:.5f}")
print(f"Accuracy of Random Forest: {accuracy_rf:.5f}")
print(f"Accuracy of Logistic Regression: {accuracy_lr:.5f}")

#-----------------------------------------------------------------------------

from sklearn.ensemble import VotingClassifier


# Define the Voting Classifier
model_voting = VotingClassifier(
    estimators=[
        ("xgb", xgb_model),
        ("svc", svm_model),
        ("rf", rf_model),
        ("lr", lr_model),
    ],
    voting="hard",
)

# Train the Voting Classifier
model_voting.fit(X_train, y_train)

# Make predictions on the testing set using the Voting Classifier
y_pred_voting = model_voting.predict(X_test)

# Evaluate the accuracy of the Voting Classifier
accuracy_voting = (y_pred_voting == y_test).mean()

# Print the accuracy of the Voting Classifier
print(f"Accuracy of Voting Classifier: {accuracy_voting:.5f}")

# Save the model as a pickle file
with open('VotingClassifier.pkl', 'wb') as f:
    pickle.dump(model_voting, f)

#-----------------------------------------------------------------------------

from sklearn.ensemble import StackingClassifier

# Define the stacking classifier
estimators = [('xgb', xgb_model), ('svm', svm_model), ('rf', rf_model), ('lr', lr_model)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=xgb_model)

# Train the stacking classifier
stack_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = stack_model.predict(X_test)

# Evaluate the model performance
accuracy = (y_pred == y_test).mean()
print(f"Accuracy of stacking classifier: {accuracy:.5f}")

# Save the model as a pickle file
with open('StackingClassifier.pkl', 'wb') as f:
    pickle.dump(stack_model, f)
#-----------------------------------------------------------------------------

from sklearn.ensemble import GradientBoostingClassifier

# Define the stacking classifier
estimators = [('xgb', xgb_model), ('svm', svm_model), ('rf', rf_model), ('lr', lr_model)]
stack_model = StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier())

# Train the stacking classifier
stack_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = stack_model.predict(X_test)

# Evaluate the model performance
accuracy = (y_pred == y_test).mean()
print(f"Accuracy of boosting classifier: {accuracy:.5f}")

# Save the model as a pickle file
with open('BoostingClassifier.pkl', 'wb') as f:
    pickle.dump(stack_model, f)

#-----------------------------------------------------------------------------

from sklearn.ensemble import BaggingClassifier

# Define the bagging classifier
estimators = [xgb_model, svm_model, rf_model, lr_model]
bag_model = BaggingClassifier(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, random_state=42)

# Train the bagging classifier
bag_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = bag_model.predict(X_test)

# Evaluate the model performance
accuracy = (y_pred == y_test).mean()
print(f"Accuracy of bagging classifier: {accuracy:.5f}")

# Save the model as a pickle file
with open('BaggingClassifier.pkl', 'wb') as f:
    pickle.dump(bag_model, f)

check=input("Enter any key to exit: ")