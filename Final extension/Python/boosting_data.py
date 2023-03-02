import pandas as pd
from sklearn.model_selection import train_test_split
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

with open('BoostingClassifier.pkl', 'rb') as file:
    stack_model = pickle.load(file)

# Make predictions on the testing set
y_pred = stack_model.predict(X_test)

# Evaluate the model performance
accuracy = (y_pred == y_test).mean()
print(f"Accuracy of boosting classifier: {accuracy:.5f}")

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# set labels for the axes
labels = ['True','False']

# plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)

# set title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# show plot
plt.show()

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

check=input("Enter any key to exit: ")