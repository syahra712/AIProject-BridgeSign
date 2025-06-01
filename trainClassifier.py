import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set the correct path to the data.pickle file
DATA_PATH = '/Users/admin/Desktop/American-Sign-language-Detection-System/data.pickle'

# Load the dataset
with open(DATA_PATH, 'rb') as f:
    data_dict = pickle.load(f)

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions and evaluate
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'The accuracy of the RandomForestClassifier model on the test data is: {score * 100:.2f}%')

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)