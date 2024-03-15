import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load data from pickle file
with open("E:/CPV/data.pickle", 'rb') as file:
    data_dict = pickle.load(file)

# Assuming data_dict['data'] is a list of lists
data = [np.array(i) for i in data_dict['data']]

labels = np.array(data_dict['labels'])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Assuming x_train is a list of lists
x_train = [np.array(i) for i in x_train]

# Initialize and train the Random Forest model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict labels for test data
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

# Print accuracy
print('{}% of samples were classified correctly!'.format(score * 100))

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump(model, f)
