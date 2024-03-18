import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load data
data_dict = pickle.load(open("E:\CPV\Handsign\data.pickle", 'rb'))

fixed_length = 42
processed_data = []
for sample in data_dict['data']:
    if len(sample) < fixed_length:
        # Pad with zeros if the length is less than the fixed length
        processed_sample = np.pad(sample, (0, fixed_length - len(sample)), mode='constant')
    else:
        # Truncate if the length is greater than the fixed length
        processed_sample = sample[:fixed_length]
    processed_data.append(processed_sample)

# Convert to NumPy array
data = np.asarray(processed_data)
labels = np.asarray(data_dict['labels'])

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, shuffle=True, stratify=labels)

# Train RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict on test set
y_predict = model.predict(x_test)

# Calculate accuracy
score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly!'.format(score * 100))

# Save the model in the specified directory
pickle_file_path = os.path.join("E:\\CPV\\Dataset", 'model.p')  # Set the model file path in the specified directory
with open(pickle_file_path, 'wb') as f:
    pickle.dump({'model': model}, f)

print("Model file saved to:", pickle_file_path)
