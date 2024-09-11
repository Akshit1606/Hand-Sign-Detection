import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Convert lists to arrays
data = data_dict['data']
labels = data_dict['labels']

# Determine the maximum length of sequences
max_length = max(len(sample) for sample in data)

# Pad sequences to ensure consistent length
data_padded = pad_sequences(data, maxlen=max_length, padding='post', dtype='float32')
labels = np.array(labels)

print(f"Padded data shape: {data_padded.shape}")

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(data_padded, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)
print(f'{score * 100:.2f}% of samples were classified correctly!')

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
