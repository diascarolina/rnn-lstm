import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Embedding, Dense
from keras.datasets import imdb
from keras.preprocessing import sequence


# Number of words to consider as features
num_words = 5000

# Load the data
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Pad sequences to be of the same length
max_review_length = 500
x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)


model = Sequential()

# Embedding layer to transform integer sequences to dense vectors
model.add(Embedding(num_words, 128))

# LSTM layer with 128 memory units
model.add(LSTM(128))

# Output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=3, batch_size=64)


loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
