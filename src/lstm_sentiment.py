from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping

# Load IMDB dataset and preprocess data
vocab_size = 20000  # Increase vocabulary size
max_len = 500       # Increase sequence length
embedding_dim = 128

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

# Build the LSTM model
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))  # Use bidirectional LSTM
model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)))                           # Another bidirectional LSTM layer
model.add(Dense(64, activation='relu'))                                                          # Additional Dense layer
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Add early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stop])

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test Accuracy:', test_acc)

# Save the entire model
model.save('improved_sentiment_analysis_model.h5')

# # Load IMDB dataset and preprocess data
# vocab_size = 10000
# max_len = 300
# embedding_dim = 128
#
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
# x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
# x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')
#
# # Constrói o modelo
# model = Sequential()
# model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
# model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
# model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.5))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
# # Treina o modelo
# model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
#
# # Salva o modelo
# model.save("model.h5")
#
# # Avalia o modelo
# test_loss, test_acc = model.evaluate(x_test, y_test)
# print('Test Accuracy:', test_acc)
#
if __name__ == "__main__":
    sample_review = "The movie was perfect."
    sample_review_encoded = [imdb.get_word_index()[word] if word in imdb.get_word_index() and imdb.get_word_index()[word] < vocab_size else 0 for word in sample_review.split()]
    sample_review_encoded = pad_sequences([sample_review_encoded], maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(sample_review_encoded)
    sentiment = "positive" if prediction >= 0.5 else "negative"
    print(f"Sample review sentiment: {sentiment}")

