from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping

# Definindo as constantes
vocab_size = 20000
max_len = 500
embedding_dim = 128

# Carrega os dados e separa em treino e teste
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')

# Cria o modelo LSTM
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5, return_sequences=True)))
model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.5)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Treina o modelo
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2, callbacks=[early_stop])

# Avalia o modelo
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Acurácia de teste:', test_acc)

# Save the entire model
model.save('improved_sentiment_analysis_model.h5')

if __name__ == "__main__":
    sample_review = "The movie was perfect."
    sample_review_encoded = [imdb.get_word_index()[word] if word in imdb.get_word_index() and
                             imdb.get_word_index()[word] < vocab_size else 0
                             for word in sample_review.split()]
    sample_review_encoded = pad_sequences([sample_review_encoded], maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(sample_review_encoded)
    sentimento = "positivo :)" if prediction >= 0.5 else "negativo :("
    print(f"O sentimento foi {sentimento}")

