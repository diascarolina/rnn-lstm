import streamlit as st
import base64
import plotly.graph_objects as go


def app():
    st.title("Entendendo Redes Neurais Recorrentes (RNNs)")

    option = st.sidebar.selectbox(
                "Escolha uma Opção",
                ["Entendendo RNNs",
                 "Prevendo a Bolsa de Valores com RNNs",
                 "Entendendo LSTMs",
                 "Prevendo a Bolsa de Valores com LSTMs",
                 "Análise de Sentimento Textual com LSTMs"]
            )

    match option:
        case "Entendendo RNNs":
            st.write("""
            \"Nossa, como eu queria conseguir prever os preços da bolsa de valores e ficar rico com isso!\"
            \n
            \"Já sei, vou criar uma rede neural pra isso!\"
            """)

            st.write("""
            Os dados históricos da bolsa de valores são sequenciais e podem representar uma quantidade 
            grande de dias (quanto mais, melhor).
            
            Olhando pras redes neurais que vimos anteriormente, elas recebem um número fixo de inputs.
            No caso da CNN para imagens, ela recebe como input uma imagem de 32x32 pixels e nada 
            diferente disso.
            
            Mas no exemplo da bolsa de valores podemos ter dados de 10 dias, ou de 100 dias ou de qualquer
            número de dias. Como isso funcionaria na nossa rede neural?
            """)

            st.write("""
            É aí que entra em ação as Redes Neurais Recorrentes (RNNs)!!!
            Elas possuem pesos, bias, camadas e funções de ativação como as outras redes neurais.
            Mas também possuem "feedback loop" que torna possível o uso de dados sequenciais de diferentes
            tamanhos.
            
            Para isso, a rede cria cópias de si mesma, uma para cada input. E é a combinação dessas 
            cópias que nos dá um resultado final da predição.*
            """)

            st.image("images/rnn.png", caption="RNN e suas 'cópias'", use_column_width=True)

            st.write("""
            *Essas 'cópias', nada mais são do que a própria rede neural sendo recalculada 
            utilizando os valores anteriores.
            """)

            st.header("Como isso funciona na prática?")

            st.write("Vamos criar então uma classe para a RNN:")

            st.code("class RedeNeuralRecorrente:")

            st.write("Como toda rede neural, a RNN também precisa ter seus pesos e bias inicializados.")

            st.code("""
                def __init__(self, tamanho_da_entrada, tamanho_das_camadas_ocultas, tamanho_da_saida):
                # Pesos inicializados com valores aleatórios e pequenos
                self.Wxh = np.random.randn(tamanho_das_camadas_ocultas, tamanho_da_entrada) * 0.01
                self.Whh = np.random.randn(tamanho_das_camadas_ocultas, tamanho_das_camadas_ocultas) * 0.01
                self.Why = np.random.randn(tamanho_da_saida, tamanho_das_camadas_ocultas) * 0.01
        
                # Bias inicializados com zero
                self.bias_oculta = np.zeros((tamanho_das_camadas_ocultas, 1))
                self.bias_saida = np.zeros((tamanho_da_saida, 1))
            """)

            st.write("Realizamos então o forward pass.")

            st.code("""
                def forward(self, entrada):
                self.entrada = entrada
                self.oculta = {0: np.zeros((self.Wxh.shape[0], 1))}
        
                saida = {}
                for t in range(1, len(entrada) + 1):
                    self.oculta[t] = tangente_hiperbolica(
                        np.dot(self.Wxh, entrada[t - 1]) +
                        np.dot(self.Whh, self.oculta[t - 1]) +  # Feedback Loop
                        self.bias_oculta
                    )
                    y_pred = sigmoide(
                        np.dot(self.Why, self.oculta[t]) +
                        self.bias_saida
                    )
                    saida[t] = y_pred
        
                return saida
            """)

            st.write("E o backward pass.")

            st.code("""
                def backward(self, saida, targets, taxa_de_aprendizado=0.001):
                # Inicializa os gradientes (as derivadas)  e os bias com zero
                d_Wxh = np.zeros_like(self.Wxh)
                d_Whh = np.zeros_like(self.Whh)
                d_Why = np.zeros_like(self.Why)
                d_bias_oculta = np.zeros_like(self.bias_oculta)
                d_bias_saida = np.zeros_like(self.bias_saida)
                d_oculta_proxima = np.zeros_like(self.oculta[0])
        
                # Realiza a retropropagação
                for t in reversed(range(1, len(saida) + 1)):
                    d_saida = saida[t] - targets[t - 1]
                    d_Why += np.dot(d_saida, self.oculta[t].T)
                    d_bias_saida += d_saida
                    d_oculta = np.dot(self.Why.T, d_saida) + d_oculta_proxima
                    d_tanh = (1 - self.oculta[t] * self.oculta[t]) * d_oculta
                    d_bias_oculta += d_tanh
                    d_Wxh += np.dot(d_tanh, self.entrada[t - 1].T)
                    d_Whh += np.dot(d_tanh, self.oculta[t - 1].T)
                    d_oculta_proxima = np.dot(self.Whh.T, d_tanh)
        
                # Esse passo evita que os gradientes explodam
                # Limita os gradientes no intervalo [-5, 5]
                for d_param in [d_Wxh, d_Whh, d_Why, d_bias_oculta, d_bias_saida]:
                    np.clip(d_param, -5, 5, out=d_param)
        
                # Atualiza os pesos e os bias
                self.Wxh -= taxa_de_aprendizado * d_Wxh
                self.Whh -= taxa_de_aprendizado * d_Whh
                self.Why -= taxa_de_aprendizado * d_Why
                self.bias_oculta -= taxa_de_aprendizado * d_bias_oculta
                self.bias_saida -= taxa_de_aprendizado * d_bias_saida
            """)

            st.write("""
            Realizando a predição de um valor utilizando apenas o valor anterior
            """)

            st.code("""
            # Dados de exemplo
            dados_de_entrada = [np.array([[i]]) for i in [0.5, 0.6, 0.7, 0.8]]
            targets = [np.array([[i]]) for i in [0.6, 0.7, 0.8, 0.9]]
        
            # Cria a RNN
            rnn = RedeNeuralRecorrente(tamanho_da_entrada=1,
                                       tamanho_das_camadas_ocultas=10,
                                       tamanho_da_saida=1)
        
            # Treina a RNN
            epocas = 10000
            for epoca in range(epocas):
                saidas = rnn.forward(dados_de_entrada)
                rnn.backward(saidas, targets)
        
            # Realiza a predição
            saidas = rnn.forward(dados_de_entrada)
            print('Valor esperado Valor Obtido')
            for i, target in enumerate(targets):
                print(f'{target[0][0]} {[o[0][0] for o in saidas.values()][i]}')
            """)

            st.write("Resultado:")

            st.code("""
            Valor esperado		Valor Obtido
            0.6 			0.7498644409833674   
            0.7 			0.7499937329132549
            0.8 			0.7501007049013371
            0.9 			0.7502071266961721
            """)

            st.header("O Problema.")

            st.subheader("Vanishing/Exploding Gradient")

            st.write("""
            [explicar aqui]
            """)

        case "Prevendo a Bolsa de Valores com RNNs":
            pass

        case "Entendendo LSTMs":

            st.write("""
            Para mitigar o problema do vanishing/exploding gradients, surgiram as LSTMs...
            """)

            st.write("""
            Aqui não temos apenas uma memória, como é o caso das RNNS.
                
            Temos a memória de longo prazo e a memória de curto prazo (daí o nome da rede neural).
            """)

            st.image("images/lstm.png", caption="Long Short-Term Memory (LSTM)",
                     use_column_width=True)

            st.subheader("Em código:")

            st.code("""
            
            """)

        case "Prevendo a Bolsa de Valores com LSTMs":
            pass

        case "Análise de Sentimento Textual com LSTMs":



if __name__ == "__main__":
    app()




#
#     elif option == "Data Preparation":
#         st.header("Data Preparation")
#         st.write("1. Tokenize the text data.")
#         st.write("2. Pad sequences to ensure uniform length.")
#         st.code("""
#         tokenizer = Tokenizer(num_words=5000)
#         tokenizer.fit_on_texts(texts)
#         sequences = tokenizer.texts_to_sequences(texts)
#         data = pad_sequences(sequences, maxlen=100)
#         """, language='python')
#         # Sample chart for token distribution
#         token_distr = {
#             "love": 120, "bad": 50, "great": 80, "good": 110,
#         }
#         fig = go.Figure(data=[go.Bar(x=list(token_distr.keys()), y=list(token_distr.values()))])
#         fig.update_layout(title="Sample Token Distribution")
#         st.plotly_chart(fig)
#
#     elif option == "LSTM Model":
#         st.header("LSTM Model Building")
#         st.write("Construct the LSTM model:")
#         st.write("1. Add an embedding layer.")
#         st.write("2. Add one or more LSTM layers.")
#         st.write("3. Add a dense layer for prediction.")
#         st.code("""
#         model = Sequential()
#         model.add(Embedding(input_dim=5000, output_dim=embedding_dim, input_length=100))
#         model.add(LSTM(32, dropout=0.2, recurrent_dropout=0.2))
#         model.add(Dense(1, activation='sigmoid'))
#         """, language='python')
#         # Sample LSTM cell diagram
#         st.image("path_to_lstm_diagram.png", caption="LSTM Cell Diagram", use_column_width=True)
#
#     elif option == "Training":
#         st.header("Training")
#         st.write("Train the LSTM model using backpropagation through time.")
#         st.code("""
#         model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         model.fit(data, labels, epochs=10, validation_split=0.2, batch_size=32)
#         """, language='python')
#         # Sample training loss chart
#         epochs = list(range(1, 11))
#         training_loss = [0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.28, 0.25, 0.22, 0.2]
#         validation_loss = [0.55, 0.51, 0.48, 0.43, 0.39, 0.37, 0.34, 0.32, 0.3, 0.29]
#         fig = go.Figure()
#         fig.add_trace(go.Scatter(x=epochs, y=training_loss, mode='lines+markers', name='Training Loss'))
#         fig.add_trace(go.Scatter(x=epochs, y=validation_loss, mode='lines+markers', name='Validation Loss'))
#         fig.update_layout(title="Training vs Validation Loss over Epochs")
#         st.plotly_chart(fig)
#
#     elif option == "Prediction":
#         st.header("Prediction")
#         st.write("Use the trained LSTM model to make predictions on new data.")
#         st.code("""
#         predictions = model.predict(new_data)
#         """, language='python')
#         # Sample predictions chart
#         texts = ["I love this", "Feels bad", "So good"]
#         sentiment = [0.85, 0.2, 0.8]
#         fig = go.Figure(data=[go.Bar(x=texts, y=sentiment)])
#         fig.update_layout(title="Sample Sentiment Predictions")
#         st.plotly_chart(fig)
#
#         """### gif from local file"""
#         file_ = open("media/videos/main/480p15/CreateCircle_ManimCE_v0.17.3.gif", "rb")
#         contents = file_.read()
#         data_url = base64.b64encode(contents).decode("utf-8")
#         file_.close()
#
#         st.markdown(
#             f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
#             unsafe_allow_html=True,
#         )
#
# # Textbox for user to enter their data points
# data_input = st.text_area("Enter your data points (comma-separated):", '0.5, 0.6, 0.7, 0.8, 0.85')
#
# try:
#     # Convert string input to list of floats
#     data = [float(item) for item in data_input.split(',')]
#
#     # Train the LSTM if the data has changed
#     if len(data) > 1:
#         inputs = [data[i] for i in range(len(data) - 1)]
#         targets = [data[i + 1] for i in range(len(data) - 1)]
#
#         for _ in range(iterations):
#             for inp, target in zip(inputs, targets):
#                 lstm.forward(np.array([[inp]]))
#                 lstm.backward(np.array([[target]]), learning_rate)
#
#         predicted = lstm.forward(np.array([[val] for val in data[:-1]]))
#         st.write(f"Predicted next value: {predicted[len(data) - 2][0][0]}")
#
#     else:
#         st.write("Please input more data points.")
#
# except ValueError:
#     st.write("Invalid input. Please enter comma-separated numbers only.")
#
# # Run the app
# if __name__ == '__main__':
#     st.button('Predict Next Value')
