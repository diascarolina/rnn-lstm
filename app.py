import streamlit as st
from src.rnn import RNN


def app():
    st.sidebar.title("RNNs e LSTMs")

    option = st.sidebar.selectbox(
        "Navegue por uma Opção",
        ["Entendendo RNNs",
         "Entendendo LSTMs",
         "Análise de Sentimento Textual com LSTMs"]
    )

    match option:
        case "Entendendo RNNs":
            st.title(option)
            st.write("""
            ### \"Nossa, como eu queria conseguir prever os preços da bolsa de valores e ficar rico com isso!\"
            \n
            ### \"Já sei, vou criar uma rede neural pra isso!\"
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

            st.write(
                "Como toda rede neural, a RNN também precisa ter seus pesos e bias inicializados.")

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

            st.header("Testando a RNN")

            data_input = st.text_area("Digite os dados de treino separados por vírgula:",
                                      "0.4, 0.5, 0.6, 0.7, 0.8")

            try:
                data = [float(item) for item in data_input.split(',')]
                rnn = RNN(input_size=1, hidden_size=10, output_size=1, learning_rate=0.001)
                if st.button('Prever Valor'):
                    rnn.train(data, epochs=1000)
                    prediction = rnn.predict_next(data[-10:])
                    st.write("Valor previsto:", prediction)

            except ValueError:
                st.write("Dados inválidos, digite apenas valores separados por vírgula.")

            st.header("O Problema.")

            st.subheader("Vanishing/Exploding Gradient")

            st.write("""
            Redes Neurais Recorrentes (RNNs) são conhecidas por terem problemas durante o
            treinamento devido aos gradientes que podem desaparecer ("vanish") ou
            explodir ("explode"). Este problema é amplamente observado quando a rede é treinada
            usando o método de Backpropagation Through Time (BPTT).
            
            1. **Vanishing Gradient (Gradiente Desvanecendo/"Sumindo")**:
               - Quando RNNs são treinadas, os gradientes dos passos de tempo posteriores são
               multiplicados por vários pesos enquanto são propagados para os passos de tempo 
               anteriores.
               - Se esses pesos são pequenos (menores que 1), multiplicar repetidamente por eles
               durante a propagação dos gradientes fará com que os gradientes se tornem 
               extremamente pequenos.
               - Como resultado, os pesos da rede não se atualizam efetivamente, fazendo com que
               a rede não aprenda as dependências de longo prazo entre as sequências.
               - Esse fenômeno é particularmente problemático em sequências mais longas (por 
               exemplo, muitos dias sendo usados para prever a bolsa de valores).
            
            2. **Exploding Gradient (Gradiente "Explodindo")**:
               - Por outro lado, se os pesos são grandes (maiores que 1),
               multiplicar os gradientes por eles repetidamente durante a propagação fará com
               que os gradientes cresçam exponencialmente.
               - Estes gradientes muito grandes podem levar a pesos que se atualizam de forma
               extremamente volátil, causando instabilidade na rede.
               - Muitas vezes isso pode ser observado com valores nulos nos pesos ou na função de 
               perda durante o treinamento.
            
            **Soluções**:
            - Para o problema de gradientes "explodindo", um remédio comum é a técnica de "clipping 
            gradients", onde gradientes que excedem um certo valor são cortados para limitar seu valor.
            - O problema de gradientes desaparecendo é mais difícil de solucionar, mas uma das
            abordagens é usar arquiteturas diferentes, como LSTM (Long Short-Term Memory)
            ou GRU (Gated Recurrent Units), que são projetadas especificamente para lidar com
            dependências de longo prazo ao prevenir contra o problema de vanishing gradient.
            """)

        case "Entendendo LSTMs":
            st.title(option)

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

        case "Análise de Sentimento Textual com LSTMs":
            st.title(option)

    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text(" ")
    st.sidebar.text("""
    Criado por
    Carolina Dias &
    Claúdio Fortier
    2023
    """)


if __name__ == "__main__":
    app()
