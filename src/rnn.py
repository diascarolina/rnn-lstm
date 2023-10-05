import numpy as np
from src.utils import sigmoide, tangente_hiperbolica


class RedeNeuralRecorrente:
    def __init__(self, tamanho_da_entrada, tamanho_das_camadas_ocultas, tamanho_da_saida):
        # Pesos inicializados com valores aleatórios e pequenos
        self.Wxh = np.random.randn(tamanho_das_camadas_ocultas, tamanho_da_entrada) * 0.01
        self.Whh = np.random.randn(tamanho_das_camadas_ocultas, tamanho_das_camadas_ocultas) * 0.01
        self.Why = np.random.randn(tamanho_da_saida, tamanho_das_camadas_ocultas) * 0.01

        # Bias inicializados com zero
        self.bias_oculta = np.zeros((tamanho_das_camadas_ocultas, 1))
        self.bias_saida = np.zeros((tamanho_da_saida, 1))

    def forward(self, entrada):
        self.entrada = entrada
        self.oculta = {0: np.zeros((self.Wxh.shape[0], 1))}

        saida = {}
        for t in range(1, len(entrada) + 1):
            self.oculta[t] = tangente_hiperbolica(
                np.dot(self.Wxh, entrada[t - 1]) +
                np.dot(self.Whh, self.oculta[t - 1]) +
                self.bias_oculta
            )
            y_pred = sigmoide(
                np.dot(self.Why, self.oculta[t]) +
                self.bias_saida
            )
            saida[t] = y_pred

        return saida

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


if __name__ == "__main__":
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
    print('Valor esperado\t\tValor Obtido')
    for i, target in enumerate(targets):
        print(f'{target[0][0]} \t\t\t\t {[o[0][0] for o in saidas.values()][i]}')

#
# # Mudando o tamanho da entrada
# dados = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# entrada = [np.array([[dados[i], dados[i+1]]]).T for i in range(len(dados) - 2)]
# targets = [np.array([[dados[i+2]]]) for i in range(len(dados) - 2)]
#
# # Inicializando a RNN
# rnn = RedeNeuralRecorrente(tamanho_da_entrada=2,
#                            tamanho_das_camadas_ocultas=10,
#                            tamanho_da_saida=1)
#
# # Treina a RNN
# epocas = 10000
# for epoca in range(epocas):
#     saidas = rnn.forward(dados)
#     rnn.backward(saidas, targets)
#
#
# def preve_proximo_dia(rnn, dados_de_dois_dias):
#     inputs = np.array([dados_de_dois_dias]).T
#     outputs = rnn.forward([inputs])
#     return outputs[len(outputs)][0][0]
#
#
# # Prevê o terceiro dia baseado nos dois dias anteriores
# dados_de_dois_dias = [0.5, 0.6]
# previsao_do_terceiro_dias = preve_proximo_dia(rnn, dados_de_dois_dias)
# print(previsao_do_terceiro_dias)
