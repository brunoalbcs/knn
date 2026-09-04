import numpy as np

class SimpleKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        # Garantir que a entrada seja um array do numpy para evitar erros futuros:
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)


    def predict(self, X_test, distance_metric='l2'):
        if distance_metric == 'l1':  # Manhattan ou city block
            dist_func = self._l1_distance
        elif distance_metric == 'l2':  # Euclidiana
            dist_func = self._l2_distance
        else:
            raise ValueError("Métrica de distancia inválida. Escolha 'l1' ou 'l2'.")

        # Garantindo que X_test será um array do numpy (boa prática):
        X_test = np.array(X_test)

        predictions = [self._predict(x, dist_func) for x in X_test]
        return np.array(predictions)

    def _predict(self, x, dist_func):
        # Escreva uma função que calcule as distancias entre o ponto de teste e os pontos de treinamento usando as funções de distância especificadas.
        # Nesta função deverá se identificar os k vizinhos mais próximos com base nessas distâncias, e determinar a classe mais comum entre esses vizinhos
        # retornando a classe como a previsão para o ponto de teste.

        # x é um vetor da matriz X_test, então tenho que iterar pegando os vetores (instâncias) de X_train
        distancias = [dist_func(x, xtrain) for xtrain in self.X_train]

        # argsort() retorna os índices que ordenariam a função
        indices_ordenados = np.argsort(distancias)
        k_indices = indices_ordenados[0:self.k]

        # agora é só pegar as classes que correspondem a esses índices:
        classes_vizinhas = self.y_train[k_indices]

        # Contagem: unique() retorna um array com as classes existentes e outro array com quantas vezes cada classe aparece. Separo em classes e contagens
        # argmax() pega o índice do maior valor dentro de contagens. Passo esse índice para classes[] e tenho a classe mais contada.
        classes, contagens = np.unique(classes_vizinhas, return_counts=True)
        return classes[np.argmax(contagens)]


    def _l1_distance(self, x1, x2):
        # Escreva uma função para calcular a distancia manhattan
        return np.sum(np.abs(x1 - x2))

    def _l2_distance(self, x1, x2):
        # Escreva uma função para calcular a distancia euclidiana
        return np.sqrt(np.sum((x1-x2)**2))