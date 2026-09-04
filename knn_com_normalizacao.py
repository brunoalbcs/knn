import numpy as np

class NormalizedKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train, tipo_normalizacao="z-score"):
        # Garantir que a entrada seja um array do numpy
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        # Reescalar features e guardar valores pra reescalar os testes também.
        self.tipo_normalizacao = tipo_normalizacao
        match self.tipo_normalizacao:
            case "z-score":
                self.X_medio = np.mean(self.X_train, axis=0)
                self.X_desvio = np.std(self.X_train, axis=0)
                self.X_train = (self.X_train - self.X_medio) / self.X_desvio

            case "min_max":
                self.X_min = np.min(self.X_train, axis=0)
                self.X_max = np.max(self.X_train, axis=0)
                self.X_train = (self.X_train - self.X_min) / (self.X_max - self.X_min)

            case "max":
                self.X_max = np.max(self.X_train, axis=0)
                self.X_train = self.X_train / self.X_max

            case _:
                raise ValueError('Normalização inválida. Escolha entre "z-score", "min_max" ou "max"')


    def predict(self, X_test, distance_metric='l2'):
        if distance_metric == 'l1':  # Manhattan ou city block
            dist_func = self._l1_distance
        elif distance_metric == 'l2':  # Euclidiana
            dist_func = self._l2_distance
        else:
            raise ValueError("Métrica de distancia inválida. Escolha 'l1' ou 'l2'.")

        # Garantir que X_test será um array do numpy:
        X_test = np.array(X_test)

        predictions = [self._predict(x, dist_func) for x in X_test]
        return np.array(predictions)

    def _predict(self, x, dist_func):

        # Feature Scaling
        match self.tipo_normalizacao:
            case "z-score":
                xteste = (x - self.X_medio) / self.X_desvio

            case "min_max":
                xteste = (x - self.X_min) / (self.X_max - self.X_min)
                xteste = np.clip(xteste, 0, 1)  # Trunca para manter entre 0 e 1

            case "max":
                xteste = x / self.X_max

        # x é um vetor da matriz X_test, então tenho que iterar pegando os vetores (instâncias) de X_train
        # Iterar por cada instância de treino calculando as distâncias entre elas e a instância de teste
        distancias = [dist_func(xteste, xtrain) for xtrain in self.X_train]

        # argsort() retorna os índices que ordenariam a função, pego as k menores distâncias
        indices_ordenados = np.argsort(distancias)
        k_indices = indices_ordenados[0:self.k]

        # agora é só pegar as classes que correspondem a esses índices:
        classes_vizinhas = self.y_train[k_indices]

        # Contagem: unique() retorna a classe e quantas vezes ela aparece. Separo em classes e contagens
        # argmax() pega o índice do maior valor dentro de contagens. Passo esse índice para classes[] e tenho a classe mais contada.
        classes, contagens = np.unique(classes_vizinhas, return_counts=True)
        return classes[np.argmax(contagens)]


    def _l1_distance(self, x1, x2):
        # distancia manhattan
        return np.sum(np.abs(x1 - x2))

    def _l2_distance(self, x1, x2):
        # distancia euclidiana
        return np.sqrt(np.sum((x1-x2)**2))
