# Classificador KNN: Implementacao do Zero e Otimizacao

## Sobre o Projeto
Este repositorio contem o desenvolvimento de um classificador K-Nearest Neighbors (KNN) construido inteiramente do zero, aplicando conceitos de algebra linear e calculos de distancia vetorial. O projeto foi desenvolvido como um estudo ativo para aprofundar o entendimento matematico por tras dos modelos de Machine Learning, sem a utilizacao de bibliotecas prontas de classificacao.

Alem do algoritmo base, foram implementadas matematicamente tecnicas de Feature Scaling e extracao de metricas de avaliacao, testadas frente a diferentes cenarios e conjuntos de dados reais.

## Principais Funcionalidades Implementadas (From Scratch)
* Algoritmo KNN Base: Implementacao nativa com suporte a calculos de distancia Euclidiana (L2) e distancia de Manhattan (L1).
* Feature Scaling Dinamico: Padronizacao e normalizacao de dados embarcadas na classe do modelo (Z-Score, Min-Max e Max).
* Metricas de Avaliacao Manuais: Construcao de matrizes de confusao e calculo de Acuracia, Precision, Recall e F1-Score (incluindo Macro-Average para cenarios multiclasse).
* Pipeline de Validacao: Estruturacao de testes iterativos para otimizacao de hiperparametros (fator K, metrica de distancia e tipo de escala).

## Datasets Analisados
Os testes de validacao foram aplicados em dois cenarios distintos (disponibilizados via Scikit-Learn):
1. Breast Cancer Wisconsin Dataset: Classificacao binaria (Maligno vs. Benigno) operando com 5 features preditivas.
2. Wine Recognition Dataset: Classificacao multiclasse (3 origens geograficas distintas) operando com 3 features preditivas.

## Tecnologias Utilizadas
* NumPy: Manipulacao matricial e operacoes matematicas vetorizadas eficientes.
* Pandas: Estruturacao de dados e geracao de matrizes de confusao.
* Scikit-Learn: Importacao dos datasets e separacao do conjunto de dados (train_test_split).
* Matplotlib & Seaborn: Analise grafica avancada para visualizacao do impacto dos hiperparametros no desempenho do modelo.

## Estrutura do Repositorio
knn/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- main.py                     # Script principal com o pipeline de testes e geracao de graficos
|-- knn_sem_normalizacao.py     # Classe do modelo KNN basico
|-- knn_com_normalizacao.py     # Classe do modelo KNN com algoritmos de escala nativos

## Como Executar o Projeto

1. Clone o repositorio:
git clone https://github.com/SEU_USUARIO/knn-from-scratch.git
cd knn-from-scratch

2. Crie e ative um ambiente virtual (recomendado):
python -m venv venv
# No Windows:
venv\Scripts\activate
# No Linux/Mac:
source venv/bin/activate

3. Instale as dependencias:
pip install -r requirements.txt

4. Execute o arquivo principal para rodar os treinamentos, validacoes e gerar as analises graficas:
python main.py

---
Projeto desenvolvido para aprofundamento teorico e pratico na construcao de modelos preditivos e manipulacao de dados.