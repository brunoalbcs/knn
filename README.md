# Classificador KNN: Implementação do Zero e Otimização

## Sobre o Projeto
Este repositório contém o desenvolvimento de um classificador K-Nearest Neighbors (KNN) construído inteiramente do zero, aplicando conceitos de álgebra linear e cálculos de distância vetorial. O projeto foi desenvolvido como um estudo ativo para aprofundar o entendimento matemático por trás dos modelos de Machine Learning, sem a utilização de bibliotecas prontas de classificação.

Além do algoritmo base, foram implementadas matematicamente técnicas de Feature Scaling e extração de métricas de avaliação, testadas frente a diferentes cenários e conjuntos de dados reais.

## Principais Funcionalidades Implementadas (From Scratch)
* Algoritmo KNN Base: Implementação nativa com suporte a cálculos de distância Euclidiana (L2) e distância de Manhattan (L1).
* Feature Scaling Dinâmico: Padronização e normalização de dados embarcadas na classe do modelo (Z-Score, Min-Max e Max).
* Métricas de Avaliação Manuais: Construção de matrizes de confusão e cálculo de Acurácia, Precision, Recall e F1-Score (incluindo Macro-Average para cenários multiclasse).
* Pipeline de Validação: Estruturação de testes iterativos para otimização de hiperparâmetros (fator K, métrica de distância e tipo de escala).

## Datasets Analisados
Os testes de validação foram aplicados em dois cenários distintos (disponibilizados via Scikit-Learn):
1. Breast Cancer Wisconsin Dataset: Classificação binária (Maligno vs. Benigno) operando com 5 features preditivas.
2. Wine Recognition Dataset: Classificação multiclasse (3 origens geográficas distintas) operando com 3 features preditivas.

## Tecnologias Utilizadas
* NumPy: Manipulação matricial e operações matemáticas vetorizadas eficientes.
* Pandas: Estruturação de dados e geração de matrizes de confusão.
* Scikit-Learn: Importação dos datasets e separação do conjunto de dados (train_test_split).
* Matplotlib & Seaborn: Análise gráfica avançada para visualização do impacto dos hiperparâmetros no desempenho do modelo.

## Estrutura do Repositório
knn/
|-- .gitignore
|-- README.md
|-- requirements.txt
|-- main.py                     # Script principal com o pipeline de testes e geração de gráficos
|-- knn_sem_normalizacao.py     # Classe do modelo KNN básico
|-- knn_com_normalizacao.py     # Classe do modelo KNN com algoritmos de escala nativos

## Como Executar o Projeto

1. Clone o repositório:

    git clone https://github.com/brunoalbcs/knn.git
    
    cd knn

2. Crie e ative um ambiente virtual (recomendado):

    python -m venv venv
    
    ### No Windows:
    venv\Scripts\activate
    
    ### No Linux/Mac:
    source venv/bin/activate

3. Instale as dependências:

    pip install -r requirements.txt

4. Execute o arquivo principal para rodar os treinamentos, validações e gerar as análises gráficas:

    python main.py

---
Projeto desenvolvido para aprofundamento teórico e prático na construção de modelos preditivos e manipulação de dados.
