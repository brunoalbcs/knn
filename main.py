import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

from knn_sem_normalizacao import SimpleKNN
from knn_com_normalizacao import NormalizedKNN


def metricas_completas(y_real, y_pred, classe=0):
    Tp = np.sum((y_real == classe) & (y_pred == classe))
    Fp = np.sum((y_real != classe) & (y_pred == classe))
    Fn = np.sum((y_real == classe) & (y_pred != classe))

    p = Tp / (Tp + Fp) if (Tp + Fp) > 0 else 0
    r = Tp / (Tp + Fn) if (Tp + Fn) > 0 else 0
    f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

    return p, r, f1


if __name__ == "__main__":

    print("======================================================================")
    print("TESTE 1: KNN BASICO SEM NORMALIZACAO DOS DADOS (Breast Cancer Dataset)")
    print("======================================================================")
    data_cancer = datasets.load_breast_cancer()
    X_c, y_c = data_cancer['data'][:, :5], data_cancer['target']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_c, y_c, test_size=0.3, random_state=42)

    knn_basico = SimpleKNN(k=5)
    knn_basico.fit(Xc_train, yc_train)
    yc_pred = knn_basico.predict(Xc_test, distance_metric='l2')
    acuracia_c = np.mean(yc_pred == yc_test)
    print(f"Taxa de acerto (K=5, Euclidiana): {(acuracia_c * 100):.2f}%\n")

    print("--- MATRIZ DE CONFUSAO (BREAST CANCER) ---")
    print(pd.crosstab(yc_test, yc_pred, rownames=['Real'], colnames=['Predito']))
    print("\n")

    print("==================================================")
    print("TESTE 2: KNN NORMALIZADO COM VALIDACAO (Wine Dataset)")
    print("==================================================")
    data_wine = datasets.load_wine()
    X_w, y_w = data_wine['data'][:, :3], data_wine['target']

    Xw_train, Xw_temp, yw_train, yw_temp = train_test_split(X_w, y_w, test_size=0.4, random_state=42)
    Xw_valid, Xw_test, yw_valid, yw_test = train_test_split(Xw_temp, yw_temp, test_size=0.5, random_state=42)

    historico = []

    for k in (3, 9):
        for distancia in ('l1', 'l2'):
            for tipo_normalizacao in ('z-score', 'min_max'):
                knn_norm = NormalizedKNN(k)
                knn_norm.fit(Xw_train, yw_train, tipo_normalizacao)
                yw_pred = knn_norm.predict(Xw_valid, distancia)
                acuracia_w = np.mean(yw_pred == yw_valid)

                p0, r0, f1_0 = metricas_completas(yw_valid, yw_pred, classe=0)
                p1, r1, f1_1 = metricas_completas(yw_valid, yw_pred, classe=1)
                p2, r2, f1_2 = metricas_completas(yw_valid, yw_pred, classe=2)

                p = (p0 + p1 + p2) / 3
                r = (r0 + r1 + r2) / 3
                f1 = (f1_0 + f1_1 + f1_2) / 3

                print(
                    f"K={k} | Distancia={distancia} | Norm={tipo_normalizacao} -> Acuracia na Validacao: {(acuracia_w * 100):.2f}%")

                historico.append({
                    'K': k,
                    'Distancia': distancia,
                    'Normalizacao': tipo_normalizacao,
                    'Acuracia': acuracia_w,
                    'F1': f1
                })

    resultados = pd.DataFrame(historico)
    grafico_barras = sns.catplot(
        data=resultados, x='K', y='Acuracia', hue='Normalizacao', col='Distancia',
        kind='bar', height=5, aspect=1.2, palette='viridis'
    )
    grafico_barras.fig.subplots_adjust(top=0.85)
    grafico_barras.fig.suptitle('Impacto dos Hiperparametros na Validacao - Wine', fontsize=16)

    for ax in grafico_barras.axes.flat:
        ax.set_ylim(0.7, 0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()

    print("\n==================================================")
    print("TESTE FINAL COM MELHORES PARAMETROS (WINE DATASET)")
    print("==================================================")

    melhores_parametros = max(historico, key=lambda x: x['F1'])
    k_opt = melhores_parametros['K']
    dist_opt = melhores_parametros['Distancia']
    norm_opt = melhores_parametros['Normalizacao']

    knn_vencedor = NormalizedKNN(k_opt)
    knn_vencedor.fit(Xw_train, yw_train, norm_opt)
    y_pred_final = knn_vencedor.predict(Xw_test, dist_opt)
    acc_final = np.mean(y_pred_final == yw_test)

    p0, r0, f1_0 = metricas_completas(yw_test, y_pred_final, classe=0)
    p1, r1, f1_1 = metricas_completas(yw_test, y_pred_final, classe=1)
    p2, r2, f1_2 = metricas_completas(yw_test, y_pred_final, classe=2)

    p_final = (p0 + p1 + p2) / 3
    r_final = (r0 + r1 + r2) / 3
    f1_final = (f1_0 + f1_1 + f1_2) / 3

    print(f"Configuracao Vencedora: K={k_opt} | Distancia = {dist_opt} | Normalizacao = {norm_opt}")
    print(f"Taxa de acerto (Acuracia): {(acc_final * 100):.2f}%")
    print(f"Precision (Macro-Average): {(p_final * 100):.2f}%")
    print(f"Recall (Macro-Average): {(r_final * 100):.2f}%")
    print(f"F1-Score (Macro-Average): {(f1_final * 100):.2f}%\n")

    print("--- MATRIZ DE CONFUSAO (WINE) ---")
    print(pd.crosstab(yw_test, y_pred_final, rownames=['Real'], colnames=['Predito']))
    print("\n")