#%% Importação de pacotes
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc as sklearn_auc
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import io
from scipy.interpolate import UnivariateSpline # curva sigmoide suavizada
from statstests.process import stepwise # procedimento Stepwise
from scipy import stats # estatística chi2
import plotly.graph_objects as go # gráficos 3D
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from statsmodels.discrete.discrete_model import MNLogit # estimação do modelo
                                                        #logístico multinomial
                                                        

#%% Carregamento de dados
df = pd.read_csv('Python_M11_support material .csv',delimiter=',')
df

# Características das variáveis do dataset
df.info()

# Estatísticas univariadas
df.describe()


#%% Preparando a "Tabela de Dados"

# 1. Limpeza das colunas que vieram como 'object'
def tratar_moeda(coluna):
    return pd.to_numeric(coluna.str.replace('.', '', regex=False).str.replace(',', '.', regex=False))

df['limite_credito'] = tratar_moeda(df['limite_credito'])
df['valor_transacoes_12m'] = tratar_moeda(df['valor_transacoes_12m'])

# 2. Criando a variável alvo: Bom Pagador
# Na base original, default=1 é o inadimplente. Para nós, 1 será o 'Bom'.
df['bom_pagador'] = 1 - df['default']

# 3. Transformação em Dummies 
# Vamos usar drop_first=True para evitar a armadilha da multicolinearidade
df_final = pd.get_dummies(df, columns=['sexo', 'escolaridade', 'estado_civil', 'salario_anual'], drop_first=True)

# 4. Verificação final
#print(df_final[['limite_credito', 'valor_transacoes_12m', 'bom_pagador']].info())

#%% Tabela de frequências absolutas da variável 'bom_pagador'

# O value_counts() mostra a quantidade em cada categoria
# O sort_index() garante que o 0 (Inadimplente) venha antes do 1 (Bom Pagador)
frequencia_absoluta = df_final['bom_pagador'].value_counts().sort_index()

print("--- Tabela de Frequências Absolutas ---")
print(frequencia_absoluta)

#%% Tabela de frequências relativas (Percentual)

frequencia_relativa = df_final['bom_pagador'].value_counts(normalize=True).sort_index() * 100

print("\n--- Tabela de Frequências Relativas (%) ---")
print(frequencia_relativa)

#%%  Estimação do modelo logístico binário
# O '~' separa a variável dependente (esquerda) das explicativas (direita)

modelo_bom_pagador = sm.Logit.from_formula('bom_pagador ~ idade + dependentes + \
                                           limite_credito + valor_transacoes_12m + \
                                           qtd_transacoes_12m + meses_inativo_12m', 
                                           df_final).fit()

# Exibindo os parâmetros básicos
print(modelo_bom_pagador.summary()) #Repare que não é necessario fazer stepwise, pois P > |z|$ para todas as variáveis (idade, dependentes, limite, etc.) já era menor que 0,05.


#%% Outputs detalhados pela função 'summary_col'

print(summary_col([modelo_bom_pagador],
            model_names=["MODELO CRÉDITO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        }))
#%% Fazendo predições para o modelo 'Bom pagador'

modelo_bom_pagador.predict(pd.DataFrame({'idade':[40], 'dependentes':[2], 
                                'limite_credito':[10000], 'valor_transacoes_12m':[5000], 
                                'qtd_transacoes_12m':[80], 'meses_inativo_12m':[1]}))

#%%Atribuindo uma coluna no dataframe para os resultados

# O método predict() sem argumentos calcula a probabilidade para cada linha da base original
df_final['phat'] = modelo_bom_pagador.predict()

# Visualizando as primeiras linhas com a nova coluna de probabilidade
print("--- Base com Probabilidades (phat) ---")
print(df_final[['bom_pagador', 'phat']].head())

#%% Construção da Sigmoide

# 1. Definindo o tamanho da figura
plt.figure(figsize=(15, 10))

# 2. Plotando os pontos REAIS (Y=0 e Y=1)
# Usando 'qtd_transacoes_12m' como exemplo de variável explicativa no Eixo X
sns.scatterplot(x=df_final['qtd_transacoes_12m'][df_final['bom_pagador'] == 0],
                y=df_final['bom_pagador'][df_final['bom_pagador'] == 0],
                color='magenta', alpha=0.5, s=200, label='Inadimplente (0)')

sns.scatterplot(x=df_final['qtd_transacoes_12m'][df_final['bom_pagador'] == 1],
                y=df_final['bom_pagador'][df_final['bom_pagador'] == 1],
                color='springgreen', alpha=0.5, s=200, label='Bom Pagador (1)')

# 3. Plotando a CURVA LOGÍSTICA (Sigmoide)
# O regplot com logistic=True desenha a curva de probabilidade média
sns.regplot(x=df_final['qtd_transacoes_12m'], 
            y=df_final['bom_pagador'],
            logistic=True, ci=None, scatter=False,
            line_kws={'color': 'indigo', 'linewidth': 5})

# 4. Linha de Cutoff (Ponto de decisão em 0.5)
plt.axhline(y = 0.5, color = 'grey', linestyle = ':', label='Cutoff (0.5)')

# 5. Formatação de eixos e legendas (Padrão MBA USP)
plt.title('Curva Sigmoide: Probabilidade de Bom Pagador vs. Qtd Transações', fontsize=22)
plt.xlabel('Quantidade de Transações (12 meses)', fontsize=18)
plt.ylabel('Probabilidade Prevista (phat)', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.legend(fontsize=15, loc='center right')
plt.grid(True, alpha=0.2)

plt.show()


#%% Sigmoide: Probabilidade vs. Transações

plt.figure(figsize=(15,10))

# Plotando a relação entre a variável explicativa e a probabilidade prevista (phat)
sns.regplot(x=df_final['qtd_transacoes_12m'], 
            y=df_final['phat'],
            ci=None, 
            logistic=True,
            scatter_kws={'color':'orange', 's':150, 'alpha':0.4},
            line_kws={'color':'darkorchid', 'linewidth':5})

# Linha de Cutoff (Ponto de decisão 0.5)
plt.axhline(y = 0.5, color = 'grey', linestyle = ':', label='Cutoff 0.5')

# Formatação Estética (Padrão USP ESALQ)
plt.title('Probabilidade de Bom Pagador por Volume de Transações', fontsize=22)
plt.xlabel('Quantidade de Transações (12 meses)', fontsize=20)
plt.ylabel('Probabilidade Prevista (phat)', fontsize=20)

# Ajuste automático dos ticks do eixo X baseado nos seus dados
plt.xticks(fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)

plt.grid(True, alpha=0.2)
plt.show()

#%%Função para Matriz de Confusão e Indicadores

from sklearn.metrics import confusion_matrix, accuracy_score, \
    ConfusionMatrixDisplay, recall_score

def matriz_confusao(predicts, observado, cutoff):
    
    # Criando a predição binária com base no cutoff (ponto de corte)
    predicao_binaria = [1 if item >= cutoff else 0 for item in predicts]
           
    # Gerando a matriz de confusão visual
    cm = confusion_matrix(observado, predicao_binaria)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Inadimplente', 'Bom Pagador'])
    
    # Plotagem seguindo o padrão estético do manual
    plt.figure(figsize=(8,6))
    disp.plot(cmap='Blues', ax=plt.gca())
    plt.title(f'Matriz de Confusão (Cutoff: {cutoff})', fontsize=15)
    plt.xlabel('Predição', fontsize=12)
    plt.ylabel('Realidade', fontsize=12)
    plt.show()
        
    # Calculando os indicadores técnicos
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Organizando em um DataFrame para o relatório do TCC
    indicadores = pd.DataFrame({
        'Sensitividade': [sensitividade],
        'Especificidade': [especificidade],
        'Acurácia': [acuracia]
    })
    
    return indicadores


#%% Gerando os resultados para o TCC

# Chamando a função para o seu modelo de crédito
resultados_modelo = matriz_confusao(predicts=df_final['phat'], 
                                    observado=df_final['bom_pagador'], 
                                    cutoff=0.83)

print("\n--- Indicadores de Performance do Modelo ---")
print(resultados_modelo)

#%% Localizando o Cutoff de Equilíbrio

def espec_sens(observado, predicts):
    # Criando o range de cutoffs de 0 a 1 (passo de 0.01)
    cutoffs = np.arange(0, 1.01, 0.01)
    
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        # Predição binária baseada no cutoff atual
        predicao_binaria = [1 if item >= cutoff else 0 for item in predicts]
        
        # Cálculo dos indicadores usando as funções do sklearn
        sens = recall_score(observado, predicao_binaria, pos_label=1)
        espec = recall_score(observado, predicao_binaria, pos_label=0)
        
        lista_sensitividade.append(sens)
        lista_especificidade.append(espec)
        
    # Gerando o DataFrame de resultados
    resultado = pd.DataFrame({
        'cutoffs': cutoffs,
        'sensitividade': lista_sensitividade,
        'especificidade': lista_especificidade
    })
    
    return resultado

# Executando a função para o  modelo
df_cutoffs = espec_sens(observado=df_final['bom_pagador'], 
                        predicts=df_final['phat'])

print("--- Primeiras linhas da tabela de Cutoffs ---")
print(df_cutoffs.head())

#%% Gráfico de Cruzamento (Sensitividade vs. Especificidade)

plt.figure(figsize=(15,10))

# Plotando as duas linhas (Padrão visual do Manual)
plt.plot(df_cutoffs.cutoffs, df_cutoffs.sensitividade, 
         marker='o', color='indigo', markersize=8, label='Sensitividade')

plt.plot(df_cutoffs.cutoffs, df_cutoffs.especificidade, 
         marker='o', color='darkorange', markersize=8, label='Especificidade')

# Estética e Títulos
plt.title('Cruzamento de Sensitividade e Especificidade', fontsize=22)
plt.xlabel('Cutoff', fontsize=20)
plt.ylabel('Indicadores', fontsize=20)

# Ajuste de escala para facilitar a leitura do cruzamento
plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)

plt.grid(True, alpha=0.3)
plt.legend(fontsize=20, loc='center right')

plt.show()


#%% Construção da Curva ROC e GINI

from sklearn.metrics import roc_curve, auc

# 1. Calculando os pontos da curva (FPR e TPR)
# fpr = 1 - especificidade | tpr = sensitividade
fpr, tpr, thresholds = roc_curve(df_final['bom_pagador'], df_final['phat'])
roc_auc = auc(fpr, tpr)

# 2. Cálculo do coeficiente de GINI
# Transforma a área da AUC em uma escala de 0 a 1
gini = (roc_auc - 0.5) / 0.5

# 3. Plotando a curva ROC 
plt.figure(figsize=(15,10))

# Linha do Modelo 
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=8, linewidth=3, label='Modelo Logístico')

# Linha de Base / Aleatória 
plt.plot(fpr, fpr, color='gray', linestyle='dashed', label='Modelo Aleatório (AUC = 0.5)')

# Título com os indicadores de performance
plt.title('Área abaixo da curva (AUC): %g' % round(roc_auc, 4) + 
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)

plt.xlabel('1 - Especificidade (FPR)', fontsize=20)
plt.ylabel('Sensitividade (TPR)', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(fontsize=15, loc='lower right')
plt.grid(True, alpha=0.2)

plt.show()

#%%  Cálculo das Razões de Chance (Odds Ratio)

# Criando a tabela com os coeficientes e exponenciais
tabela_odds = pd.DataFrame(modelo_bom_pagador.params, columns=['Coeficiente'])
tabela_odds['Odds Ratio (Exp(B))'] = np.exp(modelo_bom_pagador.params)

print("--- Interpretação das Razões de Chance ---")
print(tabela_odds)
