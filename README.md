# Regressao-Logistica

💳 Credit Scoring Model: Previsão de Bons Pagadores
Este repositório contém o projeto de modelagem preditiva desenvolvido para o MBA em Data Science e Analytics (USP/ESALQ). O projeto utiliza Regressão Logística Binária para classificar clientes de uma instituição financeira, permitindo identificar o perfil de "Bom Pagador" com base em comportamentos transacionais e dados demográficos.

📊 Performance do Modelo
Os resultados obtidos demonstram uma alta capacidade discriminatória, situando o modelo em patamares de excelência para o mercado financeiro:

AUC (Área Sob a Curva ROC): 0.8571

Coeficiente de Gini: 0.7142

Acurácia Global: 75.54% (ajustada pelo Cutoff de Equilíbrio)

🛠️ Tecnologias e Bibliotecas
Linguagem: Python 3.x

Análise Estatística: statsmodels

Machine Learning: scikit-learn

Manipulação de Dados: pandas, numpy

Visualização: matplotlib, seaborn

Procedimentos Auxiliares: statstests (Stepwise)

📖 Metodologia e Funcionalidades
Tratamento de Dados: Conversão de strings monetárias para formato numérico e criação de variáveis dummies (One-Hot Encoding) com tratamento de multicolinearidade.

Modelagem Logística: Estimação por Máxima Verossimilhança com análise detalhada de p-valores e significância estatística.

Interpretação de Negócio: Cálculo de Odds Ratio para quantificar o impacto de cada variável na chance de adimplência.

Otimização de Decisão: Implementação de função para localização do Cutoff de Equilíbrio entre Sensitividade e Especificidade.

Validação: Geração de Matriz de Confusão, Curva Sigmoide e Curva ROC.

💡 Insights Extraídos
Engajamento: Para cada transação adicional realizada no ano, a chance de o cliente ser um bom pagador aumenta em 10%.

Risco de Abandono: Cada mês de inatividade reduz a chance de adimplência em 39%, sendo o principal sinal de alerta.

Maturidade: Clientes mais velhos apresentam maior probabilidade de honrar compromissos financeiros.

📂 Estrutura do Código
O script está organizado em blocos funcionais (#%%), facilitando a execução modular em ambientes como VS Code ou Spyder:

Carregamento e Limpeza: Preparação da base de dados.

Estimação: Criação do modelo logístico.

Visualização: Gráficos da Sigmoide e Cruzamento de indicadores.

Performance: Matriz de Confusão e métricas de erro.
