PREDIÇÃO DE CÂNCER DE PULMÃO COM NAIVE BAYES

Este projeto utiliza o algoritmo de aprendizado de máquina Naive Bayes Gaussiano para prever se uma pessoa tem câncer de pulmão com base em características como gênero, idade, hábitos de fumar, entre outros.

DESCRIÇÃO DO PROJETO:
O objetivo do projeto é treinar um modelo de aprendizado de máquina para prever o diagnóstico de câncer de pulmão. O modelo é desenvolvido utilizando a biblioteca scikit-learn e os dados são manipulados com pandas.

FUNCIONALIDADES:
- Divisão dos dados: 80% para treino, 20% para teste
- Algoritmo Naive Bayes Gaussiano para predição
- Cálculo da acurácia do modelo

REQUISITOS:
Antes de rodar o código, certifique-se de que as seguintes bibliotecas estão instaladas:
- pandas
- scikit-learn

INSTALAÇÃO DAS DEPENDÊNCIAS:
1. Crie e ative um ambiente virtual (opcional):
   - No Windows: venv\Scripts\activate
   - No Linux/Mac: source venv/bin/activate

2. Instale as dependências usando o prompt do seu editor de código:
   - pip install pandas scikit-learn

COMO RODAR O CÓDIGO:
1. Abra o terminal ou prompt de comando.
2. Navegue até a pasta onde o script está localizado.
3. Execute o seguinte comando:
   - python cancer_prediction.py

ESTRUTURA DOS DADOS:
Os dados incluem as seguintes colunas:
- Gender (M/F)
- Age (Idade)
- Smoking (Nível de tabagismo)
- Yellow fingers (Dedo amarelado)
- Anxiety (Ansiedade)
- Peer_pressure (Pressão dos amigos)
- Chronic Disease (Doença crônica)
- Fatigue (Fadiga)
- Allergy (Alergia)
- Wheezing (Chiado no peito)
- Alcohol (Consumo de álcool)
- Coughing (Tosse)
- Shortness of Breath (Falta de ar)
- Swallowing Difficulty (Dificuldade de deglutição)
- Chest pain (Dor no peito)
- Lung Cancer (Resultado - Sim/Não)

RESULTADOS:
A acurácia do modelo será exibida após a execução do script.

LICENÇA:
Este projeto está licenciado sob a licença MIT.
