# SAD ENCCEJA - Sistema de Apoio à Tomada de Decisão Pedagógica (KNN)

Este é um **Sistema de Apoio à Tomada de Decisão (SAD)** pedagógica que utiliza o algoritmo de Machine Learning **K-Nearest Neighbors (K-NN)** para prever o desempenho de novos alunos matriculados em um cursinho preparatório para o **ENCCEJA (Ensino Médio)**. 

O sistema realiza previsões cruzando características socioeconômicas dos estudantes contra uma base de dados real extraída dos microdados oficiais do **INEP (2024)**, gerando diagnósticos automatizados e diretrizes estratégicas para o gestor escolar.

---

## Estrutura do Repositório para Executar o programa

```
├── sistema_encceja.py                   # Código-fonte principal da aplicação
├── MICRODADOS_ENCCEJA_2024_REG_NAC.csv  # Base de dados oficial do INEP (deve ser baixada localmente)
```
---

## Funcionalidades

- **Processamento dos dados via ETL (Chunksize):** Varredura inteligente de arquivos massivos do governo (mais de 830 mil linhas) em blocos, aplicando filtros de higienização de dados em tempo real.
- **Predição Multivariável Ativa:** Previsão simultânea das notas para as **5 disciplinas** do exame:
  - Ciências Humanas (`CH`)
  - Matemática (`MAT`)
  - Linguagens, Códigos e suas Tecnologias (`LC`)
  - Ciências da Natureza (`CN`)
  - Prova de Redação (`RED`)
- **Mapeamento Ativo de Renda Familiar:** Integração do questionário socioeconômico do INEP através da variável de renda familiar (`Q26`), convertendo alternativas qualitativas em pesos geométricos para o algoritmo.
- **Painel Gerencial Visual (Tkinter):** 
  - Exibição de notas previstas com **Formatação Condicional Dinâmica** (Verde Escuro para notas acima do corte e Vermelho Escuro para notas em zona de risco).
  - **Tabela de Boletins Individuais:** Exibição detalhada do histórico dos 5 vizinhos mais próximos localizados no espaço vetorial, destacando linhas completas em verde (Aprovados) ou vermelho (Reprovados).
- **Árvore de Decisão Pedagógica:** Geração automática de relatórios de texto direcionados, sugerindo oficinas de redação, turmas EAD/noturnas ou reforços específicos baseados no perfil e na contagem de dominância dos vizinhos.

---

## Como o Algoritmo Funciona (Abordagem Matemática)

O sistema utiliza o `KNeighborsRegressor` implementado via `scikit-learn`:

1. **Normalização Z-Score (`StandardScaler`):** Como os atributos possuem escalas diferentes (Ex: Faixa Etária varia de 1 a 15, enquanto Sexo varia de 0 a 1), o sistema centraliza e normaliza os dados para garantir que todas as variáveis socioeconômicas possuam o mesmo peso no cálculo da Distância Euclidiana.
2. **Localização por Proximidade ($K=5$):** Ao cadastrar um novo aluno, o algoritmo mapeia o seu perfil e localiza as 5 instâncias (estudantes históricos do INEP) geometricamente mais idênticas no plano multidimensional.
3. **Predição por Regressão:** A nota estimada para o candidato na interface gráfica é calculada através da **média aritmética** das notas reais obtidas por esses 5 vizinhos históricos em cada uma das competências.

---

## Tecnologias Utilizadas

- **Python 3.13+**
- **Tkinter / ttk:** Construção da interface gráfica (GUI).
- **Pandas & NumPy:** Engenharia, manipulação de dados e processamento em fluxo (*streaming*).
- **Scikit-Learn:** Implementação do modelo preditivo KNN e padronização vetorial.
- **Os & Pathlib:** Manipulação de caminhos e arquivos locais de forma dinâmica.
