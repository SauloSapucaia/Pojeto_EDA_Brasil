# 🇧🇷 Inteligência Sócio-Econômica dos Municípios Brasileiros: De EDA a Machine Learning

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

## 📊 Sobre o Projeto

O Brasil é um país de dimensões continentais, composto por mais de 5.570 municípios. Este projeto nasceu do desafio de extrair inteligência de um vasto conjunto de dados abertos (`BRAZIL_CITIES_REV2022.CSV`), contendo dezenas de indicadores demográficos, econômicos, de infraestrutura e agropecuários.

Mais do que apenas responder perguntas de negócio, o objetivo central deste repositório é demonstrar a **evolução analítica**: partindo de uma Análise Exploratória de Dados (EDA) tradicional e escalando para a aplicação de algoritmos de Machine Learning Não Supervisionado e Preditivo.

---

## 🚀 A Evolução do Estudo

Este repositório documenta duas fases distintas da minha jornada como profissional de dados:

### 📍 Fase 1: Análise Exploratória e Descritiva (O Desafio Original)
O foco inicial foi realizar o saneamento dos dados, cruzamento de bases (agregação de Regiões) e responder a perguntas estratégicas de negócio:
* Identificação de disparidades populacionais e concentração de estrangeiros.
* Análise da distribuição de renda (PIB per capita) e contribuição tributária por Estado e Cidade.
* Dimensionamento da força do Agronegócio (Área Plantada vs. Tratores).
* **Estatística Inferencial:** Aplicação de Regressão Linear Simples (OLS com `statsmodels`) para validar hipóteses estatísticas sobre a relação entre PIB, IDHM, Frota de Veículos e número de empresas.

### 🧠 Fase 2: Ciência de Dados e Inteligência Analítica (A Evolução)
Para extrair padrões ocultos que gráficos de barras não conseguem mostrar, o estudo foi refatorado com uma mentalidade avançada de Data Science:

1. **Feature Engineering (Engenharia de Variáveis):** Criação de novas métricas relativas que contam histórias mais precisas, como *Densidade Empreendedora* (empresas por habitante), *Bancarização* e *Força do Agro per Capita*.
2. **O Paradoxo da Riqueza:** Uma análise crítica cruzando as fatias de dados para descobrir cidades que geram riqueza maciça (Alto PIB), mas que sofrem com baixíssimos níveis de Educação e Longevidade.
3. **Machine Learning - Clusterização (K-Means):** Aplicação de IA não supervisionada para agrupar municípios não pela sua geografia (Norte/Sul), mas sim pelo seu **comportamento econômico**. Descoberta de perfis como "Potências do Agronegócio" e "Cidades Dormitório".
4. **Machine Learning - Preditivo (Random Forest Regressor):** Uso do algoritmo de Floresta Aleatória para calcular o *Feature Importance* e responder matematicamente: *Quais fatores estruturais realmente impactam o IDH de uma cidade brasileira?*
5. **Inteligência Geoespacial:** Mapeamento interativo dos Clusters Econômicos criados pela IA utilizando a biblioteca Plotly, revelando manchas de desenvolvimento pelo país.

---

## 🛠️ Stack Tecnológica

* **Linguagem:** Python (v 3.9+)
* **Manipulação e Limpeza de Dados:** `pandas`, `numpy`
* **Estatística e Regressão:** `scipy.stats`, `statsmodels`
* **Machine Learning:** `scikit-learn` (K-Means, RandomForest, StandardScaler)
* **Visualização de Dados:** `matplotlib`, `seaborn`, `plotly`

---

## 💡 Principais Insights e Conclusões

* **A Distribuição de Renda não é Linear:** Modelos OLS revelaram que, embora exista correlação entre PIB per capita e IDHM, o crescimento econômico puro explica apenas uma fração da qualidade de vida.
* **Fronteiras Invisíveis:** O algoritmo de Clusterização provou que municípios com forte atuação agropecuária no Centro-Oeste possuem assinaturas estruturais quase idênticas a cidades do interior do Sul, independentemente da distância geográfica.
* **O Peso da Infraestrutura:** O modelo Random Forest apontou de forma contundente quais serviços e estruturas têm o maior peso preditivo para alavancar o Índice de Desenvolvimento Humano de uma região.

---

## 👨‍💻 Autor

**Saulo Sapucaia** Analista de Dados
Conecte-se comigo no [LinkedIn](https://www.linkedin.com/in/saulo-sapucaia-2821401b9) ou conheça mais do meu trabalho no meu [Portfólio Interativo](https://saulosapucaia.github.io/).
