#%%
# Frameworks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
from datetime import datetime as dt
import warnings

# Versao da Linguagem Python
from platform import python_version
print('Versao da Linguagem Python utilizada neste notebook: ', python_version())

# Frameworks de Estatistica
from scipy.stats import skew
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Bibliotecas de Machine Learning e Geoespacial (Visão Sênior)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# Configurações para evitar warnings e criar pastas de imagens
import os
os.makedirs('Imagens_graficos', exist_ok=True)

#%%
# Carregando o dataset
df = pd.read_csv('2_BRAZIL_CITIES_REV2022.CSV')

#%%
# Verificando o formato do dataset
df.shape

#%%
# Verificando as colunas do dataset
df.columns

#%%
# Visualizando as primeiras linhas do dataset
df.head()

#%%
# Visualizando as ultimas linhas do dataset
df.tail()

#%%
# Verificando o tipo de cada coluna e a quantidade de valores nulos
df.info()

#%%
# Preenchendo os valores nulos das colunas de franquias com 0, pois a ausência de dados pode indicar que a franquia não está presente na cidade, e isso é relevante para a análise.
cols_franquias = ['UBER', 'MAC', 'WAL-MART', 'HOTELS', 'BEDS', 'Pr_Bank', 'Pu_Bank']
for col in cols_franquias:
    if col in df.columns:
        df[col] = df[col].fillna(0)

#%%
# Importando uma base auxiliar para trazer a regiao com a maior e menor per capita
regiao = pd.read_excel('Regiao_Brasil.xlsx')

regiao.head()

#%%
# Criando uma copia do dataframe para evitar perder os dados originais
regiao_backup = regiao.copy()

# Renomeando as coluna para fazer o merge
regiao.rename(columns={'UF':'STATE'}, inplace=True)
regiao.rename(columns={'Região':'REGION'}, inplace=True) 
regiao.rename(columns={'Estado':'NAME_STATE'}, inplace=True) 

# Fazendo o merge pela coluna STATE
df = pd.merge(df, regiao, on="STATE", how='inner')

#%%
# Analise Exploratoria de Dados (EDA)
# Verificando se ha valor usente
df.isnull().sum().head(29)

#%%
# Verificando a quantidade de valores nulos por coluna
df.isnull().sum().tail(29)

#%%
# Verificando a quantidade de valores nulos por coluna de forma aleatoria
df.isnull().sum().sample(29)


# Criando uma copia do dataframe para evitar perder os dados originais
df_backup = df.copy()

cols_num = ['GDP_CAPITA', 'IDHM', 'IDHM_Longevidade', 'Cars', 'Motorcycles', 
            'IBGE_RES_POP', 'Wheeled_tractor', 'IBGE_PLANTED_AREA', 'TAXES', 'GDP', 'COMP_TOT']

for col in cols_num:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Filtro crítico: remove cidades com população zero para evitar divisão por zero
df = df[df['IBGE_RES_POP'] > 0].copy()

# Analise Focada nas Resposta do Projeto

#%%
# Q1: Qual Estado com a maior e menor população geral? Mostre também onde estão concentrados a população de Estrangeiros por Estado
# =======================================================
# Filtrando o dataset para analisar a população geral
df_pg = df[['STATE', 'IBGE_RES_POP', 'IBGE_RES_POP_BRAS','IBGE_RES_POP_ESTR']].copy()

# Agrupando os estados e sumando a população
cols_estados = ['IBGE_RES_POP','IBGE_RES_POP_BRAS','IBGE_RES_POP_ESTR']
df_pg = df_pg.groupby('STATE')[cols_estados].sum().reset_index()

# Ordenando os valores
df_pg.sort_values(by='IBGE_RES_POP', ascending=False, inplace=True)
df_pg.reset_index(drop=True, inplace=True)

# Plotando e SALVANDO o Grafico com a população geral por estado
plt.figure(figsize=(16,6))
plt.bar(df_pg['STATE'], df_pg['IBGE_RES_POP'], color='green')
plt.title('Total da População Geral por Estado')
plt.xlabel('Estado')
plt.ylabel('População')
plt.savefig('Imagens_graficos/q1_grafico_pop_geral.png', bbox_inches='tight') 
plt.close()

# Plotando e SALVANDO o Grafico com a população estrangeira por estado
plt.figure(figsize=(16,6))
plt.bar(df_pg['STATE'], df_pg['IBGE_RES_POP_ESTR'], color='gray')
plt.title('Total da População Estrangeira por Estado')
plt.xlabel('Estado')
plt.ylabel('População')
plt.savefig('Imagens_graficos/q1_grafico_pop_estrangeira.png', bbox_inches='tight') 
plt.close()

# Crianda a coluna auxiliar para identificar o percentual de estrangeiros
df_pg['Perc_BRAS'] = round((df_pg['IBGE_RES_POP_BRAS'] / df_pg['IBGE_RES_POP']) * 100, 2)
df_pg['Perc_Estr'] = round((df_pg['IBGE_RES_POP_ESTR'] / df_pg['IBGE_RES_POP']) * 100, 2)

# Criando os objetos para responder a pergunta
df_maior_estado = df_pg[df_pg['IBGE_RES_POP'] == df_pg['IBGE_RES_POP'].max()]
nome_maior_estado = df_maior_estado['STATE'].iloc[0]
maior_ppg = '{:,.0F}'.format(df_maior_estado['IBGE_RES_POP'].iloc[0]).replace(',', '.')
maior_estr = '{:,.0F}'.format(df_maior_estado['IBGE_RES_POP_ESTR'].iloc[0]).replace(',', '.')
maior_perc_estr = '{:,.1F}'.format(df_maior_estado['Perc_Estr'].iloc[0])

df_menor_estado = df_pg[df_pg['IBGE_RES_POP'] == df_pg['IBGE_RES_POP'].min()]
nome_menor_estado = df_menor_estado['STATE'].iloc[0]
menor_ppg = '{:,.0F}'.format(df_menor_estado['IBGE_RES_POP'].iloc[0]).replace(',', '.')

# Guardando a SUA resposta exata na variável para o site
resp_q1 = f'O maior estado é <strong>{nome_maior_estado}</strong> que tem a "População Geral" dentre os estados com <strong>{maior_ppg}</strong> habitantes, com a maior quantidade de estrangeiros no estado de <strong>{maior_estr}</strong> habitantes, representando um percentual de <strong>{maior_perc_estr}%</strong> que é o maior dentre os outros estados.<br><br>Sendo que o menor estado apresentado em "População Geral" é <strong>{nome_menor_estado}</strong> com o total de <strong>{menor_ppg}</strong> habitantes.'

#%%
# Q2: Qual Cidade com a maior e menor população geral?
# =======================================================
# Filtrando o dataset para analisar a população geral
df_city = df[['CITY','STATE','IBGE_RES_POP', 'IBGE_RES_POP_BRAS','IBGE_RES_POP_ESTR']].copy()

# Agrupando as cidades e somando a população
cols_cidade = ['IBGE_RES_POP','IBGE_RES_POP_BRAS','IBGE_RES_POP_ESTR']
df_city = df_city.groupby(['CITY', 'STATE'])[cols_cidade].sum().reset_index()

# Ordenando os valores e ajustando o index
df_city.sort_values(by='IBGE_RES_POP', ascending=False, inplace=True)
df_city.reset_index(drop=True, inplace=True)

# Criando as colunas auxiliares de percentual
df_city['Perc_BRAS'] = round((df_city['IBGE_RES_POP_BRAS'] / df_city['IBGE_RES_POP']) * 100, 2)
df_city['Perc_Estr'] = round((df_city['IBGE_RES_POP_ESTR'] / df_city['IBGE_RES_POP']) * 100, 2)

# Separando as fatias do Top 10
top10_city = df_city.nlargest(10,'IBGE_RES_POP')
top10_city_estr = df_city.nlargest(10,'IBGE_RES_POP_ESTR').reset_index(drop=True)
top10_menor_city = df_city.nsmallest(10,'IBGE_RES_POP').reset_index(drop=True)

# Gráfico 1: Top 10 Cidades com a Maior População Geral
plt.figure(figsize=(16,6))
sns.barplot(data=top10_city, x='CITY', y='IBGE_RES_POP')
plt.title('Top 10 Cidades com a Maior População Geral', fontsize=14, fontweight='bold')
plt.xlabel('Cidades', fontstyle='italic')
plt.ylabel('População', fontstyle='italic')
plt.savefig('Imagens_graficos/q2_grafico_top10_pop_geral.png', bbox_inches='tight') 
plt.close()

# Gráfico 2: Top 10 Cidades com a Maior População Estrangeira
plt.figure(figsize=(16,6))
sns.set_palette('coolwarm')
sns.barplot(data=top10_city_estr, x='CITY', y='IBGE_RES_POP_ESTR')
plt.title('Top 10 Cidades com a Maior População Estrangeira', fontsize=14, fontweight='bold')
plt.xlabel('Cidades', fontstyle='italic')
plt.ylabel('População', fontstyle='italic')
plt.savefig('Imagens_graficos/q2_grafico_top10_pop_estr.png', bbox_inches='tight') 
plt.close()

# Gráfico 3: Top 10 Cidades com a Menor População
plt.figure(figsize=(16,6))
sns.barplot(data=top10_menor_city, x='CITY', y='IBGE_RES_POP')
plt.title('Top 10 Cidades com a Menor População', fontsize=14, fontweight='bold')
plt.xlabel('Cidades', fontstyle='italic')
plt.ylabel('População', fontstyle='italic')
plt.savefig('Imagens_graficos/q2_grafico_top10_menor_pop.png', bbox_inches='tight') 
plt.close()

# Criando os objetos para responder a pergunta
df_maior_city = top10_city.iloc[0]
nome_maior_city = df_maior_city['CITY']
maior_ppg_city = '{:,.0f}'.format(df_maior_city['IBGE_RES_POP']).replace(',', '.')
maior_estr_city = '{:,.0f}'.format(df_maior_city['IBGE_RES_POP_ESTR']).replace(',', '.')
maior_perc_estr_city = '{:,.1f}'.format(df_maior_city['Perc_Estr']).replace('.', ',')

df_menor_cidade = top10_menor_city.iloc[0]
nome_menor_cidade = df_menor_cidade['CITY']
menor_ppg_city = '{:,.0f}'.format(df_menor_cidade['IBGE_RES_POP']).replace(',', '.')

# Formatando a SUA resposta HTML exata para injeção no site
resp_q2 = f'A cidade com a maior "População Geral" é <strong><em>{nome_maior_city}</em></strong>, com <em>{maior_ppg_city} habitantes</em>, tendo também a maior quantidade de estrangeiros na cidade, totalizando <em>{maior_estr_city} habitantes</em>. Isso representa um percentual de <em>{maior_perc_estr_city}%</em> em relação à população total da cidade, o maior entre todas as cidades.<br><br>Por outro lado, a cidade com a menor "População Geral" é <strong><em>{nome_menor_cidade}</em></strong>, totalizando <em>{menor_ppg_city} habitantes</em>.'

#%%
# Q3: Em que região esta concentrado as maiores e menores rendas per capita?
# =======================================================
# Filtrando o dataset para tratar o valor per capita
df_pcapita = df[['CITY','STATE','GDP_CAPITA', 'REGION']].copy()

# Agrupando os valores por regiao e calculando a media
df_pcapita = df_pcapita.groupby('REGION')['GDP_CAPITA'].mean().reset_index()

# Ordenando os valores
df_pcapita.sort_values(by='GDP_CAPITA', ascending=False, inplace=True)
df_pcapita.reset_index(drop=True, inplace=True)

# Ajustando a Coluna GDP_CAPITA para o valor mensal
df_pcapita['GDP_CAPITA'] = df_pcapita['GDP_CAPITA'] / 12

# Criando os objetos para responder a pergunta
# Região com a maior per capita
df_maior_pcapita = df_pcapita.iloc[0]
nome_maior_pcapita = df_maior_pcapita['REGION']
maior_renda_pcapita = '{:,.0f}'.format(df_maior_pcapita['GDP_CAPITA']).replace(',', '.')

# Região com a menor per capita
df_menor_pcapita = df_pcapita.iloc[-1]
nome_menor_pcapita = df_menor_pcapita['REGION']
menor_renda_pcapita = '{:,.0f}'.format(df_menor_pcapita['GDP_CAPITA']).replace(',', '.')

# Formatando a SUA resposta HTML exata para injeção no site
resp_q3 = f'A maior concentração de renda per capita é encontrada na região <strong>{nome_maior_pcapita}</strong>, onde o valor médio é de <strong>R$ {maior_renda_pcapita}</strong>.<br><br>Já a menor renda per capita está concentrada na região <strong>{nome_menor_pcapita}</strong>, com um valor médio de <strong>R$ {menor_renda_pcapita}</strong>.'

# Plotando o Grafico de Pizza
plt.figure(figsize=(10,6))
# Paleta de cores para deixar profissional
colors = sns.color_palette('pastel')[0:5] 

plt.pie(df_pcapita['GDP_CAPITA'],
        labels=df_pcapita['REGION'],
        autopct='%1.1f%%',
        startangle=90,
        colors=colors)
        
# Adicionando o círculo branco no centro para virar "Rosca"
centro_circulo = plt.Circle((0,0), 0.7, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centro_circulo)

plt.title('Média Mensal de Renda Per Capita por Região', fontsize=14, fontweight='bold', fontstyle='italic')
plt.savefig('Imagens_graficos/q3_grafico_renda_per_capita.png', bbox_inches='tight') 
plt.close()


#%%
# Q4: Quais Estados e Cidades tiveram maior e menor contribuição de Impostos?
# =======================================================
# Filtrando o dataset para analisar as cidades e estados pela contribuição imposto
df_imposto = df[['CITY', 'STATE', 'TAXES']].copy()

# Agrupando o Estado para buscar o valor total
df_imposto_estado = df_imposto.groupby('STATE')['TAXES'].sum().reset_index()

# Ordenando os valores
df_imposto.sort_values(by='TAXES', ascending=False, inplace=True)
df_imposto_estado.sort_values(by='TAXES', ascending=False, inplace=True)

# Ajustando o index
df_imposto.reset_index(drop=True, inplace=True)
df_imposto_estado.reset_index(drop=True, inplace=True)

# Plotando e SALVANDO o Histograma
plt.figure(figsize=(12, 5))
sns.histplot(df_imposto_estado['TAXES'], kde=True, color='blue', label='STATE')
plt.title('Histograma de Contribuição de Impostos por Estado')
plt.savefig('Imagens_graficos/q4_histograma_impostos.png', bbox_inches='tight') 
plt.close()

# Plotando e SALVANDO o grafico de Boxplot
plt.figure(figsize=(18, 6))
sns.boxplot(data=df_imposto, x='STATE', y='TAXES')
plt.title('Distribuição de Contribuição de Impostos por Estado')
plt.xlabel('Estado', fontstyle='italic')
plt.ylabel('Imposto', fontstyle='italic')
plt.savefig('Imagens_graficos/q4_boxplot_impostos.png', bbox_inches='tight') 
plt.close()

# Criando os objetos para responder a pergunta (Estados)
df_maior_imposto_estado = df_imposto_estado.iloc[0]
nome_maior_imposto_estado = df_maior_imposto_estado['STATE']
maior_renda_imposto_estado = '{:,.0f}'.format(df_maior_imposto_estado['TAXES']).replace(',', '.')

df_menor_imposto_estado = df_imposto_estado.iloc[-1]
nome_menor_imposto_estado = df_menor_imposto_estado['STATE']
menor_renda_imposto_estado = '{:,.0f}'.format(df_menor_imposto_estado['TAXES']).replace(',', '.')

# Criando os objetos para responder a pergunta (Cidades)
df_maior_imposto_cidade = df_imposto.iloc[0]
nome_maior_imposto_cidade = df_maior_imposto_cidade['CITY']
maior_renda_imposto_cidade = '{:,.0f}'.format(df_maior_imposto_cidade['TAXES']).replace(',', '.')

df_menor_imposto_cidade = df_imposto.iloc[-1]
nome_menor_imposto_cidade = df_menor_imposto_cidade['CITY']
menor_renda_imposto_cidade = '{:,.0f}'.format(df_menor_imposto_cidade['TAXES']).replace(',', '.')

# Formatando a SUA resposta exata para injeção no HTML
resp_q4 = f'A maior contribuição de imposto é encontrada no estado de <strong>"{nome_maior_imposto_estado}"</strong>, com um valor de <em>R$ {maior_renda_imposto_estado}</em>.<br>Por outro lado, a menor contribuição de imposto está no estado de <strong>"{nome_menor_imposto_estado}"</strong>, totalizando <em>R$ {menor_renda_imposto_estado}</em>.<br><br>Em relação às cidades, a maior contribuição de imposto é da cidade de <strong>"{nome_maior_imposto_cidade}"</strong>, que arrecadou <em>R$ {maior_renda_imposto_cidade}</em>.<br>Já a menor contribuição de imposto foi registrada na cidade de <strong>"{nome_menor_imposto_cidade}"</strong>, com um valor de <em>R$ {menor_renda_imposto_cidade}</em>.'

#%%
# Q5: Qual Estado possui maior variação entre a população real e a população estimada?
# =======================================================
# Filtrando o dataset para analisar a população real (usando df e .copy() para evitar warnings)
df_ppl_real = df[['STATE', 'IBGE_RES_POP', 'ESTIMATED_POP']].copy()

# Calcular a variação absoluta entre a população real e estimada
df_ppl_real['VARIATION'] = abs(df_ppl_real['IBGE_RES_POP'] - df_ppl_real['ESTIMATED_POP'])

# Agrupar por estado e somar a variação
df_estado_var = df_ppl_real.groupby('STATE')['VARIATION'].sum().sort_values(ascending=False).reset_index()

# 1. Plotando e SALVANDO o gráfico de dispersão
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_ppl_real, x='IBGE_RES_POP', y='ESTIMATED_POP')
plt.title('Comparação entre População Real e Estimada')
plt.xlabel('População Real')
plt.ylabel('População Estimada')
plt.savefig('Imagens_graficos/q5_scatter_pop.png', bbox_inches='tight') 
plt.close()

# 2. Plotando e SALVANDO o Boxplot
plt.figure(figsize=(7, 9))
sns.boxplot(data=df_ppl_real[['IBGE_RES_POP', 'ESTIMATED_POP']])
plt.title('Boxplot das Populações Real e Estimada')
plt.savefig('Imagens_graficos/q5_boxplot_pop.png', bbox_inches='tight') 
plt.close()

# 3. Plotando e SALVANDO o gráfico de barras
plt.figure(figsize=(12, 6))
sns.barplot(data=df_estado_var, x='STATE', y='VARIATION')
plt.title('Variação entre População Real e Estimada por Estado')
plt.xlabel('Estado')
plt.ylabel('Variação')
plt.savefig('Imagens_graficos/q5_barplot_var.png', bbox_inches='tight') 
plt.close()

# Criando os objetos para responder a pergunta
df_maior_maior_var = df_estado_var.iloc[0]
nome_estado_maior_var = df_maior_maior_var['STATE']
estado_maior_var = '{:,.0f}'.format(df_maior_maior_var['VARIATION']).replace(',', '.')

# Formatando a SUA resposta HTML exata para injeção no site
resp_q5 = f'O estado com maior variação entre a população real (censo) e a população estimada é o <strong>"{nome_estado_maior_var}"</strong>, apresentando uma distorção total de <strong>{estado_maior_var}</strong> habitantes.'

#%%
# Q6: Mostre a região (pode ser Estado, Cidade, Região, etc.) que possui maior Area por quilometro quadrado (dica: usar gráficos de dispersão com os parâmetros da Longitude e Latitude)
# =======================================================
# Filtrando o dataset para analisar a area por km²
df_area = df[['REGION', 'STATE', 'CITY', 'AREA', 'LONG', 'LAT', 'ALT']].copy()

# Calcular o total de Area km^2 de cada regiao
regiao_area = df_area.groupby('REGION')['AREA'].sum().reset_index()

# Identificacao de regiao com a maior area km²
area_regiao_max = regiao_area.loc[regiao_area['AREA'].idxmax()]

# Criando os objetos para responder a pergunta
nome_maior_area = area_regiao_max["REGION"]
area_total_maior_regiao = '{:,.0f}'.format(area_regiao_max["AREA"]).replace(',', '.')

# Limpando do dataset Longitude, Latitude e Altitude zeradas
df_area_plt = df_area[(df_area['LONG'] != 0) & (df_area['LAT'] != 0) & (df_area['ALT'] != 0)]

# Plotando e SALVANDO o grafico de dispersao
plt.figure(figsize=(7,7))
sns.scatterplot(data=df_area_plt,
                x='LONG',
                y='LAT',
                hue='REGION',
                size='AREA',
                sizes=(20,200),
                legend=False)

max_region_data = df_area_plt[df_area_plt['REGION'] == area_regiao_max['REGION']]
plt.scatter(max_region_data['LONG'], max_region_data['LAT'], color='green', label=f'Maior Área: {area_regiao_max["REGION"]}')

plt.title('Dispersão da Area km² por Regiões (Longitude x Latitude)', fontsize=12, fontweight='bold')
plt.xlabel('Longitude', fontstyle='italic')
plt.ylabel('Latitude', fontstyle='italic')
plt.legend(fontsize=9, bbox_to_anchor=(0.40, 0.073))
plt.savefig('Imagens_graficos/q6_scatter_area.png', bbox_inches='tight') 
plt.close()

# Formatando a SUA resposta HTML exata para injeção no site
resp_q6 = f'A região com a maior área por quilômetro quadrado é a <strong>"{nome_maior_area}"</strong> com uma área total de <strong>{area_total_maior_regiao} km²</strong>.'

#%%
# Q7: Qual Região possui maior e menor Area Plantada?
# =======================================================
# Filtrando o dataset para analisar a area plantada
df_area_plantada = df[['REGION','IBGE_PLANTED_AREA']].copy()

# Calcular o total de Area de cada regiao
regiao_area_plantada = df_area_plantada.groupby('REGION')['IBGE_PLANTED_AREA'].sum().sort_values(ascending=False).reset_index()

# Extraindo as variáveis e formatando os números para leitura fácil
df_maior_area_plantada = regiao_area_plantada.iloc[0]
nome_maior_area_plantada = df_maior_area_plantada['REGION']
maior_renda_area_plantada = '{:,.0f}'.format(df_maior_area_plantada['IBGE_PLANTED_AREA']).replace(',', '.')

df_menor_area_plantada = regiao_area_plantada.iloc[-1]
nome_menor_area_plantada = df_menor_area_plantada['REGION']
menor_renda_area_plantada = '{:,.0f}'.format(df_menor_area_plantada['IBGE_PLANTED_AREA']).replace(',', '.')

# Plotando e SALVANDO o grafico com maior area
plt.figure(figsize=(8,6))
sns.barplot(data=regiao_area_plantada,
            y='REGION', 
            x='IBGE_PLANTED_AREA',
            orient='h')

plt.title('Total de Área Plantada por Região', fontsize=14, fontweight='bold')
plt.xlabel('Área Plantada', fontstyle='italic')
plt.ylabel('Região', fontstyle='italic')
plt.savefig('Imagens_graficos/q7_barplot_area_plantada.png', bbox_inches='tight') 
plt.close()

# Formatando a SUA resposta HTML exata para injeção no site
resp_q7 = f'A região com a maior área plantada é a <strong>"{nome_maior_area_plantada}"</strong> com uma área de <strong>{maior_renda_area_plantada}</strong> (unidades de medida).<br><br>Por outro lado, a região com a menor área plantada é o <strong>"{nome_menor_area_plantada}"</strong> com uma área de <strong>{menor_renda_area_plantada}</strong>.'

#%%
# Q8: Onde estão concentradas as pessoas com maior e menor IDH?
# =======================================================
# Filtrando o dataset para analisar o IDH
df_idh = df[['STATE','NAME_STATE','REGION', 'CITY', 'IDHM','LONG', 'LAT', 'ALT']].copy()

# Calcular a mediana do IDH por estado
df_idh_estado = (df_idh
                .groupby(['STATE','NAME_STATE','REGION'])
                .agg({'IDHM': 'median'})
                .sort_values(by=['IDHM'], ascending=False)
                .reset_index())

# Estatísticas Descritivas
media_idh = df_idh_estado['IDHM'].mean()
mediana_idh = df_idh_estado['IDHM'].median()
desvio_padrao_idh = df_idh_estado['IDHM'].std()
assimetria_idh = skew(df_idh_estado['IDHM'])

# Percentis e Limites de Controle
percentis = np.percentile(df_idh_estado['IDHM'], [25, 50, 75])
iqr = percentis[2] - percentis[0]
fator_iqr = 1.5
lci = percentis[0] - fator_iqr * iqr
lcs = percentis[2] + fator_iqr * iqr

# Limpando do dataset Longitude, Latitude e Altitude zeradas
df_idh_plt = df_idh[(df_idh['LONG'] != 0) & (df_idh['LAT'] != 0) & (df_idh['ALT'] != 0)]

# Região com o maior IDH (Top 5 acima do Q3)
df_maior_idh = df_idh_estado[df_idh_estado['IDHM'] >= percentis[2]].head(5)
nome_maior_idh_regiao = ', '.join(df_maior_idh['NAME_STATE'].values.tolist())

# Região com o menor IDH (Top 5 abaixo do Q1)
df_menor_idh = df_idh_estado[df_idh_estado['IDHM'] <= percentis[0]].tail(5)
nome_menor_idh_regiao = ', '.join(df_menor_idh['NAME_STATE'].values.tolist())

# Construindo a Resposta HTML
resp_q8 = f'Os Estados com os maiores índices de IDH são: <strong>"{nome_maior_idh_regiao}"</strong>.<br>Por outro lado, as regiões com os menores índices de IDH são: <strong>"{nome_menor_idh_regiao}"</strong>.<br><br><em>A estatística utilizada no processo de resolução foi a <strong>mediana</strong>, devido à sua capacidade de representar a concentração central dos dados na população, sendo menos suscetível a mudanças devido a valores extremos (outliers).</em>'

# --- CRIANDO O DASHBOARD COM OS 3 GRÁFICOS ---
fig = plt.figure(figsize=(12, 9))

# Plot 1: Grafico com maiores IDH 
ax1 = plt.subplot2grid((3, 2), (0, 0), rowspan=3)
sns.barplot(data=df_idh_estado, x='IDHM', y='STATE', orient='h', palette="cividis", alpha=0.7, ax=ax1)

# Rótulos nas barras
for p in ax1.patches:
    ax1.annotate(f'{p.get_width():.3f}', (p.get_width() + p.get_x(), p.get_y() + p.get_height() / 2.), 
                 ha='left', va='center_baseline', fontsize=9, color='black', fontweight='bold')

# Linhas Estatísticas (Q3, Q1, LCS, LCI)
ax1.axvline(percentis[2], color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.axvline(lcs, color='blue', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.axvspan(percentis[2], lcs, color='blue', alpha=0.1)

ax1.axvline(percentis[0], color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.axvline(lci, color='orange', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.axvspan(lci, percentis[0], color='yellow', alpha=0.1)

ax1.text((percentis[2] + 0.04), 25.7, 'MAIOR IDH', color='black', fontsize=20, fontweight='bold', fontstyle='italic', rotation=90, alpha=0.2) 
ax1.text((lci + 0.04), 25.7, 'MENOR IDH', color='black', fontsize=20, fontweight='bold', fontstyle='italic', rotation=90, alpha=0.2)

ax1.set_xlim(0.2, 0.9)
ax1.set_title('IDH por Estado: Do Maior para o Menor', fontweight='bold')
ax1.set_xlabel('Escala IDH', fontstyle='italic')
ax1.set_ylabel('Estados', fontstyle='italic')

# Plot 2: Grafico de Dispersao 
ax2 = plt.subplot2grid((3, 2), (1, 1), rowspan=2)
sns.scatterplot(data=df_idh_plt, x='LONG', y='LAT', hue='REGION', size='IDHM', sizes=(0,60), palette='pastel', ax=ax2)
ax2.legend(title='Região', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.set_title('Distribuição Geográfica do IDH', fontweight='bold')
ax2.set_xlabel('Longitude', fontstyle='italic')
ax2.set_ylabel('Latitude', fontstyle='italic')

# Plot 3: Histograma
ax3 = plt.subplot2grid((3, 2), (0, 1))
sns.histplot(df_idh_estado['IDHM'], kde=True, color='#5EAAA8', bins=10, ax=ax3)
ax3.axvline(media_idh, color='red', linestyle='--', linewidth=1, label=f'Média: {media_idh:.3f}')
ax3.axvline(mediana_idh, color='blue', linestyle='--', linewidth=1, label=f'Mediana: {mediana_idh:.3f}')
ax3.text(0.83, 1, f'Assimetria: {assimetria_idh:.2f}', rotation=90, fontsize=8)
ax3.set_title('Histograma do IDHM', fontweight='bold')
ax3.set_xlabel('Escala IDH')
ax3.set_ylabel('Frequência')
ax3.legend(fontsize=8)

plt.tight_layout()
plt.savefig('Imagens_graficos/q8_dashboard_idh.png', bbox_inches='tight', dpi=300) 
plt.close()

#%%
# Q9: Existe relação entre o PIB per capita e o IDHM das pessoas? Justificar
# =======================================================
# Filtrando o dataset para analisar a relação entre o PIB per capita e o IDHM
df_pib_idh = df[['STATE','REGION', 'IDHM', 'GDP_CAPITA']].copy()
df_pib_idh = df_pib_idh[(df_pib_idh['IDHM'] != 0) & (df_pib_idh['GDP_CAPITA'] != 0)]

# Preparacao dos dados para o modelo
X = np.array(df_pib_idh['IDHM']).reshape(-1,1)
y = df_pib_idh['GDP_CAPITA']

# Dividindo dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando e treinando o modelo de regressao linear (Scikit-Learn)
modelo_regressao = LinearRegression()
modelo_regressao.fit(X_treino, y_treino)

# Avalia o modelo nos dados de teste
score = modelo_regressao.score(X_teste, y_teste)
score_perc = score * 100

# Plotando e SALVANDO o gráfico de regressão (Q9)
plt.figure(figsize=(10,6))
plt.scatter(X, y, label='Dados Reais', color='#318FB5', alpha=0.6)
plt.plot(X, modelo_regressao.predict(X), color='orange', linewidth=2, label='Reta de Regressão (Modelo)')
plt.xlabel('IDH')
plt.ylabel('PIB per capita (R$)')
plt.title('Relação entre o PIB per capita e IDH')
plt.legend()
plt.savefig('Imagens_graficos/q9_regressao_pib_idh.png', bbox_inches='tight') 
plt.close()

# Formatando a SUA resposta HTML para a Q9
resp_q9 = f'Sim, existe uma relação entre o PIB per capita e o IDH das pessoas, onde regiões com maiores valores de PIB per capita tendem a ter um IDH mais alto. No entanto, o dinheiro não é um fator determinante absoluto, pois apenas <strong>{score_perc:.1f}%</strong> da variação no IDHM pode ser explicada matematicamente pelo PIB per capita.'

#%%
# Q10: Existe relação entre o PIB per capita e expectativa de vida (IDHM Longevidade)? Justificar
# =======================================================
# Filtrando o dataset para analisar a relação entre o PIB per capita e o IDHM Longevidade
df_pib_idh_long = df[['STATE','REGION', 'IDHM_Longevidade', 'GDP_CAPITA']].copy()
df_pib_idh_long = df_pib_idh_long[(df_pib_idh_long['IDHM_Longevidade'] != 0) & (df_pib_idh_long['GDP_CAPITA'] != 0)]

# Assimetria
skew_idh_long = round(skew(df_pib_idh_long['IDHM_Longevidade']), 3)

# Plotando e SALVANDO o Histograma da Longevidade (Q10)
plt.figure(figsize=(8,5))
sns.histplot(df_pib_idh_long['IDHM_Longevidade'], kde=True, color='#4D869C', bins=10, label='IDH Long')
plt.text(0.91, 0, f'Assimetria: {skew_idh_long:.2f}', rotation=90)
plt.xlabel('IDHM Longevidade')
plt.ylabel('Frequência')
plt.title('Histograma do IDH da Longevidade')
plt.legend()
plt.savefig('Imagens_graficos/q10_histograma_longevidade.png', bbox_inches='tight') 
plt.close()

# Preparacao dos dados para o modelo (Statsmodels)
X1 = np.array(df_pib_idh_long['IDHM_Longevidade']).reshape(-1,1)
y1 = df_pib_idh_long['GDP_CAPITA']
X1_sm = sm.add_constant(X1) # Adicao constante

# Treinamento do modelo OLS
modelo_ols = sm.OLS(y1, X1_sm)
resultado_ols = modelo_ols.fit()
r2_ols = resultado_ols.rsquared * 100 # Pegando o R² para o texto

# Plotando e SALVANDO o gráfico de Dispersão + OLS (Q10)
plt.figure(figsize=(10,6))
plt.scatter(df_pib_idh_long['IDHM_Longevidade'], y1, label='Dados Reais', color='#028391', alpha=0.6)
plt.plot(df_pib_idh_long['IDHM_Longevidade'], resultado_ols.fittedvalues, color='#FDDE55', linewidth=2, label='Linha de Previsão OLS')
plt.xlabel('IDHM Longevidade')
plt.ylabel('PIB per capita (R$)')
plt.title('Relação entre o PIB per capita e IDH Longevidade')
plt.legend()
plt.savefig('Imagens_graficos/q10_regressao_longevidade.png', bbox_inches='tight') 
plt.close()

# Formatando a resposta HTML para a Q10
resp_q10 = f'Sim, verificamos uma correlação positiva. Através do modelo de regressão OLS (Statsmodels), comprovamos que um maior PIB per capita reflete em melhor expectativa de vida. Contudo, assim como no IDH geral, a renda explica apenas <strong>{r2_ols:.1f}%</strong> da variância da longevidade, indicando que fatores como saneamento e infraestrutura do SUS (não apenas renda direta) são vitais.'


#%%
# Q11: Existe relação entre o PIB e o número total de empresas? Justificar
# =======================================================
# Filtrando o dataset para analisar a relação entre o PIB e o número total de empresas
df_pib_empresa = df[['STATE','REGION','COMP_TOT','GDP_CAPITA']].copy()

# Preparação dos dados para o modelo (Agrupando por Estado)
cols_empresas = ['COMP_TOT', 'GDP_CAPITA']
df_pib_empresa_estado = df_pib_empresa.groupby(['STATE','REGION'])[cols_empresas].sum().sort_values(by='GDP_CAPITA', ascending=False).reset_index()

# Assimetria do total de empresas
skew_pib_empresa = round(skew(df_pib_empresa_estado['COMP_TOT']), 2)

# Preparação das variáveis (Independente e Alvo)
X2 = np.array(df_pib_empresa_estado['COMP_TOT']).reshape(-1,1)
y2 = df_pib_empresa_estado['GDP_CAPITA']

# Adição constante da variável independente para o Statsmodels
X2_sm = sm.add_constant(X2) 

# Criando e treinando o modelo OLS - Statsmodels
modelo_ols_empresas = sm.OLS(y2, X2_sm)
resultado_empresas = modelo_ols_empresas.fit()

# Capturando o R² para a resposta
r2_empresas = resultado_empresas.rsquared * 100

# Plotando e SALVANDO o Gráfico de Dispersão com Reta OLS
plt.figure(figsize=(10,6))
plt.scatter(x=df_pib_empresa_estado['COMP_TOT'], y=y2, label='Dados Reais', color='#318FB5', alpha=0.7)
plt.plot(df_pib_empresa_estado['COMP_TOT'], resultado_empresas.fittedvalues, color='#EA5455', linewidth=2, label='Linha de Previsões OLS')
plt.xlabel('Total de Empresas')
plt.ylabel('PIB (Soma do Per Capita por Estado)')
plt.title(f'Relação entre o PIB e Total de Empresas (Assimetria: {skew_pib_empresa})')
plt.legend()
plt.savefig('Imagens_graficos/q11_regressao_empresas.png', bbox_inches='tight') 
plt.close()

# Formatando a SUA resposta HTML exata para injeção no site
resp_q11 = f'Sim, a relação é fortíssima e positiva. Através da Regressão Linear (OLS), identificamos que o número de empresas e a geração de riqueza crescem juntos de forma proporcional. O modelo nos mostra que a densidade de empresas consegue explicar <strong>{r2_empresas:.1f}%</strong> da variância do PIB nesses agrupamentos, comprovando que a força empreendedora e de infraestrutura logística é o verdadeiro motor econômico.'

#%%
# Q12: Existe relação entre o PIB per capita e o total de carros e/ou motos?
# =======================================================
# Filtrando o dataset para analisar a relação entre o PIB per capita e o total de carros e motos
df_pib_car_moto = df[['CITY','STATE','REGION', 'GDP_CAPITA', 'Cars', 'Motorcycles']].copy()

# Criando uma coluna auxiliar com a soma de Carros e Motos
df_pib_car_moto['Total_Car_Moto'] = df_pib_car_moto['Cars'] + df_pib_car_moto['Motorcycles']

# Agrupando por Estado
df_pib_car_moto_estado = df_pib_car_moto.groupby(['STATE','REGION']).agg({
    'GDP_CAPITA':'median', 
    'Cars':'sum', 
    'Motorcycles':'sum', 
    'Total_Car_Moto':'sum'
}).sort_values(by='GDP_CAPITA', ascending=False).reset_index()

# Assimetria
skew_car = round(skew(df_pib_car_moto_estado['Cars']), 2)
skew_moto = round(skew(df_pib_car_moto_estado['Motorcycles']), 2)
skew_car_moto = round(skew(df_pib_car_moto_estado['Total_Car_Moto']), 2)

# Calculando a correlação para a resposta textual
corr_veic_pib = df_pib_car_moto['GDP_CAPITA'].corr(df_pib_car_moto['Total_Car_Moto'])

# Plotando e SALVANDO o Histograma dos Dados de carros e Motos
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Histograma do Total de Carros
sns.histplot(df_pib_car_moto_estado['Cars'], kde=True, color='#4D869C', bins=15, ax=axes[0])
# Usando transform=axes.transAxes para o texto nunca sair da tela
axes[0].text(0.55, 0.8, f'Assimetria: {skew_car:.2f}', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
axes[0].set_xlabel('Carros')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Histograma - Carros')

# Histograma do Total de Motos
sns.histplot(df_pib_car_moto_estado['Motorcycles'], kde=True, color='#4D868A', bins=20, ax=axes[1])
axes[1].text(0.55, 0.8, f'Assimetria: {skew_moto:.2f}', transform=axes[1].transAxes, fontsize=10, fontweight='bold')
axes[1].set_xlabel('Motos')
axes[1].set_ylabel('Frequência')
axes[1].set_title('Histograma - Motos')

# Histograma do Total de Carros e Motos
sns.histplot(df_pib_car_moto_estado['Total_Car_Moto'], kde=True, color='#4D869C', bins=15, ax=axes[2])
axes[2].text(0.55, 0.8, f'Assimetria: {skew_car_moto:.2f}', transform=axes[2].transAxes, fontsize=10, fontweight='bold')
axes[2].set_xlabel('Carros e Motos')
axes[2].set_ylabel('Frequência')
axes[2].set_title('Histograma - Total')

plt.tight_layout()
plt.savefig('Imagens_graficos/q12_histogramas_veiculos.png', bbox_inches='tight') # <-- Salva no PC
plt.close()

# Formatando a SUA resposta HTML exata para injeção no site
resp_q12 = f'Sim, notamos uma correlação moderada a positiva (<strong>{corr_veic_pib:.2f}</strong>). O aumento da renda per capita reflete no poder de compra da frota veicular local. Observamos também através dos histogramas que a distribuição de veículos no Brasil é extremamente assimétrica (muito concentrada em poucos estados), com destaque para a assimetria total da frota, que marcou uma distorção de <strong>{skew_car_moto:.2f}</strong>.'


#%%
# Q13: Quantos veículos por pessoa em média cada Estado possui? E qual o maior e menor?
# =======================================================
# Filtrando o dataset para analisar a relação entre o número de veículos por pessoa em cada estado
df_veiculo = df[['STATE', 'IBGE_RES_POP', 'Cars', 'Motorcycles']].copy()

# Criando uma coluna auxiliar com a soma de Carros e Motos
df_veiculo['Total_Car_Moto'] = df_veiculo['Cars'] + df_veiculo['Motorcycles']

# Agrupando por Estado e somando a população e veículos
cols_agrupamento = ['IBGE_RES_POP', 'Total_Car_Moto']
df_veiculo_estado = df_veiculo.groupby('STATE')[cols_agrupamento].sum().reset_index()

# Coluna com o Total de Veículos por Pessoa
df_veiculo_estado['Veiculo_Pessoa'] = df_veiculo_estado['Total_Car_Moto'] / df_veiculo_estado['IBGE_RES_POP']

# Coluna com o Total de Pessoas por Veículo
df_veiculo_estado['Pessoa_Veiculo'] = df_veiculo_estado['IBGE_RES_POP'] / df_veiculo_estado['Total_Car_Moto']

# Organizar por Total de Veículos por Pessoa (decrescente)
df_veiculo_estado = df_veiculo_estado.sort_values(by='Veiculo_Pessoa', ascending=False).reset_index(drop=True)

# Estatísticas Básicas
media_veiculo = round(df_veiculo_estado['Veiculo_Pessoa'].mean(), 3)
mediana_veiculo = round(df_veiculo_estado['Veiculo_Pessoa'].median(), 3)
assimetria_veiculo = round(skew(df_veiculo_estado['Veiculo_Pessoa']), 3)

maior_veiculo = round(df_veiculo_estado['Veiculo_Pessoa'].max(), 3)
menor_veiculo = round(df_veiculo_estado['Veiculo_Pessoa'].min(), 3)

# Identificando os Estados
nome_maior_veiculo = df_veiculo_estado['STATE'].iloc[0]
nome_menor_veiculo = df_veiculo_estado['STATE'].iloc[-1] # Melhor usar -1 para pegar sempre o último da lista

# 1. Plotando e SALVANDO o Histograma
plt.figure(figsize=(7, 5))
ax = sns.histplot(df_veiculo_estado['Veiculo_Pessoa'], kde=True, color='gray', bins=10)
plt.axvline(media_veiculo, color='red', linestyle='--', linewidth=1, alpha=0.8, label=f'Média: {media_veiculo:.3f}')
plt.axvline(mediana_veiculo, color='blue', linestyle='--', linewidth=1, alpha=0.8, label=f'Mediana: {mediana_veiculo:.3f}')
# Texto fixo na parte direita do gráfico
plt.text(0.70, 0.4, f'Assimetria: {assimetria_veiculo:.3f}', transform=ax.transAxes, rotation=90)
plt.xlabel('Veículos por Habitante')
plt.ylabel('Frequência')
plt.title('Histograma da Densidade Veicular Estadual', fontweight='bold')
plt.legend()
plt.savefig('Imagens_graficos/q13_histograma_veiculos_pessoa.png', bbox_inches='tight') # <-- Salva no PC
plt.close()

# 2. Plotando e SALVANDO o gráfico de barras
plt.figure(figsize=(8, 7))
ax = sns.barplot(data=df_veiculo_estado, x='Veiculo_Pessoa', y='STATE', orient='h', palette='seismic', alpha=0.9)

# Adicionando rótulos de dados
for p in ax.patches:
    ax.annotate(f'{p.get_width():.3f}', (p.get_width() + 0.01, p.get_y() + p.get_height() / 2.), 
                ha='left', va='center_baseline', fontsize=10, color='black', fontweight='bold')

plt.xlim(0, 0.8)
plt.title('Veículos por Estado: Do Maior para o Menor', fontweight='bold')
plt.xlabel('Média de veículos por habitante', fontstyle='italic')
plt.ylabel('Estado', fontstyle='italic')
plt.savefig('Imagens_graficos/q13_barplot_veiculos_pessoa.png', bbox_inches='tight') # <-- Salva no PC
plt.close()

# Formatando a SUA resposta HTML exata para injeção no site
resp_q13 = f'A média nacional estadual de veículos por pessoa é <strong>{media_veiculo}</strong>.<br><br>O Estado com a maior média de veículos por pessoa é <strong>{nome_maior_veiculo}</strong>, com <strong>{maior_veiculo}</strong> veículos por habitante.<br>Enquanto isso, o Estado com a menor média é o <strong>{nome_menor_veiculo}</strong>, com apenas <strong>{menor_veiculo}</strong> veículos por pessoa.'

#%%
# Q14: Existe relação entre Áreas Plantadas e Número de Tratores? Justifique.
# =======================================================
# Filtrando o dataset para analisar a relação entre áreas plantadas e número de tratores
df_trator = df[['STATE', 'REGION', 'IBGE_PLANTED_AREA', 'Wheeled_tractor']].copy()

# Agrupando os dados por Estado
cols_trator = ['IBGE_PLANTED_AREA', 'Wheeled_tractor']
df_trator_estado = df_trator.groupby(['STATE', 'REGION'])[cols_trator].sum().sort_values(by='Wheeled_tractor', ascending=False).reset_index()

# Calculando a correlação de cada região dinamicamente para a resposta
corrs = {}
for regiao in df_trator_estado['REGION'].unique():
    subset = df_trator_estado[df_trator_estado['REGION'] == regiao]
    if len(subset) > 1: # Previne erro se a região tiver só 1 estado nos dados
        corrs[regiao] = subset['IBGE_PLANTED_AREA'].corr(subset['Wheeled_tractor'])

# Assimetria
skew_trator = round(skew(df_trator_estado['Wheeled_tractor']), 2)

# Preparacao dos dados para o modelo OLS 
X_trator = np.array(df_trator_estado['Wheeled_tractor']).reshape(-1, 1)
y_trator = df_trator_estado['IBGE_PLANTED_AREA']

# Adicao constante para o Statsmodels
X_trator_sm = sm.add_constant(X_trator)

# Treinando o modelo OLS
modelo_trator = sm.OLS(y_trator, X_trator_sm)
resultado_trator = modelo_trator.fit()
r2_trator = resultado_trator.rsquared * 100

# 1. Plotando e SALVANDO os gráficos (Histograma e Dispersão lado a lado)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico 1: Histograma
sns.histplot(df_trator_estado['Wheeled_tractor'], kde=True, color='#4D869C', bins=15, ax=axes[0])
axes[0].text(0.6, 0.8, f'Assimetria: {skew_trator:.2f}', transform=axes[0].transAxes, fontsize=10, fontweight='bold')
axes[0].set_xlabel('Total de Tratores')
axes[0].set_ylabel('Frequência')
axes[0].set_title('Histograma do Total de Tratores')

# Gráfico 2: Dispersão + Regressão
axes[1].scatter(x=df_trator_estado['Wheeled_tractor'], y=y_trator, color='#318FB5', label='Dados reais', alpha=0.7)
axes[1].plot(df_trator_estado['Wheeled_tractor'], resultado_trator.fittedvalues, color='orange', linewidth=2, label='Linha de Regressão OLS')
axes[1].set_xlabel('Total de Tratores')
axes[1].set_ylabel('Área Plantada')
axes[1].set_title('Relação: Área Plantada vs Total de Tratores')
axes[1].legend()

plt.tight_layout()
plt.savefig('Imagens_graficos/q14_regressao_tratores.png', bbox_inches='tight') # <-- Salva no PC
plt.close()

# Formatando a SUA resposta HTML exata para injeção no site
corr_texto = "".join([f"<br> &nbsp;&nbsp;• <strong>{r}</strong>: {c:.2f}" for r, c in corrs.items()])
resp_q14 = f'Sim, existe relação, mas com forte variação regional. O modelo de regressão global aponta que o número de tratores explica cerca de <strong>{r2_trator:.1f}%</strong> da variação da área plantada. No entanto, analisando as correlações isoladas por região, vemos os seguintes índices:{corr_texto}<br><br>Isso evidencia que o agronegócio no Sudeste e Sul é intensivo em maquinário (mecanização de precisão), enquanto o Nordeste apresenta forte dependência de métodos tradicionais, trabalho manual e agricultura familiar.'

#%% 
# PARTE 1 SITE: FEATURE ENGINEERING (Criando Inteligência)
# =======================================================
print("Gerando gráficos da Parte 1 (Traduzidos para Negócios)...")

# Matriz de Correlação
plt.figure(figsize=(10, 8))
cols_corr = ['GDP_CAPITA', 'IDHM', 'IDHM_Longevidade', 'IDHM_Educacao']
df_corr = df[cols_corr].rename(columns={
    'GDP_CAPITA': 'PIB per Capita (R$)',
    'IDHM': 'IDH (Qualidade de Vida)',
    'IDHM_Longevidade': 'IDH Longevidade',
    'IDHM_Educacao': 'IDH Educação'
})
sns.heatmap(df_corr.corr(), annot=True, cmap='RdBu', fmt=".2f", square=True)
plt.title('Matriz de Correlação: Riqueza x Desenvolvimento', fontsize=14, fontweight='bold')
plt.savefig('Imagens_graficos/matriz_correlacao.png', dpi=300, bbox_inches='tight')
plt.close()

# Empresas vs PIB 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='GDP', y='COMP_TOT', alpha=0.5, color='#203a43')
plt.xscale('log')
plt.yscale('log')
plt.title('O Motor do Emprego: Volume de Empresas vs PIB', fontsize=14, fontweight='bold')
plt.xlabel('PIB Total da Cidade (Escala Log)', fontsize=12)
plt.ylabel('Total de Empresas Ativas (Escala Log)', fontsize=12)
plt.savefig('Imagens_graficos/motor_emprego.png', dpi=300, bbox_inches='tight') # <-- PNG AQUI
plt.close()

# Frota vs PIB per Capita (CORRIGIDO A PROPORÇÃO)
plt.figure(figsize=(10, 6))
df['Carros_por_Habitante'] = df['Cars'] / df['IBGE_RES_POP']
sns.scatterplot(data=df, x='GDP_CAPITA', y='Carros_por_Habitante', alpha=0.5, color='#17a2b8')
plt.title('Poder de Compra: Frota de Carros vs Riqueza Média', fontsize=14, fontweight='bold')
plt.xlabel('PIB per Capita (R$)', fontsize=12)
plt.ylabel('Média de Carros por Habitante', fontsize=12)
plt.savefig('Imagens_graficos/poder_compra.png', dpi=300, bbox_inches='tight')
plt.close()

# Pairplot
cols_pair = ['IBGE_RES_POP', 'GDP_CAPITA', 'Cars', 'Motorcycles']
df_pair = df[cols_pair].rename(columns={
    'IBGE_RES_POP': 'População',
    'GDP_CAPITA': 'PIB per Capita',
    'Cars': 'Total de Carros',
    'Motorcycles': 'Total de Motos'
})
pp = sns.pairplot(df_pair, diag_kind='kde', plot_kws={'alpha': 0.4})
pp.fig.suptitle('Visão Geral: População, Renda e Frota', y=1.02, fontsize=16, fontweight='bold')
pp.savefig('visao_geral.png', dpi=150, bbox_inches='tight')
plt.close()

#%%
# O ABISMO SOCIAL (TOP 10 E BOTTOM 10 IDHM)
# =======================================================
print("Gerando gráfico do Abismo Social (IDHM)...")

# Filtrando cidades válidas
df_idh_valid = df_ml.dropna(subset=['IDHM', 'CITY', 'STATE']).copy()

# Pegando as 10 melhores e as 10 piores
top_10_idh = df_idh_valid.nlargest(10, 'IDHM')
bottom_10_idh = df_idh_valid.nsmallest(10, 'IDHM')

# Juntando tudo para o gráfico
df_idh_extremes = pd.concat([top_10_idh, bottom_10_idh])
df_idh_extremes['Cidade_UF'] = df_idh_extremes['CITY'] + ' (' + df_idh_extremes['STATE'] + ')'
df_idh_extremes = df_idh_extremes.sort_values('IDHM', ascending=True)

# Definindo cores (Verde para Top 10, Vermelho Escuro para Bottom 10)
cores_idh = ['#EA5455' if x < 0.6 else '#28C76F' for x in df_idh_extremes['IDHM']]

plt.figure(figsize=(12, 8))
sns.barplot(data=df_idh_extremes, x='IDHM', y='Cidade_UF', palette=cores_idh)

plt.title('O Abismo Social: As 10 Melhores e Piores Cidades em Qualidade de Vida', fontweight='bold', fontsize=15)
plt.xlabel('Índice de Desenvolvimento Humano (IDHM)', fontsize=12)
plt.ylabel('')

# Linhas de referência da ONU
plt.axvline(x=0.800, color='darkgreen', linestyle='--', alpha=0.7, label='IDH Muito Alto (Padrão Europeu)')
plt.axvline(x=0.500, color='darkred', linestyle='--', alpha=0.7, label='IDH Muito Baixo (Alerta Crítico)')

plt.legend(loc='lower right', fontsize=10)
plt.savefig("Imagens_graficos/ideia_idh.png", dpi=300, bbox_inches='tight')
plt.close()

#%%
# PARTE 2 SITE: CLUSTERIZAÇÃO E INSIGHTS DE NEGÓCIOS
# =======================================================
print("\n--- GERANDO INSIGHTS DE NEGÓCIOS ---")

# Criando um dataset limpo para os modelos avançados
df_ml = df.copy()
df_ml = df_ml[(df_ml['IDHM'].notnull()) & (df_ml['GDP_CAPITA'].notnull())]

# 1. Densidade Empreendedora: Quantas empresas para cada 1.000 habitantes?
df_ml['Densidade_Empreendedora'] = (df_ml['COMP_TOT'] / df_ml['IBGE_RES_POP']) * 1000

# 2. Acesso a Crédito/Bancarização: Agências por 10.000 habitantes
df_ml['Bancarizacao'] = ((df_ml['Pr_Agencies'] + df_ml['Pu_Agencies']) / df_ml['IBGE_RES_POP']) * 10000

# 3. Força do Agro per Capita: Quanto o Agro gera por habitante
df_ml['Agro_per_Capita'] = df_ml['GVA_AGROPEC'] / df_ml['IBGE_RES_POP']

print("✓ Métricas avançadas criadas com sucesso (Empreendedorismo, Bancos e Agro).")

#%% 
# CLUSTERIZAÇÃO (K-MEANS) - OS "4 BRASIS"
# =======================================================
features_cluster = ['GDP_CAPITA', 'Densidade_Empreendedora', 'Bancarizacao', 'Agro_per_Capita']
df_ml = df_ml.replace([np.inf, -np.inf], np.nan).dropna(subset=features_cluster)

# Padronizando os dados (StandardScaler é obrigatório antes do K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_ml[features_cluster])

# Aplicando K-Means para encontrar 4 "Perfis de Cidades"
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_ml['Cluster_Economico'] = kmeans.fit_predict(X_scaled)

# Traduzindo os clusters para a linguagem de negócios
dic_clusters = {
    0: 'Cidades de Base / Pouca Atividade',
    1: 'Polos de Comércio e Serviços',
    2: 'Potências do Agronegócio',
    3: 'Metrópoles e Elite Financeira'
}
df_ml['Perfil_Cidade'] = df_ml['Cluster_Economico'].map(dic_clusters)
print("✓ Cidades clusterizadas com IA nos '4 Brasis'.")

#%% 
# O PARADOXO DA RIQUEZA: PIB ALTO, MAS EDUCAÇÃO BAIXA
# =======================================================
# Buscando cidades ricas (Top 10%) mas com Educação ruim (Bottom 25%)
pib_alto = df_ml['GDP_CAPITA'].quantile(0.90)   # Top 10% mais ricas
pib_baixo = df_ml['GDP_CAPITA'].quantile(0.25)  # 25% mais pobres
edu_alta = df_ml['IDHM_Educacao'].quantile(0.90) # Top 10% em educação
edu_baixa = df_ml['IDHM_Educacao'].quantile(0.25) # 25% piores em educação

# Criando os três grupos estratégicos
# O PARADOXO: Rica, mas sem educação
paradoxo = df_ml[(df_ml['GDP_CAPITA'] >= pib_alto) & (df_ml['IDHM_Educacao'] <= edu_baixa)].copy()

# A ELITE: Rica e com excelente educação
elite = df_ml[(df_ml['GDP_CAPITA'] >= pib_alto) & (df_ml['IDHM_Educacao'] >= edu_alta)].copy()

# A EFICIÊNCIA: Baixa renda, mas alta educação
eficiencia = df_ml[(df_ml['GDP_CAPITA'] <= pib_baixo) & (df_ml['IDHM_Educacao'] >= edu_alta)].copy()

# Função para gerar as tabelas HTML que o seu index.html precisa
def gera_tabela_top7(dataframe):
    # Pegamos as 7 cidades de maior PIB de cada grupo para o ranking
    df_top = dataframe.nlargest(7, 'GDP_CAPITA')[['CITY', 'STATE', 'GDP_CAPITA', 'IDHM_Educacao']]
    # Formatação para o padrão brasileiro
    df_top['GDP_CAPITA'] = df_top['GDP_CAPITA'].apply(lambda x: f"R$ {x:,.0f}".replace(',', '.'))
    # Renomeando as colunas para o formato que o site espera
    df_top.columns = ['Cidade', 'UF', 'PIB per Capita', 'IDH Educação']
    # Convertendo para HTML com classes CSS para estilização
    return df_top.to_html(classes='table-custom text-center', index=False, border=0)

# Gerando os objetos de tabela para injeção
tabela_paradoxo = gera_tabela_top7(paradoxo)
tabela_elite = gera_tabela_top7(elite)
tabela_eficiencia = gera_tabela_top7(eficiencia)

# Gráfico unificado para o site
plt.figure(figsize=(12, 6))

# Fundo com todas as cidades
sns.scatterplot(data=df_ml, x='GDP_CAPITA', y='IDHM_Educacao', color='lightgrey', alpha=0.3, label='Demais Cidades')

# Destaque dos grupos
sns.scatterplot(data=paradoxo, x='GDP_CAPITA', y='IDHM_Educacao', color='red', s=100, edgecolor='black', label='Paradoxo')
sns.scatterplot(data=elite, x='GDP_CAPITA', y='IDHM_Educacao', color='green', s=100, edgecolor='black', label='Elite')
sns.scatterplot(data=eficiencia, x='GDP_CAPITA', y='IDHM_Educacao', color='blue', s=100, edgecolor='black', label='Eficiência Social')

# LINHAS DE FRONTEIRA (O segredo da visualização)
plt.axvline(pib_alto, color='darkred', linestyle='--', alpha=0.6, label=f'Corte Riqueza (Top 10%)')
plt.axvline(pib_baixo, color='darkblue', linestyle='--', alpha=0.6, label=f'Corte Baixa Renda (25%)')
plt.axhline(edu_alta, color='darkgreen', linestyle=':', alpha=0.6, label=f'Corte Alta Educação (Top 10%)')
plt.axhline(edu_baixa, color='orange', linestyle=':', alpha=0.6, label=f'Corte Baixa Educação (25%)')

plt.title('Os Extremos do Brasil: Riqueza vs Educação (Com Linhas de Corte)', fontweight='bold', fontsize=14)
plt.xlabel('PIB per Capita (R$)', fontsize=12)
plt.ylabel('Índice de Educação (IDHM)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

plt.savefig("Imagens_graficos/grafico_paradoxo.png", dpi=300, bbox_inches='tight')
plt.close()

#%% 
# MACHINE LEARNING: RANDOM FOREST (O VERDADEIRO MOTOR DO IDH)
# =======================================================
features_rf = ['GDP_CAPITA', 'TAXES', 'COMP_TOT', 'Cars', 'Motorcycles', 'MUN_EXPENDIT', 'GVA_INDUSTRY', 'GVA_SERVICES']
df_rf = df_ml.dropna(subset=features_rf + ['IDHM'])

X = df_rf[features_rf]
y = df_rf['IDHM']

# Treinando a Floresta Aleatória
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# TRADUÇÃO DE NEGÓCIOS DAS VARIÁVEIS (Para o gráfico ficar legível para qualquer um)
dicionario_rf = {
    'GDP_CAPITA': 'Riqueza (PIB per Capita)',
    'TAXES': 'Arrecadação de Impostos',
    'COMP_TOT': 'Densidade de Empresas',
    'Cars': 'Frota de Carros',
    'Motorcycles': 'Frota de Motos',
    'MUN_EXPENDIT': 'Investimento Público da Prefeitura',
    'GVA_INDUSTRY': 'Força do Setor Industrial',
    'GVA_SERVICES': 'Força do Setor de Serviços'
}

importancias = pd.DataFrame({
    'Variavel_Original': features_rf,
    'Importancia': rf_model.feature_importances_
})
importancias['Variavel'] = importancias['Variavel_Original'].map(dicionario_rf)
importancias = importancias.sort_values(by='Importancia', ascending=False)

# Plotando e Salvando
plt.figure(figsize=(10, 6))
sns.barplot(data=importancias, x='Importancia', y='Variavel', palette='viridis')
plt.title('O Que Mais Impacta a Qualidade de Vida (IDH) nas Cidades?', fontweight='bold')
plt.xlabel('Nível de Importância (Calculado pela Inteligência Artificial)')
plt.ylabel('Setores da Economia')
plt.savefig("Imagens_graficos/grafico_random_forest.png", dpi=300, bbox_inches='tight')
plt.close()
print("✓ Gráfico de Inteligência Artificial (Random Forest) gerado.")


#%%
# ATRATIVIDADE COMERCIAL (MINAS DE OURO DO VAREJO)
# =======================================================
# Definindo as "notas de corte" (mediana das cidades que já têm a franquia)
pib_corte = df[df['MAC'] > 0]['GDP_CAPITA'].median()
pop_corte = df[df['MAC'] > 0]['IBGE_RES_POP'].median()

# Função para classificar o nível de atratividade da cidade
def classifica_oportunidade(row):
    if row['MAC'] > 0:
        return "Já possui McDonald's"
    elif (row['GDP_CAPITA'] >= pib_corte) and (row['IBGE_RES_POP'] >= pop_corte):
        return "Mina de Ouro (Potencial)"
    else:
        return "Fora do Radar"

df['STATUS_VAREJO'] = df.apply(classifica_oportunidade, axis=1)

# Gráfico de Dispersão Focado na Oportunidade
fig_retail = px.scatter(
    df, 
    x="IBGE_RES_POP", 
    y="GDP_CAPITA", 
    color="STATUS_VAREJO",
    hover_name="CITY",
    hover_data={"STATE": True, "STATUS_VAREJO": False},
    log_x=True, # Escala logarítmica para visualização de extremos de população
    title="Mapa de Expansão: Onde abrir a próxima franquia?",
    labels={"IBGE_RES_POP": "Tamanho da População", "GDP_CAPITA": "Poder de Compra (PIB per Capita)"},
    color_discrete_map={
        "Já possui McDonald's": "#EA5455", # Vermelho
        "Mina de Ouro (Potencial)": "#28C76F", # Verde forte chamativo
        "Fora do Radar": "#E0E0E0" # Cinza claro para não distrair
    }
)

# Melhorando as bordas dos pontos para ficarem nítidos
fig_retail.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')))

html_franquias = fig_retail.to_html(full_html=False, include_plotlyjs='cdn')

# --- NOVA PARTE: CRIANDO A TABELA RANKING (TOP 10) ---
potenciais = df[df['STATUS_VAREJO'] == "Mina de Ouro (Potencial)"].copy()

# Pegando as Top 10 cidades com maior população (maior mercado consumidor)
top10_potenciais = potenciais.nlargest(10, 'IBGE_RES_POP')[['CITY', 'STATE', 'IBGE_RES_POP', 'GDP_CAPITA']]

# Formatando os números para o padrão brasileiro
top10_potenciais['IBGE_RES_POP'] = top10_potenciais['IBGE_RES_POP'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
top10_potenciais['GDP_CAPITA'] = top10_potenciais['GDP_CAPITA'].apply(lambda x: f"R$ {x:,.0f}".replace(',', '.'))

# Renomeando as colunas
top10_potenciais.columns = ['Cidade', 'UF', 'População', 'PIB per Capita']

# Gerando o HTML da Tabela com classes do Bootstrap para ficar impecável no site
tabela_html = top10_potenciais.to_html(classes='table-custom text-center mt-3', index=False, border=0, justify='center')

#%%
# INCLUSÃO FINANCEIRA E DESBANCARIZAÇÃO
# =======================================================
df['STATUS_BANCARIO'] = df['Pr_Bank'].apply(lambda x: 'Sem Bancos Privados' if x == 0 else 'Com Bancos Privados')
df['EMPRESAS_POR_1K_HAB'] = (df['COMP_TOT'] / df['IBGE_RES_POP']) * 1000

plt.figure(figsize=(9, 5))
sns.barplot(data=df, x='STATUS_BANCARIO', y='EMPRESAS_POR_1K_HAB', estimator=np.median, palette='Set2')
plt.title('A Falta de Bancos Atrasa o Empreendedorismo?', fontweight='bold')
plt.xlabel('')
plt.ylabel('Média de Empresas (a cada 1.000 Habitantes)')
plt.savefig("Imagens_graficos/grafico_desbancarizacao.png", dpi=300, bbox_inches='tight')
plt.close()

#%% 
# O PODER DO TURISMO (HOTÉIS E RESTAURANTES)
# =======================================================
df_turismo = df.dropna(subset=['CATEGORIA_TUR']).copy()
df_turismo['COMP_I_POR_10K'] = (df_turismo['COMP_I'] / df_turismo['IBGE_RES_POP']) * 10000

plt.figure(figsize=(10, 5))
sns.barplot(data=df_turismo.sort_values('CATEGORIA_TUR'), x='CATEGORIA_TUR', y='COMP_I_POR_10K', palette='magma', errorbar=None)
plt.title('A Força do Turismo na Geração de Negócios', fontweight='bold')
plt.xlabel('Categoria Turística IBGE (A = Internacional, E = Local)')
plt.ylabel('Volume de Hotéis e Restaurantes (a cada 10k Hab.)')
plt.savefig("Imagens_graficos/grafico_turismo.png", dpi=300, bbox_inches='tight')
plt.close()

#%% 
# BOOM POPULACIONAL E A CONSTRUÇÃO CIVIL
# =======================================================
df['CRESCIMENTO_POP_%'] = ((df['ESTIMATED_POP'] - df['IBGE_RES_POP']) / df['IBGE_RES_POP']) * 100
df_cresc = df[(df['CRESCIMENTO_POP_%'] > -20) & (df['CRESCIMENTO_POP_%'] < 100)].copy()
df_cresc['RITMO_CRESCIMENTO'] = pd.qcut(df_cresc['CRESCIMENTO_POP_%'], q=3, labels=['Cidades Encolhendo', 'Crescimento Estável', 'Explosão Populacional'])
df_cresc['CONSTRUCAO_POR_10K'] = (df_cresc['COMP_F'] / df_cresc['IBGE_RES_POP']) * 10000

plt.figure(figsize=(9, 5))
sns.barplot(data=df_cresc, x='RITMO_CRESCIMENTO', y='CONSTRUCAO_POR_10K', palette='Blues')
plt.title('O "Boom" da Construção Civil nas Cidades em Expansão', fontweight='bold')
plt.xlabel('Ritmo de Crescimento Populacional')
plt.ylabel('Obras e Construtoras Ativas (a cada 10k Hab.)')
plt.savefig("Imagens_graficos/grafico_construcao.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Todos os gráficos de negócios foram salvos e prontos para o Site!")

#%%
# SCRIPT FINAL: GERADOR AUTOMÁTICO DO PORTFÓLIO HTML
# =======================================================
print("Gerando o portfólio web...")

# Abre o nosso molde HTML atualizado
with open('index.html', 'r', encoding='utf-8') as f:
    html_template = f.read()

# Injeta os Textos das Respostas de 1 a 14
for i in range(1, 15):
    try:
        valor = locals()[f'resp_q{i}']
        html_template = html_template.replace(f'{{{{RESPOSTA_{i}}}}}', str(valor))
    except KeyError:
        print(f"Aviso: resp_q{i} não calculada.")

# Injeta as Tabelas dos Grupos Estratégicos 
html_template = html_template.replace('{{TABELA_PARADOXO}}', tabela_paradoxo)
html_template = html_template.replace('{{TABELA_ELITE}}', tabela_elite)
html_template = html_template.replace('{{TABELA_EFICIENCIA}}', tabela_eficiencia)

# Gráfico de Franquias
html_franquias = fig_retail.to_html(full_html=False, include_plotlyjs='cdn')
html_template = html_template.replace('{{GRAFICO_FRANQUIAS}}', html_franquias)
html_template = html_template.replace('{{TABELA_FRANQUIAS}}', tabela_html)

# Salva a obra de arte final
with open('index_atualizado.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

print("-> TUDO PRONTO! Abra o arquivo 'index_atualizado.html' no navegador!")

#%%