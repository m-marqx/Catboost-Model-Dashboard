## EN-US
If you want to read this in English, click here. [click here](https://github.com/m-marqx/Beautiful-Model/blob/master/README.md)

# 📊 Catboost Model Dashboard

## 🎯 Objetivo Principal do Repositório

Simplificar a visualização de modelos de Machine Learning (ML) utilizando um dashboard interativo.

## 🚀 Funcionalidades

- **📁 Upload de Modelos**: Permite ao usuário carregar um arquivo JSON com os parâmetros do modelo.
- **📈 Visualização de Dados**: Utiliza dados de `data/assets/asset_data.parquet` para criar modelos de ML com o algoritmo CatBoost.
- **📊 Geração de Gráficos e Tabelas**: Produz gráficos de resultados, métricas (Drawdown, Resultados Sequenciais, Win Rate e Retorno Esperado) e tabelas de recomendação e de resultado sequencial.

## 🛠️ Instruções de Uso

### Passo 1: 📄 Criação do Arquivo JSON

Crie um arquivo JSON contendo todos os parâmetros necessários para a função `base_model_creation` localizada em [machine_learning/model_builder.py](machine_learning/model_builder.py).

### Passo 2: ⬆️ Upload do Modelo

No dashboard, clique em "Upload Model" e selecione o arquivo JSON criado no passo anterior. O dashboard utilizará os dados de `data/assets/asset_data.parquet` para criar o modelo de ML utilizando o algoritmo CatBoost.

### Passo 3: 📊 Visualização dos Resultados

Após o upload, o dashboard gerará automaticamente:
- **Gráfico de Resultados**
- **Gráfico das Métricas**: Inclui Drawdown, Resultados Sequenciais, Win Rate e Retorno Esperado.
- **Tabelas de Recomendação e de Resultado Sequencial**

## 📂 Estrutura do Repositório

- `data/`: Contém os dados utilizados para treinar os modelos.
- `machine_learning/`: Contém scripts para criação e mineração de modelos.
  - `model_builder.py`: Script principal para criação de modelos.
  - `model_miner.py`: Facilita a busca pela configuração ideal de um modelo de ML.
  - `model_features.py`: Simplifica a criação de features para o algoritmo CatBoost.
- `run_model.py`: Script principal para execução do modelo e geração de visualizações.

## 📋 Não Implementado (To-Do para Próximos Projetos)

### Model Miner

Embora não implementado no dashboard, a pasta `machine_learning` possui um arquivo chamado [model_miner.py](machine_learning/model_miner.py) que facilita a busca pela configuração ideal de um modelo de ML. O resultado final será um dict que se transformado em JSON, será compatível com a visualização do site. É possível realizar a mineração de modelos utilizando um arquivo de Jupyter Notebook e utilizar o dashboard para acompanhamento de resultados.

### Model Features

Outro arquivo relevante é o [model_features.py](machine_learning/model_features.py), que possui uma classe que simplifica a criação de features para serem utilizadas no algoritmo CatBoost.

## 📦 Instalação de Dependências

Para baixar todas as dependências do dashboard, utilize o comando:

```sh
pip install -r requirements.txt