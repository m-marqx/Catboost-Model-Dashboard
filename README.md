## PT-BR
Se você quiser ler em português basta [clicar aqui](https://github.com/m-marqx/Beautiful-Model/blob/master/Leia-me.md)

# 📊 Catboost Model Dashboard

## 🎯 Main Objective of the Repository

Simplify the visualization of Machine Learning (ML) models using an interactive dashboard.

## 🚀 Features

- **📁 Model Upload**: Allows the user to upload a JSON file with the model parameters.
- **📈 Data Visualization**: Uses data from `data/assets/asset_data.parquet` to create ML models with the CatBoost algorithm.
- **📊 Generation of Charts and Tables**: Produces result charts, metrics (Drawdown, Sequential Results, Win Rate, and Expected Return), and recommendation and sequential result tables.

## 🛠️ Usage Instructions

### Step 1: 📄 Create the JSON File

Create a JSON file containing all the necessary parameters for the [`base_model_creation`](machine_learning/model_builder.py) function located in [machine_learning/model_builder.py](machine_learning/model_builder.py).

### Step 2: ⬆️ Upload the Model

In the dashboard, click on "Upload Model" and select the JSON file created in the previous step. The dashboard will use the data from `data/assets/asset_data.parquet` to create the ML model using the CatBoost algorithm.

### Step 3: 📊 View the Results

After the upload, the dashboard will automatically generate:
- **Result Chart**
- **Metrics Chart**: Includes Drawdown, Sequential Results, Win Rate, and Expected Return.
- **Recommendation and Sequential Result Tables**

## 📂 Repository Structure

- `data/`: Contains the data used to train the models.
- `machine_learning/`: Contains scripts for model creation and mining.
  - [`model_builder.py`](machine_learning/model_builder.py): Main script for model creation.
  - [`model_miner.py`](machine_learning/model_miner.py): Facilitates the search for the ideal ML model configuration.
  - [`model_features.py`](machine_learning/model_features.py): Simplifies the creation of features for the CatBoost algorithm.
- `run_model.py`: Main script for model execution and visualization generation.

## 📋 Not Implemented (To-Do for Future Projects)

### Model Miner

Although not implemented in the dashboard, the `machine_learning` folder has a file called [`model_miner.py`](machine_learning/model_miner.py) that facilitates the search for the ideal ML model configuration. The final result will be a dict that, when transformed into JSON, will be compatible with the site's visualization. It is possible to perform model mining using a Jupyter Notebook file and use the dashboard to track results.

### Model Features

Another relevant file is [`model_features.py`](machine_learning/model_features.py), which has a class that simplifies the creation of features to be used in the CatBoost algorithm.

## 📦 Dependency Installation

To download all the dashboard dependencies, use the command:

```sh
pip install -r requirements.txt