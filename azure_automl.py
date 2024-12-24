#pip install azure-ai-ml

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.automl import classification
from azure.ai.ml.entities import Data

# Autenticación y configuración del cliente
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

# Configuración del área de trabajo
subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AZUREML_WORKSPACE_NAME>"
ml_client = MLClient(credential, subscription_id, resource_group, workspace)

# Cargar datos
data = Data(
    path="path/to/your/dataset.csv",
    type="mltable",
    description="Descripción de tus datos"
)

# Configuración del experimento de AutoML
automl_settings = classification.AutoMLClassificationSettings(
    task="classification",
    primary_metric="accuracy",
    training_data=data,
    label_column_name="label",
    n_cross_validations=5
)

# Ejecutar el experimento
automl_run = ml_client.automl.run(
    experiment_name="automl_classification_experiment",
    settings=automl_settings
)

# Obtener el mejor modelo
best_model = automl_run.get_output()
print(f"Mejor modelo: {best_model}")