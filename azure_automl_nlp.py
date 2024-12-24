#pip install azure-ai-ml


from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.automl import text

# Autenticación y configuración del cliente
credential = DefaultAzureCredential()
ml_client = MLClient.from_config(credential)

# Configuración del área de trabajo
subscription_id = "<SUBSCRIPTION_ID>"
resource_group = "<RESOURCE_GROUP>"
workspace = "<AZUREML_WORKSPACE_NAME>"
ml_client = MLClient(credential, subscription_id, resource_group, workspace)

#Cargar datos
from azure.ai.ml.entities import Data

data = Data(
    path="path/to/your/qa_dataset.csv",
    type="mltable",
    description="Conjunto de datos de preguntas y respuestas"
)

#Configurar Experimento
automl_settings = text.AutoMLTextClassificationSettings(
    task="text-classification",
    primary_metric="accuracy",
    training_data=data,
    label_column_name="answer",
    n_cross_validations=5
)

#Ejecutar el experimento
automl_run = ml_client.automl.run(
    experiment_name="automl_qa_experiment",
    settings=automl_settings
)

# Obtener el mejor modelo
best_model = automl_run.get_output()
print(f"Mejor modelo: {best_model}")