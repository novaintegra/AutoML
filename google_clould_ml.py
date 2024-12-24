from google.cloud import automl_v1 as automl

# Crear un cliente
client = automl.AutoMlClient()

# Especificar el nombre del proyecto y la ubicación
project_id = "your-project-id"
location = "us-central1"

# Crear un dataset
metadata = automl.TextClassificationDatasetMetadata()
dataset = automl.Dataset(
    display_name="my_text_classification_dataset",
    text_classification_dataset_metadata=metadata
)

# Crear la operación de creación del dataset
parent = f"projects/{project_id}/locations/{location}"
operation = client.create_dataset(parent=parent, dataset=dataset)

# Esperar a que la operación se complete
response = operation.result()
dataset_id = response.name.split("/")[-1]
print("Dataset creado: {}".format(dataset_id))