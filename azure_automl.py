#pip install azureml
#pip install azureml.automl.core

import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.dataset import Dataset
from azureml.automl.core.experiment import AutoMLConfig

# Configuración del workspace
ws = Workspace.from_config()

# Carga de datos
dataset = Dataset.Tabular.from_delimited_files(path="https://raw.githubusercontent.com/Azure/MachineLearningSamples/master/datasets/adult-census/AdultCensusIncome.csv")

# Configuración de AutoML
automl_config = AutoMLConfig(
    task="classification",
    primary_metric='accuracy',
    compute_target="local",  # O una instancia de AzureML Compute
    training_data=dataset,
    label_column_name="income",
    n_cross_validations=5
)

# Ejecución de AutoML
experiment = Experiment(workspace=ws, name="my_automl_experiment")
local_run = experiment.submit(automl_config)

# Visualización de los resultados
local_run.wait_for_completion(show_output=True)
best_run, fitted_model = local_run.get_output()
print(best_run.properties)