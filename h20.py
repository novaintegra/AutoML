import h2o
from h2o.automl import H2OAutoML

# Inicializar H2O
h2o.init()

# Cargar los datos
train = h2o.import_file("train.csv")
test = h2o.import_file("test.csv")

# Especificar la columna objetivo
y = "Delivery_Time_min"

# Crear y entrenar el objeto AutoML
aml = H2OAutoML(max_models=20, seed=1)
aml.train(x=train.columns, y=y, training_frame=train)

# Leaderboard de los modelos
print("modelos")
lb = aml.leaderboard
lb.head(rows=5)

print(lb.head(rows=5))

print("prediccion")
# Hacer predicciones con el mejor modelo
predictions = aml.leader.predict(test)

print(predictions)