# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "sentiment_analysis"
model_info = client.get_registered_model("sentiment_analysis")
version_name = model_info.latest_versions[-1].name
loaded_model = mlflow.pyfunc.load_model(model_uri=f"models:/{version_name}/Production")

# COMMAND ----------

loaded_model.predict(data={"sentence": "I hate you"})

# COMMAND ----------

loaded_model.predict(data={"sentence": "I love you"})
