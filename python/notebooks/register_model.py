# Databricks notebook source
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
config = AutoConfig.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# COMMAND ----------

model_path = "/dbfs/tmp/sentiment/model/"
tokenizer_path = "/dbfs/tmp/sentiment/tokenizer"
config_path = "/dbfs/tmp/sentiment/config"

model.save_pretrained(model_path)
tokenizer.save_pretrained(tokenizer_path)
config.save_pretrained(config_path)

# COMMAND ----------

import mlflow
from python.modules.huggingface_wrapper import HuggingFaceWrapper


model_name = "sentiment_analysis"
artifact_path = "model"
artifacts = {
  "hf_model_path": model_path,
  "hf_tokenizer_path": tokenizer_path,
  "hf_config_path": config_path,
}

model_info = None

with mlflow.start_run() as run:
  model_info = mlflow.pyfunc.log_model(
      artifact_path = artifact_path,
      python_model = HuggingFaceWrapper(),
      artifacts = artifacts,
      pip_requirements = ["pytorch", "transformers"]
  )

print(model_info)

# COMMAND ----------

version_info = mlflow.register_model(model_uri = model_info.model_uri, name = model_name)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
stage = "Production"

client.transition_model_version_stage(
    name=version_info.name,
    version=version_info.version,
    stage=stage
)

# COMMAND ----------


