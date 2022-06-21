import torch
import mlflow.pyfunc

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

class HuggingFaceWrapper(mlflow.pyfunc.PythonModel):
    """
    Class to use HuggingFace Models
    """

    def load_context(self, context):
        """This method is called when loading an MLflow model with pyfunc.load_model(), as soon as the Python Model is constructed.
        Args:
            context: MLflow context where the model artifact is stored.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["hf_tokenizer_path"])
        self.model = AutoModelForSequenceClassification.from_pretrained(context.artifacts["hf_model_path"])
        self.config = AutoConfig.from_pretrained(context.artifacts["hf_config_path"])
        

    def predict(self, context, model_input):
        """This is an abstract function which returns the sentiment prediction.
        Args:
            model_input ([type]): the input data to fit into the model.
        Returns:
            [dict]: the detected sentiment for the sentence.
        """

        # create embeddings for inputs
        inputs = model_input["sentence"]
        embeddings = self.tokenizer(
            inputs,
            return_tensors = "pt",
            max_length = self.config.max_length,
            padding = "max_length",
            truncation = True,
        )
        # convert to tuple for neuron model
        hf_inputs = tuple(embeddings.values())

        # run prediciton
        with torch.no_grad():
            predictions = self.model(*hf_inputs)[0]
            scores = torch.nn.Softmax(dim=1)(predictions)

        # return dictonary, which will be json serializable
        return [{"label": self.config.id2label[item.argmax().item()], "score": item.max().item()} for item in scores]


def _load_pyfunc(data_path):
    """
    Load PyFunc implementation. Called by ``pyfunc.load_pyfunc``.
    """
    return HuggingFaceWrapper(data_path)