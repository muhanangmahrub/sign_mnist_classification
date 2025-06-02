import torch
from pathlib import Path
from model.pipeline.model import build_model, CNNModel
from config import model_settings
from loguru import logger


class ModelService:
    """
    ModelService class for loading and predicting with a CNN model.
    This class handles the loading of the model and
    making predictions based on input parameters.
    """

    def __init__(self):
        self.model = CNNModel()

    def load_model(self, model_name=model_settings.model_name):
        """
        Load the model from the specified path.
        If the model file does not exist, it will build a new model.
        Args:
            model_name (str): Name of the model to load.
        """
        logger.info(f"Loading model: {model_name}")
        model_path = Path(f"{model_settings.model_save_path}/{model_name}")

        if not model_path.exists():
            logger.warning(f"Model file {model_path} does not exist. \
                           Building a new model.")
            build_model()
            self.model.load_state_dict(torch.load(model_path))
        else:
            logger.info(f"Model file found at {model_path}. Loading model.")
            self.model.load_state_dict(torch.load(model_path))

    def predict(self, input_parameters):
        """
        Make predictions using the loaded model.
        Args:
            input_parameters (torch.Tensor): Input data for prediction.
        Returns:
            numpy.ndarray: Predicted class labels.
        """
        logger.info("Starting prediction")
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        input_data = input_parameters.to(device)
        self.model.to(device)
        with torch.no_grad():
            output = self.model(input_data)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()
        return output
