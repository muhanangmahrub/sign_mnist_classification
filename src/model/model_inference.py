import torch
from pathlib import Path
from model.pipeline.model import build_model, CNNModel
from config import model_settings
from loguru import logger


class ModelInferenceService:
    """ModelInferenceService class for loading and predicting with a CNN model."""

    def __init__(self):
        self.model = CNNModel()
        self.model_name = model_settings.model_name
        self.model_path = model_settings.model_save_path

    def load_model(self):
        """
        Load the model from the specified path.
        If the model file does not exist, it will raise an error.
        """
        logger.info(f"Loading model: {self.model_name}")
        model_path = Path(f"{self.model_path}/{self.model_name}")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file {model_path} does not exist. "
                "Please ensure the model is built before loading.")
        
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
