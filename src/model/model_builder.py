import torch
from pathlib import Path
from model.pipeline.model import build_model, CNNModel
from config import model_settings
from loguru import logger


class ModelBuilderService:
    """ModelBuilderService class for building and training a CNN model.
    This class handles the initialization of the model and the training process.
    """

    def __init__(self):
        self.model = CNNModel()

    def train_model(self):
        """Train the CNN model.
        This method initializes the model and starts the training process.
        """
        logger.info("Starting model training")
        build_model()
        