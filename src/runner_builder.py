from model.model_builder import ModelBuilderService
import torch
from loguru import logger


def main():
    """
    Main function to run the model service and make predictions.
    """
    logger.info("Starting the model service")
    ml_svc = ModelBuilderService()
    ml_svc.train_model()
    

if __name__ == "__main__":
    main()
