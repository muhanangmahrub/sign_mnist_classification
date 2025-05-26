from pathlib import Path
import torch
from model import build_model, CNNModel
from config import settings

class ModelService:
    def __init__(self):
        self.model = CNNModel()

    def load_model(self, model_name=settings.model_name):
        model_path = Path(f"{settings.model_save_path}/{model_name}")

        if not model_path.exists():
            build_model()
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path))

    def predict(self, input_parameters):
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        input_data = input_parameters.to(device)
        self.model.to(device)
        with torch.no_grad():
            output = self.model(input_data)
            output = torch.argmax(output, dim=1)
            output = output.cpu().numpy()
        return output
    
