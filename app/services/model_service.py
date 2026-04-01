import torch
from ml.src.model import UNet3D
from ml.src.inference import inference


class ModelService():
    def __init__(self, weights: str):
        self.model = None
        self.device = None
        self.weights = weights

    def set_device(self, device: str):
        self.device = torch.device(device)

    def create_model(self):
        self.model = UNet3D(in_ch=1, out_ch=1, base_filters=16)

    def load_weights(self):
        weights = torch.load(self.weights)
        self.model.load_state_dict(weights["model_state"])
        self.model.to(self.device)
        self.model.half()
        self.model.eval()

    def predict(self, data):
        mask = inference(data, self.model, device=self.device)
        return mask