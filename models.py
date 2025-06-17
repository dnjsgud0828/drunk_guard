import torch
from torchvision import transforms
from PIL import Image
import cv2
# from efficientnet_pytorch import EfficientNet
from architectures.fornet import EfficientNetB4
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'drunk_guard')))

class DrunkClassifier:
    def __init__(self, model_path):
        self.model = EfficientNetB4() #모델 생성
        # self.model._fc = torch.nn.Linear(self.model._fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False) #가중치 load
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def predict(self, face_img):
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        tensor = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(tensor)
            _, pred = torch.max(output, 1)
        return 'Drunk' if pred.item() == 1 else 'Sober'
