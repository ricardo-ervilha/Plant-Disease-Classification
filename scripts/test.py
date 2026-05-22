import torch
from pathlib import Path
import kagglehub
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from torch import nn
from tqdm import tqdm
from model import Classifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

def test(random_image_path, device, classes):
    test_transforms=T.Compose([
        T.Resize(size=(128, 128)),
        T.ToTensor() 
    ])

    img = test_transforms(Image.open(random_image_path))

    # carregando o modelo
    model_path = Path("../notebooks")
    state_dict = torch.load(model_path / "model.h5", map_location=device)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k,v in state_dict.items()
    }

    model = Classifier(in_channels=3, hidden_layers=1024, num_classes=38).to(device)
    model.load_state_dict(state_dict)

    with torch.inference_mode():
        y = model(img.unsqueeze(0).to(device))
        label = torch.argmax(torch.softmax(y, dim=1), dim=1)

    plt.imshow(img.permute(1,2,0))
    plt.title(f"Correct label: {random_image_path.name}\nPredicted Label: {classes[label]}")
    plt.axis(False)
    plt.show()
   
if __name__ == "__main__":
    classes = ['Apple___Apple_scab', 'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'] 

    device="cuda" if torch.cuda.is_available() else "cpu"

    # carregando os dados
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    base_path = Path(path)
    child_paths = list(base_path.iterdir())
    train_and_validation_path = child_paths[0]
    test_path = child_paths[1] / "test"
    
    imgs=list(test_path.iterdir())
    random_img = np.random.randint(0, len(imgs))

    test(imgs[random_img], device, classes)