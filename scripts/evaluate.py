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

def test_step(
        model:nn.Module, 
        test_dataloader:torch.utils.data.DataLoader, 
        device:str
    ):
    model.eval()

    y_pred=[]
    y_true=[]
    with torch.inference_mode():
        for batch, (X,y) in enumerate(test_dataloader):
            X, y = X.to(device),y.to(device)

            y_logits = model(X)

            y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)

            y_pred.append(y_pred_class)
            y_true.append(y)
    
    y_pred=torch.cat(y_pred)
    y_true=torch.cat(y_true)
   
    return y_pred, y_true

def extract_performance(path, device):
    test_transforms=T.Compose([
        T.Resize(size=(128, 128)),
        T.ToTensor() 
    ])

    nw = os.cpu_count()
    validation_dataset = ImageFolder(root=path, transform=test_transforms)
    validation_dataloader = DataLoader(validation_dataset, batch_size=128, shuffle=False, num_workers=nw, pin_memory=True) 

    # carregando o modelo
    model_path = Path("../notebooks")
    state_dict = torch.load(model_path / "model.h5", map_location=device)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k,v in state_dict.items()
    }

    model = Classifier(in_channels=3, hidden_layers=1024, num_classes=len(validation_dataset.classes)).to(device)
    model.load_state_dict(state_dict)

    # obtendo as predições
    y_pred, y_true = test_step(model, validation_dataloader, device=device)
    y_pred=y_pred.detach().cpu().numpy()
    y_true=y_true.detach().cpu().numpy()

    # report
    cr = classification_report(y_true, y_pred, target_names=validation_dataset.classes)
    with open("classification_report.txt", "w") as f:
        f.write(cr)

    # confusion matrix
    labels=list(range(len(validation_dataset.classes)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    sns.heatmap(cm)
    plt.savefig('../artifacts/heatmap_test.png', format='png', dpi=300)

if __name__ == "__main__":
    device="cuda" if torch.cuda.is_available() else "cpu"

    # carregando os dados
    path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")
    base_path = Path(path)
    child_paths = list(base_path.iterdir())
    train_and_validation_path = child_paths[0]
    test_path = child_paths[1] / "test"
    validation_path = train_and_validation_path / "valid"
    extract_performance(validation_path, device)