import streamlit as st
from pathlib import Path
import torch
from scripts.model import Classifier
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

device="cuda" if torch.cuda.is_available() else "cpu"
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
        'Tomato___healthy'
    ] 

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY")
)

def load_model():
    model_path = Path("notebooks")
    state_dict = torch.load(model_path / "model.h5", map_location=device)
    state_dict = {
        k.replace("_orig_mod.", ""): v for k,v in state_dict.items()
    }

    model = Classifier(in_channels=3, hidden_layers=1024, num_classes=38).to(device)
    model.load_state_dict(state_dict)
    return model

def get_transforms():
    return T.Compose([
        T.Resize(size=128),
        T.ToTensor()
    ])

def predict(model, img):
    with torch.inference_mode():
        y = model(img.unsqueeze(0).to(device))
        y_predicts = torch.softmax(y, dim=1)
    
    values, indices = torch.topk(y_predicts, 3)
    return values, indices # indice das 3 maiores probabilidades

st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("", ["Home", "Sobre", "Disease Recognition"])

if app_mode == "Home":
    image_path = "artifacts/home_page.png"
    st.image(image_path, width="content")
    st.markdown("""
    # Como funciona ?
                
    1. `upload`: selecione **Disease Recognition** no menu ao lado e faça o upload da imagem da planta.
    2. `análise`: Nosso sistema processará a imagem usando Aprendizado Profundo.
    3. `resultado`: Visualize os resultados e recomendações de ações futuras. 
""")
elif app_mode == "Sobre":
    st.header("Sobre")
    st.markdown("""
        ### Sobre o dataset

O dataset utilizado neste projeto é uma versão recriada por meio de técnicas de *offline augmentation* a partir do dataset original de doenças em plantas.

- Contém aproximadamente **87 mil imagens RGB** de folhas saudáveis e doentes.
- As imagens estão organizadas em **38 classes diferentes**.
- O conjunto de dados foi dividido em:
  - **80%** para treinamento
  - **20%** para validação
- A estrutura de diretórios foi preservada para facilitar o carregamento dos dados com bibliotecas como PyTorch e TensorFlow.
- Além disso, foi criado posteriormente um diretório contendo **33 imagens de teste** para fins de predição e avaliação visual do modelo.

### Link do Dataset

- https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset?select=New+Plant+Diseases+Dataset%28Augmented%29[oaicite:0]{index=0}                
""")
elif app_mode == "Disease Recognition":
    
    st.header("Disease Recognition")
    test_image = st.file_uploader("Escolha uma imagem:")
    if test_image:
        st.image(test_image, width='content')

    if st.button("Predição") and test_image:
        img=Image.open(test_image)

        model = load_model()
        transforms = get_transforms()
        img = transforms(img) # converte ela pro formato adequado
        values, indices = predict(model, img)
        values=values.detach().cpu().numpy()[0]
        indices=indices.detach().cpu().numpy()[0]
        labels = [classes[i] for i in indices]

        # exibindo para o usuário os scores
        df = pd.DataFrame({
            'Classe': labels,
            'Probabilidade': values
        })


        top1 = df.iloc[0]
        st.success(
            f"Predição: {top1['Classe']} "
            f"({top1['Probabilidade']*100:.2f}%)"
        )

        st.divider()

        with st.container(border=True):
            st.markdown("### 🏆 Ranking")
            for i, row in df.iterrows():
                st.write(
                    f"#{i+1} - {row['Classe']} "
                    f"({row['Probabilidade']*100:.2f}%)"
                )

                st.progress(float(row["Probabilidade"]))

        st.divider()

        # código do Geeks for Geeks
        conv_weights=[]
        conv_layers=[]
        total_conv_layers=0

        for module in model.feature_extractor.children():
            if isinstance(module, nn.Conv2d):
                total_conv_layers+=1
                conv_weights.append(module.weight)
                conv_layers.append(module)

        input_img = img.unsqueeze(0).to(device)
        feature_maps=[]
        layer_names = []
        for layer in conv_layers:
            input_img = layer(input_img)
            feature_maps.append(input_img)
            layer_names.append(str(layer))

        # Display feature maps shapes
        print("\nFeature maps shape")
        for feature_map in feature_maps:
            print(feature_map.shape)

        # Process and visualize feature maps
        processed_feature_maps = []  # List to store processed feature maps
        for feature_map in feature_maps:
            feature_map = feature_map.squeeze(0)  # Remove the batch dimension
            mean_feature_map = torch.sum(feature_map, 0) / feature_map.shape[0]  # Compute mean across channels
            processed_feature_maps.append(mean_feature_map.data.cpu().numpy())

        # Display processed feature maps shapes
        print("\n Processed feature maps shape")
        for fm in processed_feature_maps:
            print(fm.shape)

        # Plot the feature maps
        fig = plt.figure(figsize=(4, 4))
        for i in range(len(processed_feature_maps)):
            ax = fig.add_subplot(2, 4, i + 1)
            ax.imshow(processed_feature_maps[i])
            ax.axis("off")
            # ax.set_title(layer_names[i].split('(')[0], fontsize=30)

        feature_map_filepath = "artifacts/feature_map.png" 
        plt.tight_layout()
        plt.savefig(feature_map_filepath, format='png', dpi=300)

        st.markdown("### [INTERPRETABILIDADE] Mapa de características do Modelo")
        if feature_map_filepath:
            st.image(feature_map_filepath, width='content')

        st.divider()
        st.markdown("### Recomendações")

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a plant disease specialist with expertise in plant health, diseases, symptoms, causes, and treatments.

The user will provide the plant disease they believe is affecting their plant. Your task is to respond in Brazilian Portuguese (pt-BR) with a short and practical initial recommendation.

Response rules:
- Keep the response concise and beginner-friendly.
- Use Markdown formatting.
- Limit the response to a maximum of 4 short sections.
- Avoid long explanations or excessive technical details.
- Do not generate long lists with many subtopics.
- Focus only on:
  1. Possible cause of the disease
  2. Main symptoms to confirm
  3. Recommended treatment/care
  4. Prevention tips (optional and brief)
- If the class informed by user containing the prefix 'healthy', you must give tips to continue to preventing diseases.

Formatting requirements:
- Use short paragraphs or bullet points.
- Do not exceed approximately 200 words.
- Do not include disclaimers, introductions, or conclusions unless necessary.
- Do not create extensive diagnostic guides.

The goal is to provide a quick and useful first recommendation for the plant owner."""
                },
                {
                    "role": "user",
                    "content": str(top1['Classe']),
                }
            ],
            model="openai/gpt-oss-20b",
        )
        
        with st.container(border=True):
            st.markdown(chat_completion.choices[0].message.content)