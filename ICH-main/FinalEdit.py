from io import BytesIO
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from models import resnext101_32x8d_wsl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms import CenterCrop,ToTensor,Compose

def get_cam(model, img):
    target_layers = [model.module.layer4[-1]]
    cam_extractor = SmoothGradCAMpp(model, target_layers)
    out = model(img)
    activ_map = cam_extractor(out.squeeze(0).argmax().item(), out)

    for name, cam in zip(cam_extractor.target_names, activ_map):
        vis = overlay_mask(to_pil_image(img.squeeze(0)), to_pil_image(cam, mode='F'), alpha=0.5)
    return vis

def get_model(path, n_classes):
    model = resnext101_32x8d_wsl()
    model.fc = torch.nn.Linear(2048, n_classes)
    model = nn.DataParallel(model)
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    return model

def main():
    # Wide mode
    st.set_page_config(layout="wide")

    # Designing the interface
    st.title("ICH Visualiser")
    # For newline
    st.write("\n")
    # Set the columns
    cols = st.columns((1, 1))
    cols[0].header("Input image")
    cols[-1].header("GRADCAM Visualisation")
    st.sidebar.title("Input selection")
    # Disabling warning
    st.set_option("deprecation.showfileUploaderEncoding", False)
    # Choose your own image
    uploaded_file = st.sidebar.file_uploader("Upload files", type=["png", "jpeg", "jpg"])
    if uploaded_file is not None:
        img = Image.open(BytesIO(uploaded_file.read()), mode="r").convert("RGB")
        #cols[0].image(img, use_column_width=True)
        cols[0].image(img, width=300)
    if st.sidebar.button("Predict"):

        if uploaded_file is None:
            st.sidebar.error("Please upload an image first")

        else:
            with st.spinner("Analyzing..."):

                # Preprocess image
                tfms_img = Compose([CenterCrop(200), ToTensor()])
                img_tensor = tfms_img(img).float()
                label_list = ["Epidural",
                            "Intraparenchymal",
                            "Intraventricular",
                            "Subarachnoid",
                            "Subdural",
                            "Any",
                        ]

                model=get_model('models/png_model_e10_final.pt',n_classes=6)
                # Forward the image to the model
                out = model(img_tensor.unsqueeze(0))
         
                out = torch.sigmoid(out)
                out_trgt = torch.round(out)
                
                out_np = out_trgt.detach().numpy()
                print(out_np)
                preds = np.where(out_np == 1)[1]

                probas = torch.round((out) * 100).detach().numpy()[0]
                # print(probas)

                proba_ord = np.argsort(out.detach().numpy())[0][::-1]
                # print(proba_ord)

                label = []
                if len(out_np):
                    for i in preds:
                        label.append(label_list[i])
                else:
                    label.append("ICH Not Present")

                arg_s = {}
                for i in proba_ord:
                    arg_s[label_list[int(i)]] = probas[int(i)]
                       
                visualisation = get_cam(model,img_tensor.unsqueeze(0))
                #cols[-1].image(visualisation, use_column_width=True)
                cols[-1].image(visualisation, width=300)
                #df = pd.DataFrame(data=np.zeros((6,2)),
                     # columns=['Subtype','Predicted Probability'],
                     # index=np.linspace(1, 6, 6, dtype=int))
                df=pd.DataFrame(arg_s,index=[0])
                st.write('Probability Distibution(In Percentage)')
                st.table(data=df)
                if(len(label)==0):
                    final ='ICH Not Present'
                else:
                    final = label[0]
                #print()
                #new = pd.DataFrame(data=probas,columns=("epidural",
                #            "intraparenchymal",
                #            "intraventricular",
                #            "subarachnoid",
                #            "subdural",
                #            "any",
                #        ))
                #st.table(data=arg_s)
                st.write('Conclusion:',final)


if __name__ == "__main__":
    main()
