import numpy as np
import streamlit as st
from PIL import Image
import os
import torch
from torch.autograd import Variable
from densenet import densenet169
from torchvision import transforms

MODEL_PATH= 'G:\python\DeepLearning\DeepLearningProjects\muraprov1\model.pth'

model = densenet169(pretrained=True)
model = model.cuda()

title= st.title(' Wrist X-Ray Abnormality Detection')

#Loading image file using Pickle
@st.cache
def load_image(image_file):
    # MAX_SIZE = (224, 224)
    img = Image.open(image_file)
    img= img.convert('RGB')
    # img= img.resize(MAX_SIZE) 
    return img 

loader = transforms.Compose([
     transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def image_loader(img):
    """load image, returns cuda tensor"""
    image = loader(img).float()
    image = Variable(image)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()

image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
if image_file is not None:

    # To See Details
    # st.write(type(image_file))
    # st.write(dir(image_file))
    file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
    st.write(file_details)

    img = load_image(image_file)
    print(len(img.getbands()), "Bands")
    st.image(img)
    st.write(img.size)
    
    #Prediction Loop
    with torch.no_grad():

        model.load_state_dict(torch.load(MODEL_PATH, map_location= 'cuda'))
        model.eval()

        print("Model Loaded ")
  
        inputs = image_loader(img)
        outputs = model(inputs).type(torch.cuda.FloatTensor)
        if outputs > 0.5 :
            st.write("Abnormality Present")
