
import os
import pandas as pd
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.folder import pil_loader

def get_study_level_data(BASE_DIR):
    study_data = {}
    study_label = {'positive': 1, 'negative': 0}
    
    study_data= pd.DataFrame(columns=['Path', 'Count', 'Label'])
    i = 0
    for study in os.listdir(BASE_DIR): # for each study in that patient folder
        print(study,"study")
        label = study_label[study.split('_')[1]] # get label 0 or 1
        path = BASE_DIR + '/' + study + '/' # path to this study
        study_data.loc[i] = [path, len(os.listdir(path)), label] # add new row
        i+=1
    return study_data

class ImageDataset(Dataset):
    """training dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): a pandas DataFrame with image path and labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        study_path = self.df.iloc[idx, 0]
        count = self.df.iloc[idx, 1]
        images = []
        for i in range(count):
            image = pil_loader(study_path + 'image%s.png' % (i+1))
            images.append(self.transform(image))
        images = torch.stack(images)
        label = self.df.iloc[idx, 2]
        sample = {'images': images, 'label': label}
        return sample

def get_dataloaders(data, batch_size=8, study_level=False):
    '''
    Returns dataloader pipeline with data augmentation
    '''
    data_transforms= transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    image_datasets = ImageDataset(data, transform=data_transforms)
    dataloaders =  DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloaders

def plot(model, dataloaders):

  predictions= []
  label_list= []

  for i, data in enumerate(dataloaders):
        print(i, end='\r')
        labels = data['label'].type(torch.FloatTensor)
        inputs = data['images'][0]
        print(inputs.shape, "Input shape")
        # wrap them in Variable
        inputs = Variable(inputs.cuda())
        labels = Variable(labels.cuda())
        # forward

        outputs = model(inputs)
        outputs = torch.mean(outputs)

        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        temp = inputs.cpu().numpy().transpose(0, 3, 2, 1)
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(20, 20))
        
        for ax, img in zip(axs.ravel(), temp):
            ax.imshow(img)
            ax.title.set_text(f'Prediction: {preds} - Actual target: {labels.cpu().numpy()[0]}')
            ax.axis('off')
        plt.show()
        



        




