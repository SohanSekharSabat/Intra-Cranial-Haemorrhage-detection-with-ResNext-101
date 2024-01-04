#importing libraries
import os
import random
import glob
import pandas as pd
import numpy as np
import pydicom 
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck
import torch
import torch.optim as optim
from albumentations import Compose, ShiftScaleRotate, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast,CenterCrop
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset,Subset
import cv2
from sklearn.decomposition import PCA
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, classification_report, recall_score,f1_score,accuracy_score
from tqdm import notebook as tqdm

#declaring file paths
dir_csv = '/home/navneeth/ICH_Code/input/'
# test_images_dir = '/home/navneeth/ICH_Code/input/stage_2_test/'
train_images_dir = '/home/navneeth/ICH_Code/png dataset/stage_1_train_png_224x/'
train_metadata_csv = '/home/navneeth/ICH_Code/input/train_metadata_noidx.csv'
test_metadata_csv = '/home/navneeth/ICH_Code/input/test_metadata_noidx.csv'
tb  = SummaryWriter('runs/ich_detection_experiment_12')
#PARAMS
n_classes = 6
n_epochs = 10
batch_size = 32

COLS = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']

# Read train and test data
train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))
test = pd.read_csv(os.path.join(dir_csv, 'stage_2_sample_submission.csv'))

#Preparing data
train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']

#remove images that are not saved properly as png 
png = glob.glob(os.path.join(train_images_dir, '*.png'))
png = [os.path.basename(png)[:-4] for png in png]
png = np.array(png)


train = train[train['Image'].isin(png)]
print('csv')
train.to_csv('train.csv',index=False)

#Declaring dataset class
class IntracranialDataset(Dataset):
  def __init__(self, csv_file, path, labels, transform=None):
        self.path = path
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.labels = labels

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):
      img_id = self.data.loc[idx, 'Image']
      img_name = os.path.join(self.path, img_id + '.png')
      img = cv2.imread(img_name)   
#      img_id = self.data.loc[idx, 'ImageId']
     
     #try:
      #    img = pydicom.dcmread(self.path, img_id + '.dcm')
          #img = bsb_window(dicom)
      #except:
       #   img = np.zeros((512, 512, 3))
      
      if self.transform:       
          augmented = self.transform(image=img)
          img = augmented['image']   
          
      if self.labels:
          
          labels = torch.tensor(
              self.data.loc[idx, ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any']])
          return {'image_id': img_id, 'image': img, 'labels': labels}         
      else:      
          
          return {'image_id': img_id, 'image': img}

print('data')
#Dataloader
transform_train = Compose([CenterCrop(200,200),HorizontalFlip(),
                           ShiftScaleRotate(),
                           RandomBrightnessContrast(),
                           ToTensorV2()])
train_dataset = IntracranialDataset(
    csv_file='train.csv', path=train_images_dir, transform=transform_train, labels=True)

valid_dataset = IntracranialDataset(
    csv_file='train.csv', path=train_images_dir, transform=transform_train, labels=True)

valid_dataset = torch.utils.data.Subset(valid_dataset,range(0,50000))

train_dataset = torch.utils.data.Subset(train_dataset,range(50000,len(train_dataset)-1))

data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(len(data_loader_train))

data_loader_valid = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
print(len(data_loader_valid))
print('model')

#Importing the model
model_urls = {
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
}

#Setting up the model
def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    model.load_state_dict(state_dict)
    return model

def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

#Declaring the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = resnext101_32x8d_wsl()

#Declaring fully connected layer(2048 are the input features)
model.fc = torch.nn.Linear(2048,n_classes)

#Using data parallel to use all GPUs
model = torch.nn.DataParallel(model)

#Loading model to the device and declaring loss function and optimiser
model.to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

#Training starts

for epoch in range(n_epochs):

    print('Epoch {}/{}'.format(epoch + 1, n_epochs))
    print('-' * 10)

#put model in training mode
    model.train()
    tr_loss = 0
    tr_correct = 0

    #tk0 = tqdm.tqdm(data_loader_train, desc="Iteration")
    #new = []
    #train_pred = np.zeros((len(train_dataset) * n_classes, 1)
    #iterate over the batches
    for step, batch in enumerate(data_loader_train):

        inputs = batch["image"]
#         print(inputs.shape)
        labels = batch["labels"]
#         print(labels.shape)

        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
#model predictions
        outputs = model(inputs)
#         print(outputs.shape)
#         print(outputs)
        loss = criterion(outputs, labels)
        # preds = (torch.sigmoid(outputs) >=0.5).float()*1
#         print(preds.shape)
        #Array with sigmoid function applied
        new1=torch.sigmoid(outputs).detach().cpu()>=0.5
       #Backpropagation
       loss.backward()
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
#Add training loss
        tr_loss += loss.item()
        # tr_correct += torch.sum(preds == labels)
#         print(tr_correct)
        tr_correct += torch.sum(new1 == labels.cpu())
        optimizer.step()
        optimizer.zero_grad()

        if step % 512 == 0:
            epoch_loss = tr_loss / (step + 1)
            print('Training Loss at {}: {:.4f}'.format(step, epoch_loss))

    epoch_loss = tr_loss / len(data_loader_train)
    print('Training Loss: {:.4f}'.format(epoch_loss))
    print('-----------------------')
    #Tensorboard code for visualisations
    tb.add_scalar("Training Loss", tr_loss, epoch)
    # tb.add_scalar("Training Correct preds", tr_correct, epoch)
    tb.add_scalar("Training Accuracy", tr_correct/ len(train_dataset), epoch)
    print('Finished Training!')

#Put model in evalauation mode
    model.eval()
    tr_loss = 0
    tr_correct = 0

   # auc_preds = []
   # auc_truths = []


    print('Validation starts...')
#Arrays to store metrics for each batch
    rec=[]
    acc=[]
    f1=[]
    #new=[]
    #test_pred = np.zeros((len(valid_dataset)*n_classes,1))
    for i,x_batch in enumerate(data_loader_valid):
        #Declare images and labels of the batch and put it into the device
        x_image = x_batch['image']
        x_image = x_image.to(device,dtype=torch.float)
        labels = x_batch["labels"]
        labels = labels.to(device, dtype=torch.float)
        with torch.no_grad():
            #create predictions and calculate loss
            pred = model(x_image)
            loss=criterion(pred,labels)
    #        test_pred[(i*batch_size*n_classes):((i+1)*batch_size*n_classes)]=torch.sigmoid(pred).detach().cpu().reshape((len(x_image)*n_classes,1))
     #       new.append(torch.sigmoid(pred).detach().cpu())
            new1=torch.sigmoid(pred).detach().cpu()>=0.5

            tr_loss += loss.item()
            tr_correct += torch.sum(new1 == labels.cpu())
            #loss=criterion(pred,labels)
            #tr_loss+=loss.item()
            #tb.add_scalar("Validation Loss", tr_loss, epoch)
            #tb.add_scalar("Validation Accuracy", tr_correct/ len(validation_dataset), epoch)
            #Add metrics to the array
            acc.append(accuracy_score(new1.float(),labels.cpu()))
            f1.append(f1_score(new1.float(),labels.cpu(),average=None,zero_division=1))
            #jacc.append(jaccard_score(new1.float(),labels.cpu(),average=None,zero_division=1))
            #roc.append(roc_auc_score(new1.float(),labels.cpu(),average=None))
            rec.append(recall_score(new1.float(),labels.cpu(),average=None,zero_division=1))
    
            print(classification_report(new1.float(),labels.cpu(),zero_division=1))

    epoch_loss = tr_loss / len(data_loader_valid)
    
    print('Validation  Loss: {:.4f}'.format(epoch_loss))
    print('-----------------------')

#Add to tensorboard graph
    tb.add_scalar("Validation Loss", tr_loss, epoch)
    tb.add_scalar("Validation Accuracy", tr_correct/ len(valid_dataset), epoch)
        
        #find average of metrics of accuracy,f1 score and recall
    ans=sum(acc)/len(acc)
    ans1 = sum(f1)/len(f1)
    ans2 = sum(rec)/len(rec)
    #ans3 = sum(roc)/len(roc)

    print('Accuracy:',ans)
    print('F1:',ans1)
    print("Recall:",ans2)
   # print("ROC_AUC",ans3)
#save model
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    # 'amp': amp.state_dict()
}
torch.save(checkpoint, 'models/png_model.pt')
tb.close()
