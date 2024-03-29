import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset

## A function to correct pixel data and rescale intercercepts ob 12 bit images
def dcm_correction(dcm_img):
        x = dcm_img.pixel_array + 1000
        px_mode = 4096
        x[x >= px_mode] = x[x >= px_mode] - px_mode #if there are extra bits in 12-bit grayscale(<=4096)
        dcm_img.PixelData = x.tobytes()
        dcm_img.RescaleIntercept = -1000 #setting a common value across all 12-bit US images
        
#Systemic/linear windowing
def window_image(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        dcm_correction(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept #reconstructing the image from pixels
    img_min = window_center - window_width // 2 #lowest visible value
    img_max = window_center + window_width // 2 #highest visible value
    img = np.clip(img, img_min, img_max)

    return img

#Combining all
def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1, 2, 0)

    return bsb_img


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
