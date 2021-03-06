from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
from os.path import join
import torchvision.transforms as transforms 
from torch import nn
HEIGHT = 128
WIDTH = 128
DEVICE = 'cpu'


class NIRDataset(Dataset):
  """
  TODO: Implement dataset for colorization
  """
  def __init__(self, input_files="/Users/tbc/Downloads/celeba-dataset/img_align_celeba/img_align_celeba", transform=None):
      self.filenames = glob.glob(join(input_files,'*.jpg'))
      if not transform:
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((HEIGHT,WIDTH))])
      else:
        self.transform = transform

  def __len__(self):
      return len(self.filenames)

  def __getitem__(self, idx):
      img = cv2.imread(self.filenames[idx])
      red_channel_img = img[:,:,2]
      if self.transform:
        img = self.transform(img)
        red_channel_img = self.transform(red_channel_img)
      # img = nn.functional.interpolate(img,(HEIGHT, WIDTH))
      # red_channel_img = nn.functional.interpolate(red_channel_img,(HEIGHT, WIDTH))
      return red_channel_img, img


if __name__ == "__main__":
  """
  Run the main function to sample the dataset as sanity test:
  """
  dataset = NIRDataset()

  while True:
     
      img, img_hires = dataset[np.random.randint(0, len(dataset) - 1)]
      # print(img.cpu().numpy().shape)
      # print(img_hires.cpu().numpy().shape)
      # break
      cv2.imshow("test1", img_hires.cpu().permute(1, 2, 0).numpy())
      cv2.imshow("test2", img.cpu().permute(1, 2, 0).numpy())
      cv2.waitKey(1500)

