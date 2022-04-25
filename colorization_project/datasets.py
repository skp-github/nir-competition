from torch.utils.data import Dataset
import numpy as np
import cv2
import glob


HEIGHT = 128
WIDTH = 128
DEVICE = 'cuda'


class NIRDataset(Dataset):
    """
    TODO: Implement dataset for colorization
    """
    def __init__(self):
        self.filenames = glob.glob('*.hdf5')

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return None, None


if __name__ == "__main__":
    """
    Run the main function to sample the dataset as sanity test:
    """
    dataset = NIRDataset()

    while True:
        print(len(dataset))
        # img, img_hires = dataset[np.random.randint(0, len(dataset) - 1)]
        # cv2.imshow("test1", img_hires.cpu().permute(1, 2, 0).numpy())
        # cv2.imshow("test2", img.cpu().permute(1, 2, 0).numpy())
        # cv2.waitKey(1500)
