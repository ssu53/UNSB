from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random
import util.util as util
import numpy as np
import rasterio

class UnalignedSeasonetDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        print("Initialising UnalignedSeasonetDataset!")
        BaseDataset.__init__(self, opt)
        
        self.opt = opt
        self.dir_A = None
        self.dir_B = None
        A_paths_train, B_paths_train, A_paths_test, B_paths_test = self._make_dataset_seasonet(opt.dataroot)
        if opt.phase == 'train':
            self.A_paths = A_paths_train
            self.B_paths = B_paths_train
        if opt.phase == 'test':
            self.A_paths = A_paths_test
            self.B_paths = B_paths_test
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
    
    def __getitem__(self, idx):
        """
        Generate one example datapoint.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Tuple containing image A and image B tensors.
        """
        A_path = self.A_paths[idx % self.A_size]
        if self.opt.serial_batches:
            idx_B = idx % self.B_size
        else:
            idx_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[idx_B]

        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        imgA = self._load_image_seasonet(A_path, transform)
        imgB = self._load_image_seasonet(B_path, transform)

        return {'A': imgA, 'B': imgB, 'A_paths': A_path, 'B_paths': B_path}

    @staticmethod
    def _load_image_seasonet(image_path, transform):
        """
        Load an image from the given path and preprocess it.

        Args:
            image_path (Path): Path to the image file.

        Returns:
            Tensor: Preprocessed image tensor.
        """
        NORMALISATION_FACTOR = 2000. # suggested by empirical distribution of SeasoNet
        with rasterio.open(image_path) as src:
            image = src.read()
        image = np.stack([image[0], image[1], image[2]], axis=-1)
        image = image.astype(np.float32)
        image = (image / NORMALISATION_FACTOR) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)
        image_transformed = transform(image)
        return image_transformed

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return len(self.A_paths)

    @staticmethod
    def _make_dataset_seasonet(dir):

        MODALITY = '10m_RGB'

        seasonA = 'summer'
        seasonB = 'winter'

        with open(f"{dir}/index_{seasonA}_train.txt") as file:
            A_paths_train = [f"{dir}/{seasonA}/grid1/{line.rstrip()}/{line.rstrip()}_{MODALITY}.tif" for line in file]
        with open(f"{dir}/index_{seasonB}_train.txt") as file:
            B_paths_train = [f"{dir}/{seasonB}/grid1/{line.rstrip()}/{line.rstrip()}_{MODALITY}.tif" for line in file]
        with open(f"{dir}/index_{seasonA}_test.txt") as file:
            A_paths_test = [f"{dir}/{seasonA}/grid1/{line.rstrip()}/{line.rstrip()}_{MODALITY}.tif" for line in file]
        with open(f"{dir}/index_{seasonB}_test.txt") as file:
            B_paths_test = [f"{dir}/{seasonB}/grid1/{line.rstrip()}/{line.rstrip()}_{MODALITY}.tif" for line in file]

        return A_paths_train, B_paths_train, A_paths_test, B_paths_test
