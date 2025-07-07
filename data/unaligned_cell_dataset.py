import os
from pathlib import Path
import pdb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pytorch_lightning import LightningDataModule
from training.data_utils import CustomTransform, read_files_batch, read_files_pert, make_dataset, make_dataset_seasonet, make_dataset_brats
from training.data_transform import get_transform
from PIL import Image
import rasterio
import nibabel as nib
import torchvision.transforms as transforms


class CellDataset:
    """
    Dataset class for cell image data.

    This class handles the loading and preprocessing of cell image datasets, 
    including the initialization of dataset splits, normalization, and embedding creation.
    """
    
    def __init__(self, args, device):
        """
        Initialize the CellDataset instance.
        
        Args:
            args (argparse.Namespace): Arguments containing dataset configuration.
            device (torch.device): Device to load the data onto (e.g., 'cuda' or 'cpu').
        """

        assert os.path.exists(args.image_path), 'The data path does not exist'
        assert os.path.exists(args.data_index_path), 'The data index path does not exist'

        # Set up the variables
        self.image_path = args.image_path  # Path to the image folder (.pkl file)
        self.data_index_path = args.data_index_path  # Path to data index (.csv file)
        self.embedding_path = args.embedding_path  # Path to embeddings
        self.augment_train = args.augment_train  # Whether to apply data augmentation during training
        self.normalize = args.normalize  # Controls whether to normalize input images
        self.mol_list = args.mol_list  # List of molecules to include
        self.ood_set = args.ood_set  # List of out-of-distribution drugs
        self.trainable_emb = args.batch_correction  # Whether embeddings are trainable
        self.dataset_name = args.dataset_name  # Name of the dataset

        self.batch_correction = args.batch_correction  # If True, perform batch correction
        self.multimodal = args.multimodal  # If True, handle multiple types of perturbations
        if self.trainable_emb or self.batch_correction:
            self.latent_dim = args.latent_dim

        if not self.batch_correction:
            self.add_controls = args.add_controls  # Whether to add controls in non-batch correction mode
            self.batch_key = None
        else:
            self.add_controls = None
            self.batch_key = args.batch_key  # Key for batch correction

        # Fix the training specifics
        self.device = device 

        # Read the datasets
        self.fold_datasets = self._read_folds()

        if args.dataset_n_samples_train is not None:
            len_train = len(self.fold_datasets['train']['SAMPLE_KEY'])
            assert args.dataset_n_samples_train <= len_train, f"Requested subset size ({args.dataset_n_samples_train}) is larger than data avail ({len_train})"
            rng = np.random.default_rng(seed=43)
            indices = rng.choice(len_train, args.dataset_n_samples_train, replace=False)
            for key in self.fold_datasets['train']:
                self.fold_datasets['train'][key] = self.fold_datasets['train'][key][indices]
            print(f"Training on subset of {args.dataset_n_samples_train}")

        self.y_names = np.unique(self.fold_datasets['train']["ANNOT"])  # Sorted annotation names

        # Count the number of compounds 
        self._initialize_mol_names()

        self.y2id = {y: id for id, y in enumerate(self.y_names)}  # Map annotations to IDs
        self.n_y = len(self.y_names)  # Number of unique annotations
        self.iter_ctrl = args.iter_ctrl
        # Initialize embeddings 
        self.initialize_embeddings()

        # Initialize the datasets
        self.fold_datasets = {
            'train': CellDatasetFold('train', 
                                     self.image_path, 
                                     self.fold_datasets['train'],
                                     self.mol2id,
                                     self.y2id, 
                                     self.augment_train, 
                                     self.normalize,
                                     dataset_name=self.dataset_name,
                                     add_controls=self.add_controls, 
                                     batch_correction=self.batch_correction,
                                     batch_key=self.batch_key,
                                     multimodal=self.multimodal,
                                     cpd_name=self.cpd_name,
                                     iter_ctrl=self.iter_ctrl),
            
            'test': CellDatasetFold('test',
                                    self.image_path,
                                    self.fold_datasets['test'],
                                    self.mol2id, 
                                    self.y2id, 
                                    self.augment_train, 
                                    self.normalize,
                                    dataset_name=self.dataset_name,
                                    add_controls=self.add_controls,
                                    batch_correction=self.batch_correction,
                                    batch_key=self.batch_key,
                                    multimodal=self.multimodal,
                                    cpd_name=self.cpd_name,
                                    iter_ctrl=False)}

    def _read_folds(self):
        """
        Extract the filenames of images in the train and test sets.
        
        Returns:
            dict: Dictionary containing train and test datasets.
        """
        # Read the index CSV file
        dataset = pd.read_csv(self.data_index_path, index_col=0)
        
        # Initialize CPD_NAME differently based on the dataset 
        self.cpd_name = "BROAD_SAMPLE" if self.dataset_name == "cpg0000" else "CPD_NAME"

        # Subset the perturbations if provided in mol_list
        if self.mol_list:
            dataset = dataset.loc[dataset[self.cpd_name].isin(self.mol_list)]
        # Remove the leave-out drugs if provided in ood_set
        if self.ood_set is not None:
            dataset = dataset.loc[~dataset[self.cpd_name].isin(self.ood_set)]
        
        # Collect the dataset splits
        dataset_splits = dict()
        
        for fold_name in ['train', 'test']:
            # Divide the dataset in splits 
            dataset_splits[fold_name] = {}
            
            # Divide the dataset into splits
            subset = dataset.loc[dataset.SPLIT == fold_name]
            for key in subset.columns:
                dataset_splits[fold_name][key] = np.array(subset[key])
            if not self.batch_correction:
                if self.dataset_name == 'bbbc021':
                    # Add control and treated flags
                    if not self.add_controls:
                        dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["STATE"] == 1)
                    else:
                        dataset_splits[fold_name]["trt_idx"] = (np.isin(dataset_splits[fold_name]["STATE"], ["trt", "control"]))
                    dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["STATE"] == 0)
                elif self.dataset_name == "rxrx1":
                    assert not self.add_controls, "Controls are not supported for rxrx1 dataset."
                    dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["ANNOT"] == "treated")
                    dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["ANNOT"] == "negative_control")
                elif self.dataset_name == "cpg0000":
                    assert not self.add_controls, "Controls are not supported for cpg0000 dataset."
                    dataset_splits[fold_name]["trt_idx"] = (dataset_splits[fold_name]["STATE"] == "trt")
                    dataset_splits[fold_name]["ctrl_idx"] = (dataset_splits[fold_name]["STATE"] == "control")
        return dataset_splits

    def _initialize_mol_names(self):
        """
        Initialize molecule names and counts based on dataset splits.
        """
        # Get unique mol names 
        if not self.batch_correction:
            if not self.multimodal:
                if self.add_controls:
                    self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name])
                else:
                    self.mol_names = np.unique(self.fold_datasets["train"][self.cpd_name][self.fold_datasets["train"]["trt_idx"]])
                self.n_mol = len(self.mol_names)
            else:
                self.mol_names = {}
                for pert_type in self.y_names:
                    idx_pert = self.fold_datasets["train"]["ANNOT"] == pert_type
                    if self.add_controls:
                        self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert])
                    else:
                        trt_idx = self.fold_datasets["train"]["trt_idx"][idx_pert]
                        self.mol_names[pert_type] = np.unique(self.fold_datasets["train"][self.cpd_name][idx_pert][trt_idx])
                self.n_mol = {key: len(val) for key, val in self.mol_names.items()} 
        else: 
            self.mol_names = np.unique(self.fold_datasets['train'][self.batch_key])
            self.n_mol = len(self.mol_names)

    def initialize_embeddings(self):
        """
        Create and initialize the embeddings for molecules.
        """
        if self.multimodal and (not self.trainable_emb and not self.batch_correction):
            embedding_matrix = []
            mol2id = {}
            self.latent_dim = {}

            for mod in self.y_names:
                embedding_matrix_modality = pd.read_csv(self.embedding_path[mod], index_col=0)
                embedding_matrix_modality = embedding_matrix_modality.loc[self.mol_names[mod]]
                embedding_matrix_modality = torch.tensor(embedding_matrix_modality.values, dtype=torch.float32, device=self.device)
                self.latent_dim[mod] = embedding_matrix_modality.shape[1]
                embedding_matrix_modality = torch.nn.Embedding.from_pretrained(embedding_matrix_modality, freeze=True).to(self.device)
                embedding_matrix.append(embedding_matrix_modality)
                mol2id[mod] = {mol: id for id, mol in enumerate(self.mol_names[mod])}
                
            self.embedding_matrix = torch.nn.ModuleList(embedding_matrix)
            self.mol2id = mol2id
            
        else:
            if self.trainable_emb or self.batch_correction:
                self.latent_dim = self.latent_dim
                self.embedding_matrix = torch.nn.Embedding(self.n_mol, self.latent_dim).to(self.device).to(torch.float32)
            else:
                embedding_matrix = pd.read_csv(self.embedding_path, index_col=0)
                embedding_matrix = embedding_matrix.loc[self.mol_names]
                embedding_matrix = torch.tensor(embedding_matrix.values, dtype=torch.float32, device=self.device)
                self.embedding_matrix = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=True).to(self.device)
            
                self.latent_dim = embedding_matrix.shape[1]
            
            self.mol2id = {mol: id for id, mol in enumerate(self.mol_names)}

class CellDatasetFold(Dataset):
    """
    Dataset fold class for handling train and test splits of cell image data.

    This class inherits from PyTorch's Dataset and provides methods to 
    handle data loading, transformations, and batch processing.
    """
    
    def __init__(self,
                 fold, 
                 image_path, 
                 data, 
                 mol2id,
                 y2id,
                 augment_train=True, 
                 normalize=False, 
                 dataset_name="bbbc021", 
                 add_controls=None,
                 batch_correction=False, 
                 batch_key="BATCH", 
                 multimodal=False, 
                 cpd_name="CPD_NAME",
                 iter_ctrl=False):
        """
        Initialize the CellDatasetFold instance.
        
        Args:
            fold (str): 'train' or 'test' to specify the dataset split.
            image_path (str): Path to the image folder.
            data (dict): Data dictionary containing sample information.
            mol2id (dict): Mapping from molecule names to IDs.
            y2id (dict): Mapping from annotation names to IDs.
            augment_train (bool, optional): Whether to apply data augmentation. Defaults to True.
            normalize (bool, optional): Whether to normalize the images. Defaults to False.
            dataset_name (str, optional): Name of the dataset. Defaults to "bbbc021".
            add_controls (bool, optional): Whether to add controls. Defaults to None.
            batch_correction (bool, optional): Whether to perform batch correction. Defaults to False.
            batch_key (str, optional): Key for batch correction. Defaults to "BATCH".
            multimodal (bool, optional): Whether to handle multiple perturbation types. Defaults to False.
            cpd_name (str, optional): Column name for compound names. Defaults to "CPD_NAME".
        """
        super(CellDatasetFold, self).__init__()

        self.image_path = image_path
        self.fold = fold  
        self.data = data
        self.dataset_name = dataset_name
        self.add_controls = add_controls
        self.batch_correction = batch_correction
        self.multimodal = multimodal
        self.cpd_name = cpd_name
        self.iter_ctrl = iter_ctrl
        # Extract variables
        if self.batch_correction:
            self.file_names = data['SAMPLE_KEY']
            self.mols = data[batch_key]
            self.y = data['ANNOT']
            if dataset_name == "bbbc021":
                self.dose = data['DOSE']
            else:
                self.dose = None
        else:
            self.file_names = {}
            self.mols = {}
            self.y = {}
            self.batch = {}
            if dataset_name == "bbbc021":
                self.dose = {}
            else:
                self.dose = None
            
            for cond in ["ctrl", "trt"]:
                if cond == "trt" and add_controls:
                    self.file_names[cond] = self.data['SAMPLE_KEY']
                    self.mols[cond] = self.data[cpd_name]
                    self.y[cond] = self.data['ANNOT']
                    if dataset_name == "bbbc021":
                        self.dose[cond] = self.data['DOSE']
                else:
                    self.file_names[cond] = self.data['SAMPLE_KEY'][self.data[f"{cond}_idx"]]
                    self.mols[cond] = self.data[cpd_name][self.data[f"{cond}_idx"]]
                    self.y[cond] = self.data['ANNOT'][self.data[f"{cond}_idx"]]
                    batch_key = "PLATE" if dataset_name == "cpg0000" else "BATCH"
                    self.batch[cond] = self.data[batch_key][self.data[f"{cond}_idx"]]
                    if dataset_name == "bbbc021":
                        self.dose[cond] = self.data['DOSE'][self.data[f"{cond}_idx"]]
        del data 
        
        # Whether to perform training augmentation
        self.augment_train = augment_train
        
        # One-hot encoders 
        self.mol2id = mol2id
        self.y2id = y2id
        
        # Transform only the training set and only if required
        self.transform = CustomTransform(augment=(self.augment_train and self.fold == 'train'), normalize=normalize)

        
    def __len__(self):
        """
        Return the total number of samples.
        
        Returns:
            int: Number of samples.
        """
        if self.batch_correction:
            return len(self.file_names)
        else:
            return len(self.file_names["ctrl"]) if self.iter_ctrl else len(self.file_names["trt"])

    def __getitem__(self, idx):
        """
        Generate one example datapoint.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing the image tensor, one-hot encoded molecule, annotation ID, dose, and file name.
        """
        # Image must be fetched from disk
        if self.batch_correction:
            return read_files_batch(self.file_names, 
                                    self.mols,
                                    self.mol2id,
                                    self.y2id, 
                                    self.y, 
                                    self.transform,
                                    self.image_path, 
                                    self.dataset_name, 
                                    idx)
        else:
            return read_files_pert(self.file_names, 
                                   self.mols, 
                                   self.mol2id, 
                                   self.y2id, 
                                   self.dose, 
                                   self.y, 
                                   self.transform, 
                                   self.image_path, 
                                   self.dataset_name,
                                   idx,
                                   self.multimodal,
                                   self.batch,
                                   self.iter_ctrl,)

