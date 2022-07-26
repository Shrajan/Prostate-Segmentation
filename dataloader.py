# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : dataloader.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 14 March 2021
    Program Description:
        This program contains contains training data generator.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    14 March 2021      |     Dejan Kostyszyn      |  Implemented necessary functions.   |
    |    01 August 2021     |     Shrajan Bhandary     |    Changed image reading method.    |
    |    05 August 2021     |     Dejan Kostyszyn      |    Implemented cross-validation.    |
    |    25 August 2021     |     Shrajan Bhandary     | Unit tests to ensure data validity. |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |    26 July 2022       |     Shrajan Bhandary     |    Optimized data loading method.   |
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS
import os, torch, utils, json, random, nrrd
import numpy as np
from os.path import join
from natsort import natsorted
import SimpleITK as sitk
torch.manual_seed(2021)

# IMPLEMENTATION

class Dataset(torch.utils.data.Dataset):
    """
    Loads data and corresponding label and returns pytorch float tensor. In detail:
    * Retrieves complete dataset information from the JSON file.
    * Reads file paths from the JSON file.
    * Segregates files into train, val and test.
     
    param opt               # Command line arguments from the user or default values from the options file.
    param split_type        # Type of data split: options = ["train", "val", "test"]

    The data should have the following folder structure. Train and test can be in separate folders if
    cross-validation is not being done.

    data_root/
    ├── train_and_test/
    |   └── original_gts/
    │      ├── 000_label.nrrd
    │      ├── 001_label.nrrd
    |      ...
    │   └── 000/
    │      ├── data.nrrd
    │      ├── label.nrrd
    |      ...
    │   └── 001/
    │      ├── data.nrrd
    │      ├── label.nrrd
    |       ...
    |   ├──dataset_info.json
    
    """
    def __init__(self, opt=None, split_type="train"):
        
        # Options.
        self.opt = opt

        # Data path.
        self.data_root = opt.data_root

        # Type of data split.
        self.split_type = split_type

        # Variable to store the complete information about the dataset.
        self.dataset_info = None

        # Get the required information about the volumes.
        self.patient_ids, self.data_paths, self.label_paths, self.mask_paths = self.read_data_paths()
        
        # Get data indices based on the split type, k-fold and fold.
        self.determine_split_indices(opt=self.opt, split_type=self.split_type)

        # Variable to ensure all the volumes have the spacing voxel spacings.
        self.voxel_spacing = None

        # Variable store all the loaded files.
        self.loadedFiles = {}
    
        # This variable is set, so that the main memory does not explode.
        if "store_loaded_data" in self.opt:
            if not self.opt.store_loaded_data:
                self.max_nr_of_files_to_store = 0
            else:
                self.max_nr_of_files_to_store = 170

        # Determine the data indices of the current split.
        self.split_idx = self.get_split_idx(split_type=self.split_type)


    def __len__(self):
        return len(self.split_idx)
        
    def get_split_idx(self, split_type="train"):
        if split_type == "train":
            return self.train_idx
        elif split_type == "test":
            return self.test_idx
        elif split_type == "val":
            return self.val_idx

    def nr_of_patients(self):
        return self.__len__()

    def get_dataset_info(self):
        """
        Return the complete information about the dataset.
        """
        return self.dataset_info

    def read_data_paths(self):
        """
        Reads data paths after retrieving "dataset_info.json" file.
        """
        
        data_root = self.opt.data_root

        # Reads all the information about the dataset from 'dataset_info.json' file.
        jsonFile = open(join(data_root, "dataset_info.json"))
        self.dataset_info = json.load(jsonFile)

        # Read patient ids.
        patient_ids = list(self.dataset_info["files"].keys())
        patient_ids = natsorted(patient_ids)

        if len(patient_ids) != self.dataset_info["numFiles"]:
            raise Exception("The current number of files and actual count don't match. \
                            Please check the dataset.json file.")

        # Read data and label paths.
        data_paths = []
        label_paths = []
        mask_paths = []

        for p_id in patient_ids:
            data_paths.append(join(data_root, 
                                self.dataset_info["files"][p_id]["new_volume_info"]["dataFilePath"]))
            label_paths.append(join(data_root, 
                                self.dataset_info["files"][p_id]["new_volume_info"]["labelFilePath"]))
            
            # Future work for prostate cancer detection.
            m_path = join(data_root, p_id, "prostate.nrrd")
            if os.path.exists(m_path):
                mask_paths.append(m_path)
            else:
                mask_paths.append(None)
            
        return patient_ids, data_paths, label_paths, mask_paths

    def shuffle_patch_choice(self):
        """
        It is randomly decided for which patients only background patches
        shall be returned.
        """
        # Randomly choose 20% of val patches to include only background.
        self.no_prostate_patch_idx = random.sample(list(self.val_idx), int(len(self.val_idx)*0.2))

    def different_spacing(self, spacing_1, spacing_2, tolerance=0.0001):
        """
        Checks whether the spacings match with a tolerance.
        """
        if abs(spacing_1[0]-spacing_2[0]) > tolerance:
            return True
        if abs(spacing_1[1]-spacing_2[1]) > tolerance:
            return True
        if abs(spacing_1[2]-spacing_2[2]) > tolerance:
            return True
        return False

    def determine_split_indices(self, opt=None, split_type="train"):
        """
        Splits the patients into train, validation and train based on the k-fold and fold parameters.
        """
        # Split data into training, validation and testing set.
        self.data_idx = np.arange(len(self.patient_ids))
        np.random.seed(self.opt.seed)

        # If shuffle is required.
        if not self.opt.no_shuffle == True:
            self.data_idx = np.random.permutation(self.data_idx)

        # No cross-validation.
        if opt.k_fold == 0:
            if split_type == "test":
                self.test_idx = self.data_idx
            else:    
                self.train_size = int(0.8*len(self.patient_ids))
                self.val_size = len(self.patient_ids) - self.train_size
                self.train_idx = self.data_idx[:self.train_size]
                self.val_idx = self.data_idx[self.train_size:]

        # 8-fold cross-validation.
        elif opt.k_fold == 8:
            actualFold = opt.fold
            fold = int(actualFold * (len(self.patient_ids)/8))
            
            self.test_idx = self.data_idx[fold:fold+int(len(self.patient_ids)/8)]
            
            if actualFold == 7:
                self.val_idx = self.data_idx[0:int(len(self.patient_ids)/8)]
            else:
                self.val_idx = self.data_idx[fold+int(len(self.patient_ids)/8):fold+2*int(len(self.patient_ids)/8)]
            
            self.train_idx = list(self.data_idx)
            for value in self.test_idx:
                self.train_idx.remove(value)
            for value in self.val_idx:
                self.train_idx.remove(value)

        # 5-fold cross-validation.
        elif opt.k_fold == 5:  
            self.train_val_size = int(0.8*len(self.patient_ids))
            self.test_size = len(self.patient_ids) - self.train_val_size

            if opt.fold == 0:
                self.test_idx = self.data_idx[:self.test_size]
                self.train_val_idx = self.data_idx[self.test_size:]

            elif opt.fold == 1:
                self.test_idx = self.data_idx[self.test_size:self.test_size*2]
                list1, list2 = list(self.data_idx[:self.test_size]), list(self.data_idx[self.test_size*2:])
                self.train_val_idx = list1 + list2

            elif opt.fold == 2:
                self.test_idx = self.data_idx[self.test_size*2:self.test_size*3]
                list1, list2 = list(self.data_idx[:self.test_size*2]), list(self.data_idx[self.test_size*3:])
                self.train_val_idx = list1 + list2
            
            elif opt.fold == 3:
                self.test_idx = self.data_idx[self.test_size*3:self.test_size*4]
                list1, list2 = list(self.data_idx[:self.test_size*3]), list(self.data_idx[self.test_size*4:])
                self.train_val_idx = list1 + list2

            elif opt.fold == 4:
                self.test_idx = self.data_idx[self.test_size*4:]
                self.train_val_idx = list(self.data_idx[:self.test_size*4])
            
            self.train_val_idx = np.random.permutation(self.train_val_idx)
            self.train_idx = self.train_val_idx[:int(0.8*len(self.train_val_idx))]
            self.val_idx = self.train_val_idx[int(0.8*len(self.train_val_idx)):]

    def __getitem__(self, idx):
        """
        Read patient id, data and label and return them.
        """
        current_idx = self.split_idx[idx]
        p_id = self.patient_ids[current_idx]

        # return already loaded data
        if p_id not in self.loadedFiles:
            
            # Read data from memory.
            data = sitk.ReadImage(self.data_paths[current_idx])

            # Read label from memory.
            label = sitk.ReadImage(self.label_paths[current_idx])

            # Get the voxel spacing of the data and the label.
            data_spacing = data.GetSpacing()
            label_spacing = label.GetSpacing()

            # Check whether the input and ground truth have same spacing.
            if self.different_spacing(data_spacing, label_spacing):
                print("The spacing of the input is: {}. ".format(data_spacing))
                print("The spacing of the label is: {}. ".format(label_spacing))
                raise Exception("The spacing of data and label don't match.")

            # Set the voxel spacing of the dataset.
            if self.voxel_spacing is None:
                self.voxel_spacing = data_spacing

            # Make sure that all samples of the dataset have the same spacing.
            elif self.different_spacing(self.voxel_spacing, data_spacing):
                print("The spacing of the previous input is: {}. ".format(self.voxel_spacing))
                print("The spacing of the current input is: {}. ".format(data_spacing))
                raise Exception("The spacing of current input and previous input don't match.")

            data = sitk.GetArrayFromImage(data).astype(np.float32).transpose(2, 1, 0) # Transpose, because sitk uses different coordinate system than pynrrd.
            label = sitk.GetArrayFromImage(label).astype(np.uint8).transpose(2, 1, 0) # Transpose, because sitk uses different coordinate system than pynrrd.
            label[label > 0] = 1 # Ensure that all the labels are between 0 and 1.
            
            # Clip data to 0.5 and 99.5 percentiles.
            if not self.opt.no_clip == True:
                low, high = np.percentile(data, [0.5, 99.5])
                data = np.clip(data, low, high)

            # Convert numpy to torch format.
            data = torch.FloatTensor(data)
            
            # Real mask if exists and cut label.
            if self.mask_paths[current_idx] is not None:
                mask = sitk.ReadImage(self.mask_paths[current_idx])
                mask = sitk.GetArrayFromImage(mask).astype(np.uint8).transpose(2, 1, 0) # Transpose, because sitk uses different coordinate system than pynrrd.
                label = np.where(mask == 1, label, 0)
                mask = torch.ByteTensor(mask)
            else:
                mask = None

            label = torch.ByteTensor(label)

            # Store already loaded files in RAM.
            if self.max_nr_of_files_to_store > 0:
                self.loadedFiles[p_id] = [p_id, data, label, mask]
                self.max_nr_of_files_to_store -= 1

        else:
            p_id, data, label, mask = self.loadedFiles[p_id]

        if self.split_type != "test":
            # If mask exists, concatenate data and mask, and return as combined data with two channels.
            if self.split_type == "train":

                # Randomiser to select data with and without ROI.
                random_float_number = torch.rand(1)             # Get a random floating value between 0 and 1.
                if random_float_number < self.opt.p_foreground:
                    data, label = utils.select_roi_patches(data, label, mask, self.opt)
                else:
                    data, label = utils.select_random_patches(data, label, mask, self.opt)
            elif self.split_type == "val":
                data, label = utils.select_roi_patches(data, label, mask, self.opt)

            if self.opt.switch == True:
                # Normalization.
                data = utils.normalize(data, self.opt)

            # Data augmentation.
            if not self.opt.no_augmentation == True and self.split_type == "train":
                data, label = utils.data_augmentation_batch(data, label, self.opt)

            if self.opt.switch == False:
                # Normalization.
                data = utils.normalize(data, self.opt)

        return p_id, data, label

