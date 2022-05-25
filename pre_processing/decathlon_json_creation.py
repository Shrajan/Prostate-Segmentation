# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : decathlon_json_creation.py
    Code Author : Shrajan Bhandary
    Created on  : 14 January 2022
    Program Description:
        Convert the raw data to be into the format used by the Medical Segmentation Decathlon (MSD).

    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    14 January 2022    |     Shrajan Bhandary     |  Implemented necessary functions.   |
    |    21 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS

from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
import argparse, os

# IMPLEMENTATION

def convert_to_decathlon_format(src_folder=None, dst_folder=None):
    """
    Convert the raw data to be into the format used by the Medical Segmentation Decathlon (MSD).
     
    param src_folder          # Path to read the raw data.
    param dst_folder          # Path to store the reformated volumes.

    The raw data should have the following folder structure.

    src_folder/
    ├── test/
    │   └── 1/
    │      ├── data.nrrd
    │      ├── label.nrrd
    |      ...
    |
    ├── train/
    │   └── 1/
    │      ├── data.nrrd
    │      ├── label.nrrd
    |       ...
    
    """

    # Initialize an ordered dictionary to store contents in to the JSON file.
    json_dict = OrderedDict()
    json_dict['name'] = "FREIBURG"                          # Name of the dataset.
    json_dict['description'] = "prostate"                   # Description.
    json_dict['tensorImageSize'] = "4D"                     # Number of dimensions in the image.    
    json_dict['reference'] = "see challenge website"        # Additional info.
    json_dict['licence'] = "see challenge website"          # Additional info.
    json_dict['release'] = "0.0"                            # Additional info.
    json_dict['modality'] = {                               # Modalities of the input data.
        "0": "CT",
    }
    json_dict['labels'] = {                                 # Ground-truth values in the annotated data.
        "0": "background",
        "1": "prostate"
    }
    
    json_dict['training'] = []                              # To store training file names.
    json_dict['test'] = []                                  # To store testing file names.
    json_dict['numTest'] = 0                                # To store testing file count.
    json_dict['numTraining'] = 0                            # To store training file count.    

    maybe_mkdir_p(join(dst_folder, "imagesTr"))             # Create training data folder in the destination directory.
    maybe_mkdir_p(join(dst_folder, "labelsTr"))             # Create training label folder in the destination directory.

    # train
    current_dir = join(src_folder, "train")                 # Set current working directory to "src_folder/train/"
    patientFolders = os.listdir(current_dir)                # Get the names of folders in the source directory.
    patientFolders.sort()                                   # Sort the file names.

    # Remove the JSON file that contains the information about the dataset.
    if "dataset_info.json" in patientFolders:
        patientFolders.remove("dataset_info.json")

    json_dict['numTraining'] = len(patientFolders)          # Determine the total count of training files.
    
    # Go through each folder and get the required files, change the names and reformat them as needed.
    for index, fileNum in enumerate(patientFolders):
        index = str(index)
        data_out_fname = os.path.join(dst_folder, "imagesTr", "frei_" + index.zfill(3) + "_0000.nii.gz")
        sitk.WriteImage(sitk.ReadImage(os.path.join(current_dir, fileNum, "data.nrrd")), data_out_fname)

        label_out_fname = os.path.join(dst_folder, "labelsTr", "frei_" + index.zfill(3) + ".nii.gz")
        sitk.WriteImage(sitk.ReadImage(os.path.join(current_dir, fileNum, "label.nrrd")), label_out_fname)

        training_dict =  {'image': "./imagesTr/frei_" + index.zfill(3) + ".nii.gz", "label": "./labelsTr/frei_" + index.zfill(3) + ".nii.gz"}   
        json_dict['training'].append(training_dict)

    # If you want to save the test files. 
    if opt.no_test == False:
        # test  
        maybe_mkdir_p(join(dst_folder, "imagesTs"))         # Create testing data folder in the destination directory.
        maybe_mkdir_p(join(dst_folder, "labelsTs"))         # Create testing label folder in the destination directory.
        
        current_dir = os.path.join(src_folder, "test")      # Set current working directory to "src_folder/test/"
        patientFolders = os.listdir(current_dir)            # Get the names of folders in the source directory.
        patientFolders.sort()                               # Sort the file names.

        # Remove the JSON file that contains the information about the dataset.
        if "dataset_info.json" in patientFolders:
            patientFolders.remove("dataset_info.json")

        json_dict['numTest'] = len(patientFolders)          # Determine the total count of testing files. 

        # Go through each folder and get the required files, change the names and reformat them as needed.
        for index, fileNum in enumerate(patientFolders):
            index = str(index)
            data_out_fname = os.path.join(dst_folder, "imagesTs", "frei_" + index.zfill(3) + "_0000.nii.gz")
            sitk.WriteImage(sitk.ReadImage(os.path.join(current_dir, fileNum, "data.nrrd")), data_out_fname)
            
            label_out_fname = os.path.join(dst_folder, "labelsTs", "frei_" + index.zfill(3) + ".nii.gz")
            sitk.WriteImage(sitk.ReadImage(os.path.join(current_dir, fileNum, "label.nrrd")), label_out_fname)

            #test_dict =  {'image': "./imagesTs/frei_" + index.zfill(3) + ".nii.gz", "label": "./labelsTs/frei_" + index.zfill(3) + ".nii.gz"}   
            test_dict =  "./imagesTs/frei_" + index.zfill(3) + ".nii.gz"
            json_dict['test'].append(test_dict)
    
    # Save the final JSON file with the complete data description.
    save_json(json_dict, os.path.join(dst_folder, "dataset.json"))


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_folder", type=str, default="/home/shrajan/workspace/3D/Datasets/CT", required=False, help="Folder/directory to read the original prostate data. Make sure train or/and test volumes are in sub-folders.")
    parser.add_argument("--dst_folder", type=str, default="/home/shrajan/workspace/3D/DKFZ/nnUNet_raw_data_base/nnUNet_raw_data/Task534_Prostate/", required=False, help="Folder/directory to save the converted data.")
    parser.add_argument('--no_test', action='store_true', required=False, help='If set, test volumes will not be considered')
    opt = parser.parse_args()
    convert_to_decathlon_format(src_folder = opt.src_folder, dst_folder = opt.dst_folder)
    
    