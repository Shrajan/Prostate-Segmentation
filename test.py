# DOCUMENT INFORMATION
'''
    Project Name: Prostate Segmentation
    File Name   : test.py
    Code Author : Dejan Kostyszyn and Shrajan Bhandary
    Created on  : 14 March 2021
    Program Description:
        This program predicts the segmentation for the test dataset. It can also generate 
        noisy data to check robustness.
            
    Versions:
    |----------------------------------------------------------------------------------------|
    |-----Last modified-----|----------Author----------|---------------Remarks---------------|
    |----------------------------------------------------------------------------------------|
    |    14 March 2021      |     Dejan Kostyszyn      |  Implemented necessary functions.   |
    |    20 April 2021      |     Shrajan Bhandary     |   Added postprocessing and noise.   |
    |    22 January 2022    |     Shrajan Bhandary     | Cleaned up stuff and added comments.|
    |    26 July 2022       |     Shrajan Bhandary     |      Removed unnecessary stuff.     |
    |----------------------------------------------------------------------------------------|
'''

# LIBRARY IMPORTS
import numpy as np
from pydantic import NoneIsAllowedError
import torch, os, time, csv, argparse, tqdm, utils, json
import SimpleITK as sitk
from torch.utils.data import DataLoader
from models import get_model
from natsort import natsorted
from collections import OrderedDict
import torchio as tio
torch.manual_seed(2021)
from pre_processing.resampler_utils import resample_3D_image
import dataloader as custom_DL 

# IMPLEMENTATION

def select_noise(opt="none"):

    # Create noise pipeline.
    if opt.noise == "none":
        noise = None
    elif opt.noise == "random":
        noise = tio.transforms.RandomNoise(std=opt.random_std)
    elif opt.noise == "motion":
        noise = tio.transforms.RandomMotion(num_transforms=opt.motion_transforms)
    elif opt.noise == "blur":
        noise = tio.transforms.RandomBlur(std=opt.blur_std)

    return noise

def get_train_options(opt=None, model_checkpoint=None):
    opt.normalize = model_checkpoint["normalize"]
    opt.patch_shape = model_checkpoint["patch_shape"]
    opt.n_kernels = model_checkpoint["n_kernels"]
    opt.no_clip = model_checkpoint["no_clip"]
    opt.seed = model_checkpoint["seed"]
    opt.model_name = model_checkpoint["model_name"]
    opt.voxel_spacing = model_checkpoint["voxel_spacing"]
    opt.input_channels = model_checkpoint["input_channels"]
    opt.output_channels = model_checkpoint["output_channels"]
    opt.no_shuffle = model_checkpoint["no_shuffle"]
    opt.dropout_rate = 0.0 # This is just a formality for model declaration.

    # Write options into file.
    with open(os.path.join(opt.results_path, "test_options.txt"), "w") as f:
        f.write("opt.normalize = " + str(opt.normalize) + "\n")
        f.write("opt.patch_shape = " + str(opt.patch_shape) + "\n")
        f.write("opt.n_kernels = " + str(opt.n_kernels) + "\n")
        f.write("opt.no_clip = " + str(opt.no_clip) + "\n")
        f.write("opt.voxel_spacing = " + str(opt.voxel_spacing) + "\n")
        f.write("opt.trained_model_path = " + str(opt.trained_model_path) + "\n")
        f.write("opt.divisor = " + str(opt.divisor) + "\n")
        f.write("opt.results_path = " + str(opt.results_path) + "\n")
        f.write("opt.data_root = " + str(opt.data_root) + "\n")
        f.write("opt.model_name = " + str(opt.model_name) + "\n")
        f.write("opt.fold = " + str(opt.fold) + "\n")
        f.write("opt.k_fold = " + str(opt.k_fold) + "\n")
        f.write("opt.seed = " + str(opt.seed) + "\n")

    return opt

def write_final_results(opt, test_metrics, overall_seg_time, overall_time):
    # Write final results into file.
    with open(os.path.join(opt.results_path, "test_results.csv"), 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Mean",
                    np.mean(test_metrics["DSC"]),
                    np.mean(test_metrics["HSD"]),
                    np.mean(test_metrics["ICC"]),
                    np.mean(test_metrics["ARI"]),
                    np.mean(test_metrics["ASSD"]),
                    np.mean(overall_seg_time),
                    np.mean(overall_time)])
        writer.writerow(["Median",
                    np.median(test_metrics["DSC"]),
                    np.median(test_metrics["HSD"]),
                    np.median(test_metrics["ICC"]),
                    np.median(test_metrics["ARI"]),
                    np.median(test_metrics["ASSD"]),
                    np.median(overall_seg_time),
                    np.median(overall_time)])
        writer.writerow(["Min",
                    np.min(test_metrics["DSC"]),
                    np.min(test_metrics["HSD"]),
                    np.min(test_metrics["ICC"]),
                    np.min(test_metrics["ARI"]),
                    np.min(test_metrics["ASSD"]),
                    np.min(overall_seg_time),
                    np.min(overall_time)])
        writer.writerow(["Max",
                    np.max(test_metrics["DSC"]),
                    np.max(test_metrics["HSD"]),
                    np.max(test_metrics["ICC"]),
                    np.max(test_metrics["ARI"]),
                    np.max(test_metrics["ASSD"]),
                    np.max(overall_seg_time),
                    np.max(overall_time)])
        writer.writerow(["Standard deviation",
                    np.std(test_metrics["DSC"]),
                    np.std(test_metrics["HSD"]),
                    np.std(test_metrics["ICC"]),
                    np.std(test_metrics["ARI"]),
                    np.std(test_metrics["ASSD"]),
                    np.std(overall_seg_time),
                    np.std(overall_time)])

def test(opt):
    
    # Create results files.
    if not os.path.exists(opt.results_path):
        os.makedirs(opt.results_path)

    with open(os.path.join(opt.results_path, "test_results_detailed.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Patient", "DSC", "HSD", "ICC", "ARI", "ASSD", "Comp. time (s)", "Comp. time total (s)"])
    
    with open(os.path.join(opt.results_path, "test_results.csv"), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["", "DSC", "HSD", "ICC", "ARI", "ASSD", "Comp. time (s)", "Comp. time total (s)"])

    # Load information about trained model from saved model.
    device_id = "cuda:" + str(opt.device_id)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")
    print("Using device : {}".format(device))

    checkpoint = torch.load(os.path.join(opt.trained_model_path), map_location=device)

    opt = get_train_options(opt=opt, model_checkpoint=checkpoint)

    dataset = custom_DL.Dataset(opt=opt, split_type="test")
    dataset_info = dataset.get_dataset_info()

    # Load trained model.
    print("Found model that was trained for {} epochs with highest DSC {}, {} normalization, patch shape {} and no_clip = {}. Loading... ".
            format(checkpoint["epoch"], checkpoint["mean_val_DSC"], checkpoint["normalize"], checkpoint["patch_shape"], opt.no_clip), end="")
    segM = get_model(opt)
    segM.load_state_dict(checkpoint["model_state_dict"])
    segM = segM.to(device)
    segM.eval()

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    noise = select_noise(opt=opt)

    patch_shape = opt.patch_shape
    test_metrics = {"DSC":[], "HSD":[], "ICC":[], "ARI":[], "ASSD":[] }
    overall_seg_time = []
    overall_time = []

    predictions_path = os.path.join(opt.results_path,'predictions')
    if not os.path.exists(predictions_path):
        os.makedirs(predictions_path)

    postprocessed_path = os.path.join(opt.results_path,"predictions_postprocessed")
    if not os.path.exists(postprocessed_path):
        os.makedirs(postprocessed_path)

    with torch.no_grad():
        for p_idx, Dict in enumerate(dataloader):
            start_time = time.time()
            p_id, data, label = Dict["p_id"][0], Dict["data"], Dict["label"]
            
            print("\nPredicting prostate for patient: ", p_id)
            
            if noise != None:
                data = noise(data) 

            # Create results folder.
            res_path = os.path.join(predictions_path, p_id)
            if not os.path.exists(res_path):
                os.makedirs(res_path)

            start_seg = time.time()
            result_array = torch.zeros_like(data, dtype=float).squeeze(0).squeeze(0)
            counter = torch.zeros_like(result_array)
            divisor = opt.divisor
            
            for x in tqdm.tqdm(range(0, data.shape[-3] - (patch_shape[0] - patch_shape[0] // divisor),
                           patch_shape[0] // divisor), desc="Sliding window inference"):
                for y in range(0, data.shape[-2] - (patch_shape[1] - patch_shape[1] // divisor),
                               patch_shape[1] // divisor):
                    for z in range(0, data.shape[-1] - (patch_shape[2] - patch_shape[2] // divisor),
                                   patch_shape[2] // divisor):

                        x = min(x, data.shape[-3] - patch_shape[0])
                        y = min(y, data.shape[-2] - patch_shape[1])
                        z = min(z, data.shape[-1] - patch_shape[2])

                        # Crop data.
                        crop = data[..., x:x + patch_shape[0], y:y + patch_shape[1], z:z + patch_shape[2]]
                        
                        # Normalization.
                        crop = utils.normalize(crop, opt)

                        # Prediction.
                        outM_array = (segM(crop.unsqueeze(0).to(device)).squeeze(0).squeeze(0)).cpu()
                        outM_array = torch.sigmoid(outM_array)
                        outM_array = torch.where(outM_array < 0.5, torch.tensor(0), torch.tensor(1))
                        result_array[x:x + patch_shape[0], y:y + patch_shape[1], z:z + patch_shape[2]] += outM_array
                        counter[x:x + patch_shape[0], y:y + patch_shape[1], z:z + patch_shape[2]] += 1

    
            seg_time = time.time() - start_seg
            overall_seg_time.append(seg_time)
            
            # Take a threshold to make it binary.
            result_no_threshold = result_array.div(counter)
            result_array = torch.where(result_no_threshold >= torch.tensor(0.5), torch.tensor(1), torch.tensor(0))

            # Find connected components and only keep largest.
            try:
                result_array = utils.keep_only_largest_connected_component(result_array)

                # Find centroid of segmentation.
                centroid = utils.get_roi_centroid(result_array)

                # Select patch with centroid as centroid of patch.
                x = min(max(0, centroid[0] - patch_shape[0]//2), data.shape[-3] - patch_shape[0])
                y = min(max(0, centroid[1] - patch_shape[1]//2), data.shape[-2] - patch_shape[1])
                z = min(max(0, centroid[2] - patch_shape[2]//2), data.shape[-1] - patch_shape[2])

                # Create crop with zeros in case that dimension of input image is smaller than path shape.
                crop = data[..., x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]].unsqueeze(0).to(device)
                crop = utils.normalize(crop, opt)

                # Feed only important patch into generator.
                outM_array = segM(crop).squeeze(0).squeeze(0).cpu().detach()
                outM_array = torch.sigmoid(outM_array).numpy()
                outM_array_thres = np.where(outM_array < 0.5, 0, 1)

                result_array = torch.zeros_like(data.squeeze(0), dtype=torch.int8).numpy()
                result_array[x:x+patch_shape[0], y:y+patch_shape[1], z:z+patch_shape[2]] = outM_array_thres

            except:
                # If nothing was contoured during sliding window.
                result_array = torch.zeros_like(data.squeeze(0), dtype=torch.int8).numpy()

            # Save the predicted image.
            result_array = result_array.transpose(2,1,0)
            predicted_image = sitk.GetImageFromArray(result_array)

            # Set other image characteristics
            predicted_image.SetOrigin(dataset_info["files"][p_id]["new_volume_info"]["origin"])
            predicted_image.SetSpacing(dataset_info["files"][p_id]["new_volume_info"]["spacing"])
            predicted_image.SetDirection(tuple(dataset_info["files"][p_id]["new_volume_info"]["direction"]))

            # Write the image to the disk.
            sitk.WriteImage(predicted_image, res_path + "/prediction.seg.nrrd")

            print("Postprocessing... ", end="")
            orig_label_path = dataset_info["files"][p_id]["orig_volume_info"]["labelFilePath"]
            orig_label_image = sitk.ReadImage(os.path.join(opt.data_root, orig_label_path))

            resampled_result_image = resample_3D_image(sitkImage = predicted_image,
                                                newSpacing = orig_label_image.GetSpacing(), 
                                                interpolation = "nearest", 
                                                newDirection = orig_label_image.GetDirection(), 
                                                change_spacing = True, change_direction = True,
                                                newOrigin = orig_label_image.GetOrigin(),
                                                postProcess=True)

            result_orig_shape = sitk.GetArrayFromImage(resampled_result_image).astype(np.uint8)
            orig_label = sitk.GetArrayFromImage(orig_label_image).astype(np.uint8)

            print("done.", end="\n")

            # Store result.
            print("Writing to permanent memory... ", end="")
            start_write = time.time()

            sitk.WriteImage(sitk.Cast(resampled_result_image, sitk.sitkUInt8), 
                            os.path.join(postprocessed_path, os.path.split(orig_label_path)[-1]), True)

            print("Finished in {}s.".format(round(time.time() - start_write, 2)))
            total_time = time.time() - start_time
            overall_time.append(total_time)

            # Compute validation metrics.
            print("Computing metrics...")
            test_metrics = utils.compute_metrics(input=torch.ByteTensor(np.expand_dims(result_orig_shape, axis=(0,1))), 
                                                target=torch.ByteTensor(np.expand_dims(orig_label, axis=(0,1))), 
                                                metrics=test_metrics, opt=opt)
          
            # Write losses into file.
            print("DSC: {}, HSD: {}, ICC: {}, ARI: {}, ASSD: {} \n".
                  format(test_metrics["DSC"][-1], test_metrics["HSD"][-1], test_metrics["ICC"][-1], test_metrics["ARI"][-1], test_metrics["ASSD"][-1]
                  ))
            # Write final results into file.
            with open(os.path.join(opt.results_path, "test_results_detailed.csv"), 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([p_id, test_metrics["DSC"][-1], test_metrics["HSD"][-1], test_metrics["ICC"][-1], test_metrics["ARI"][-1], test_metrics["ASSD"][-1], seg_time, total_time])
    
    write_final_results(opt=opt, test_metrics=test_metrics, overall_seg_time=overall_seg_time, overall_time=overall_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisor", type=int, default=2, required=False, help="Patch overlap for sliding window. Overlap is 1/divisor. More = slower computation time, less = worse performance... More or less...")
    parser.add_argument("--trained_model_path", type=str, default="best_net.sausage", required=True, help="Path to trained model.")
    parser.add_argument("--results_path", type=str, default="results/test/", required=False, help="Path to store the results.")
    parser.add_argument("--data_root", type=str, default="", required=True, help="Path to data.")
    parser.add_argument('--device_id', type=int, default=0, required=False, help='Use the different GPU(device) numbers available. Example: If two Cuda devices are available, options: 0 or 1')
    parser.add_argument("--model_name", type=str, default="unet", required=False, help="Select name of trained model. If available, model name will be read from model state dict, else from this option.")
    parser.add_argument('--input_channels', type=int, default=1, required=False, help='Number of channels in the input data. If available, it will be read from model state dict, else from this option.')
    parser.add_argument('--output_channels', type=int, default=1, required=False, help='Number of channels in the output label. If available, it will be read from model state dict, else from this option.')
    parser.add_argument('--fold', type=int, default=-1, required=False, choices = [0,1,2,3,4,5,6,7], help='Select the fold index for 5-fold crosss validation. If fold == -1 all data in the folder will be used for training and validation.')
    parser.add_argument('--no_shuffle', action='store_true', required=False, help='If set, training and validation data will not be shuffled')
    parser.add_argument('--k_fold', type=int, default=0, required=False, choices = [0,5,8], help='Choose between no crosss validation (ensure separate folders for train and test), or 5-fold crosss validation or 8-fold crosss validation. ')
    parser.add_argument("--noise", type=str, default="none", choices=["none", "random", "motion", "blur"], required=False, help="Select noise to add.")
    parser.add_argument("--blur_std", type=float, default=2, required=False, help="The amount of standard deviation to be applied to create blur noise.")
    parser.add_argument("--random_std", type=float, default=45, required=False, help="The amount of standard deviation to be applied to create random noise.")
    parser.add_argument("--motion_transforms", type=int, default=1, required=False, help="The number of transforms to be applied to create motion noise.")
    opt = parser.parse_args()

    test(opt=opt)

