"""
    Preprocess BRATS2018 Data
"""
import h5py
import SimpleITK as sitk
import nibabel as nib
import matplotlib.pyplot as plt
import warnings
import cv2
import numpy as np
import numpy.ma as ma
import torchvision.transforms
from alive_progress import alive_bar
import time
from torchvision.transforms import transforms
import os
from multiprocessing import Process

warnings.simplefilter('ignore')


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        #print("image:", image)
        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return image


def getLR(hr_data, scaling_factor):
    imgfft = np.fft.fftn(hr_data)
    imgfft = np.fft.fftshift(imgfft)

    x, y, z = imgfft.shape
    diff_x = x // (scaling_factor * 2)
    diff_y = y // (scaling_factor * 2)
    diff_z = z // (scaling_factor * 2)

    x_centre = x // 2
    y_centre = y // 2
    z_centre = z // 2

    mask = np.zeros(imgfft.shape)
    mask[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y,
    z_centre - diff_z: z_centre + diff_z] = 1
    # mask[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y, :] = 1
    imgfft = imgfft * mask

    #imgfft = imgfft[x_centre - diff_x: x_centre + diff_x, y_centre - diff_y: y_centre + diff_y, :]

    imgifft = np.fft.ifftshift(imgfft)
    imgifft = np.fft.ifftn(imgifft)
    img_out = abs(imgifft)
    return img_out


def N4_Bias_Field_Correction(img_obj, process=True):
    """
        (Optional): Perform BraTS MRI processing on the original .nii image object.
    """
    if not process:
        return img_obj
    else:
        maskImage = sitk.OtsuThreshold(img_obj, 0, 1, 200)
        image = sitk.Cast(img_obj, sitk.sitkFloat32)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        numberFilltingLevels = 4
        corrector.SetMaximumNumberOfIterations([4] * numberFilltingLevels)
        output = corrector.Execute(image, maskImage)
        return output


def normalize(data):
    data = np.clip(np.fabs(data), -np.inf, np.inf)
    data -= np.amin(data)
    data /= np.amax(data)
    return data


def process(path, modality, scaling_factor, transform):
    # image = np.array(load_nib(path + modality + '.nii'))
    image = np.array(load_nib(path))
    #print("shape:", image.shape)
    hr_img = transform(image)
    lr_img = getLR(hr_img, scaling_factor=scaling_factor)
    #hr_img = normalize(hr_img)
    #lr_img = normalize(lr_img)
    return hr_img, lr_img


def get_hr_lr(start, end, folder_names, path, data_shape, scaling_factor, transform):
    hr_arr, lr_arr = [], []

    with alive_bar(len(folder_names[start:end]), force_tty=True) as bar:
        for idx in range(start, end):
            sub_path = "{}{}".format('D:/HCP dataset/process/T1/', folder_names[idx])
            #print("sub_path:", sub_path)

            hr_img, lr_img = process(sub_path, modality, scaling_factor=scaling_factor, transform=transform)
            
            if hr_img.shape == data_shape:
                hr_arr.append(hr_img)
                lr_arr.append(lr_img)
            else:
                print('problem subpath', sub_path)
            bar()

    hr_arr, lr_arr = np.array(hr_arr), np.array(lr_arr)

    hr_arr = np.moveaxis(hr_arr, -1, 1)
    lr_arr = np.moveaxis(lr_arr, -1, 1)
    return hr_arr, lr_arr


def create_npy_dataset(path, modality, output_path, data_shape, scaling_factor=4, transform=None):
    print(scaling_factor)
    folder_names = os.listdir(path)
    #print("folder_name:", folder_names)
    print('.DS_Store' in folder_names)
    # hr_arr, lr_arr = np.empty([len(folder_names), *data_shape]), np.empty(
    #     [len(folder_names), *data_shape])

    training_hr, training_lr = get_hr_lr(0, 1, folder_names, path, data_shape, scaling_factor, transform)

    with open(output_path + f'/IXI_training_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
        np.save(f, training_hr)

    with open(output_path + f'/HCP_{modality}_scale_by_{scaling_factor}_imgs_small.npy', 'wb') as f:
        np.save(f, training_lr)

    # testing_hr, testing_lr = get_hr_lr(454, 504, folder_names, path, data_shape, scaling_factor, transform)

    # with open(output_path + f'/IXI_testing_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
    #     np.save(f, testing_hr)

    # with open(output_path + f'/IXI_testing_lr_{modality}_scale_by_{scaling_factor}_imgs_small.npy', 'wb') as f:
    #     np.save(f, testing_lr)

    # valid_hr, valid_lr = get_hr_lr(504, 510, folder_names, path, data_shape, scaling_factor, transform)

    # with open(output_path + f'/IXI_valid_hr_{modality}_scale_by_{scaling_factor}_imgs.npy', 'wb') as f:
    #     np.save(f, valid_hr)

    # with open(output_path + f'/IXI_valid_lr_{modality}_scale_by_{scaling_factor}_imgs_small.npy', 'wb') as f:
    #     np.save(f, valid_lr)

    print("=======Completed=======")


def load_nib(file_path):
    try:
        proxy = nib.load(file_path)
        data = proxy.get_fdata()
        proxy.uncache()
        return data
    except:
        print("Invalid file path is given")


if __name__ == "__main__":


    modality = 't2'

    data_shape = (224, 224, 96)
    transforms = torchvision.transforms.Compose([CenterCrop(data_shape)])
    create_npy_dataset(modality=modality,
                       path='D:/HCP dataset/T1',
                       output_path='D:/HCP dataset/output',
                       data_shape=data_shape,
                       scaling_factor=4,
                       transform=transforms)


    outpath = 'D:/HCP dataset/output'




