from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import numpy as np
import os
from scipy.ndimage import zoom
import nibabel as nib
from alive_progress import alive_bar



def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        if fname.endswith('.nii.gz') or fname.endswith('.nii'):
            path = os.path.join(dir, fname)
            images.append(path)
    return images


def clip_image(im):
    clip_value = np.sort(im.ravel())[int(np.prod(im.shape) * 0.999)]
    im = np.clip(im, 0, clip_value)
    return im


def get_lr(im, scale):
    downscaled_lr = zoom(im, 1.0 / scale, order=2, prefilter=False)
    lr = np.clip(zoom(downscaled_lr, scale, order=2, prefilter=False), 0, im.max())
    lr[np.where(im == 0)] = 0
    return lr

# 주어진 배열에서 유효한 데이터가 존재하는 영역만을 잘라내서 반환
def crop_slice(array, padding, factor):
    for i in range(padding, array.shape[0] - padding):
        if not np.all(array[i, :, :] == 0):
            x_use1 = i - padding
            x_use1 = x_use1 - (x_use1 % factor)
            break
    for i in reversed(range(padding, array.shape[0] - padding)):
        if not np.all(array[i, :, :] == 0):
            x_use2 = i + padding
            break
    for i in range(padding, array.shape[1] - padding):
        if not np.all(array[:, i, :] == 0):
            y_use1 = i - padding
            y_use1 = y_use1 - (y_use1 % factor)
            break
    for i in reversed(range(padding, array.shape[1] - padding)):
        if not np.all(array[:, i, :] == 0):
            y_use2 = i + padding
            break
    for i in range(padding, array.shape[2] - padding):
        if not np.all(array[:, :, i] == 0):
            z_use1 = i - padding
            z_use1 = z_use1 - (z_use1 % factor)
            break
    for i in reversed(range(padding, array.shape[2] - padding)):
        if not np.all(array[:, :, i] == 0):
            z_use2 = i + padding
            break

    area = (slice(x_use1, x_use2), slice(y_use1, y_use2), slice(z_use1, z_use2))
    return area


# 주어진 이미지를 특정 크기의 배수로 크기 조정
# 이미지 크기를 표준화하기 위해
def mod_crop(im, modulo):
    H, W, D = im.shape
    size0 = H - H % modulo
    size1 = W - W % modulo
    size2 = D - D % modulo

    out = im[0:size0, 0:size1, 0:size2]

    return out


def get_data(data_dir, scale):
    images = make_dataset(data_dir)
    image_list_hr = []
    image_list_lr = []
    for file_idx, file_name in enumerate(images):
        print('\r{} / {}'.format(file_idx + 1, len(images)), end='')
        raw_image = nib.load(file_name)
        raw_array = np.array(raw_image.get_fdata(), dtype=np.float32)
        raw_header = raw_image.header.copy()

        pad_area = ((5,5),(5,5),(5,5))
        pad_array = np.pad(raw_array, pad_area, "constant", constant_values=0)

        clipped_image = clip_image(pad_array)
        im = mod_crop(clipped_image, scale)
        slice_area = crop_slice(im, 5, scale)

        im_blank_LR = get_lr(im, scale)
        im_LR = im_blank_LR[slice_area] / im.max()
        im_HR = im[slice_area] / im.max()

        image_list_hr.append(im_HR)
        image_list_lr.append(im_LR)

    print()
    return image_list_hr, image_list_lr

def resize_images(image_list, target_shape=(256, 256, 96)):
    resized_images = []
    for img in image_list:
        if img.shape != target_shape:
            resized_img = np.resize(img, target_shape)  # np.resize를 사용하여 이미지 크기 조정
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    return resized_images


# def train_data_generator_3d(image_list_hr, image_list_lr, batch_size):
#     i = 0
#     while True:
#         batch_hr = []
#         batch_lr = []
#         while len(batch_lr) != batch_size:
#             xa = np.random.randint(image_list_hr[i].shape[0] - 60)
#             ya = np.random.randint(image_list_hr[i].shape[1] - 60)
#             za = np.random.randint(image_list_hr[i].shape[2] - 60)

#             if image_list_hr[i][xa+30, ya+30, za+30] != 0:
#                 # batch_hr.append(image_list_hr[i][xa+6:xa+54, ya+6:ya+54, za+6:za+54, np.newaxis])
#                 batch_hr.append(image_list_hr[i][xa:xa+60, ya:ya+60, za:za+60, np.newaxis])
#                 batch_lr.append(image_list_lr[i][xa:xa+60, ya:ya+60, za:za+60, np.newaxis])
#                 i = (i + 1) % len(image_list_hr)

#         yield np.array(batch_lr), np.array(batch_hr)
def get_hr_lr(data_dirs):
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs] 

    hr_arr, lr_arr = [], []
    with alive_bar(len(data_dirs)) as bar:
        for data_dir in data_dirs:
            hr_img, lr_img = get_data(data_dir, 2)  # Assuming 'scale' is constant at 2
            hr_img = resize_images(hr_img, (256, 256, 96))
            lr_img = resize_images(lr_img, (256, 256, 96))
            hr_arr.extend(hr_img)
            lr_arr.extend(lr_img)
            bar()
    hr_arr, lr_arr = np.array(hr_arr), np.array(lr_arr)
    hr_arr = np.moveaxis(hr_arr, -1, 1)
    lr_arr = np.moveaxis(lr_arr, -1, 1)
    return hr_arr, lr_arr