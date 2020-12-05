#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
import torch
import torch.utils.data as utils
import numpy as np
import pickle
# from retina.retina import warp_image
from collections import namedtuple, Iterable
import os
from pathlib import Path
from neuralpredictors.data.samplers import RepeatsBatchSampler
from .utility import get_validation_split, get_cached_loader, get_fraction_of_training_images
    #get_crop_from_stimulus_location  # ImageCache
from nnfabrik.utility.nn_helpers import get_module_output, set_random_seed, get_dims_for_loader_dict
from nnfabrik.utility.dj_helpers import make_hash
from skimage.transform import rescale
import cv2
from scipy import ndimage

def monkey_saliency_loader(dataset,
                           neuronal_data_files,
                           image_cache_path,
                           saliency_cache_path,
                           batch_size=64,
                           seed=None,
                           train_frac=0.8,
                           subsample=1,
                           crop=((96, 96), (96, 96)),
                           scale=1.,
                           time_bins_sum=tuple(range(12)),
                           avg=False,
                           image_file=None,
                           return_data_info=False,
                           store_data_info=True,
                           image_frac=1.,
                           image_selection_seed=None,
                           randomize_image_selection=True,
                           logarithm = True,
                           gradient = False,
                           include_all = False):
    """
    Function that returns cached dataloaders for monkey ephys experiments.

     creates a nested dictionary of dataloaders in the format
            {'train' : dict_of_loaders,
             'validation'   : dict_of_loaders,
            'test'  : dict_of_loaders, }

        in each dict_of_loaders, there will be  one dataloader per data-key (refers to a unique session ID)
        with the format:
            {'data-key1': torch.utils.data.DataLoader,
             'data-key2': torch.utils.data.DataLoader, ... }

    requires the types of input files:
        - the neuronal data files. A list of pickle files, with one file per session
        - the image file. The pickle file that contains all images.
        - individual image files, stored as numpy array, in a subfolder

    Args:
        dataset: a string, identifying the Dataset:
            'PlosCB19_V1', 'CSRF19_V1', 'CSRF19_V4'
            This string will be parsed by a datajoint table

        neuronal_data_files: a list paths that point to neuronal data pickle files
        image_file: a path that points to the image file
        image_cache_path: The path to the cached images
        batch_size: int - batch size of the dataloaders
        seed: int - random seed, to calculate the random split
        train_frac: ratio of train/validation images
        subsample: int - downsampling factor
        crop: int or tuple - crops x pixels from each side. Example: Input image of 100x100, crop=10 => Resulting img = 80x80.
            if crop is tuple, the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_top, crop_bottom), (crop_left, crop_right)]
        scale: float or integer - up-scale or down-scale via interpolation hte input images (default= 1)
        time_bins_sum: sums the responses over x time bins.
        avg: Boolean - Sums oder Averages the responses across bins.

    Returns: nested dictionary of dataloaders
    """

    dataset_config = locals()

    # initialize dataloaders as empty dict
    dataloaders = {'train': {}, 'validation': {}, 'test': {}}

    if not isinstance(time_bins_sum, Iterable):
        time_bins_sum = tuple(range(time_bins_sum))

    if isinstance(crop, int):
        crop = [(crop, crop), (crop, crop)]

    if not isinstance(image_frac, Iterable):
        image_frac = [image_frac for i in neuronal_data_files]

    # clean up image path because of legacy folder structure
    image_cache_path = image_cache_path.split('individual')[0]

    # Load image statistics if present
    stats_filename = make_hash(dataset_config)

    stats_path = os.path.join(image_cache_path, 'statistics/', stats_filename)
    stats_path2 = os.path.join(saliency_cache_path, 'statistics/', stats_filename)

    # Get mean and std
    if os.path.exists(stats_path):
        with open(stats_path, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        img_mean = list(data_info.values())[0]["img_mean"]
        img_std = list(data_info.values())[0]["img_std"]

    if os.path.exists(stats_path2):
        with open(stats_path2, "rb") as pkl:
            data_info = pickle.load(pkl)
        if return_data_info:
            return data_info
        maps_mean = list(data_info.values())[0]["maps_mean"]
        maps_std = list(data_info.values())[0]["maps_std"]


        # Initialize cache
        cache = ImageCache(path=image_cache_path, sal_path=saliency_cache_path, subsample=subsample, crop=crop,
                           scale=scale, img_mean=img_mean, img_std=img_std, transform=True, normalize=True, logarithm=logarithm, gradient=gradient, include_all=include_all)

    else:  # if stats not given
        # Initialize cache with no normalization
        cache = ImageCache(path=image_cache_path, sal_path=saliency_cache_path, subsample=subsample, crop=crop,
                           scale=scale, transform=True, normalize=False, logarithm=logarithm, gradient=gradient, include_all= include_all)

        # Compute mean and std of transformed images and zscore data (the cache wil be filled so first epoch will be fast)

        cache.zscore_images(update_stats=True)
        img_mean = cache.img_mean
        img_std = cache.img_std



    n_images = len(cache)
    data_info = {}

    # set up parameters for the different dataset types
    if dataset == 'PlosCB19_V1':
        # for the "Amadeus V1" Dataset, recorded by Santiago Cadena, there was no specified test set.
        # instead, the last 20% of the dataset were classified as test set. To make sure that the test set
        # of this dataset will always stay identical, the `train_test_split` value is hardcoded here.
        train_test_split = 0.8
        image_id_offset = 1
    else:
        train_test_split = 1
        image_id_offset = 0

    all_train_ids, all_validation_ids = get_validation_split(n_images=n_images * train_test_split,
                                                             train_frac=train_frac,
                                                             seed=seed)

    # cycling through all datafiles to fill the dataloaders with an entry per session
    for i, datapath in enumerate(neuronal_data_files):

        with open(datapath, "rb") as pkl:
            raw_data = pickle.load(pkl)

        subject_ids = raw_data["subject_id"]
        data_key = str(raw_data["session_id"])
        responses_train = raw_data["training_responses"].astype(np.float32)
        responses_test = raw_data["testing_responses"].astype(np.float32)
        training_image_ids = raw_data["training_image_ids"] - image_id_offset
        testing_image_ids = raw_data["testing_image_ids"] - image_id_offset

        if dataset != 'PlosCB19_V1':
            if len(responses_test.shape) != 3:
                responses_test = responses_test[None, ...]
                responses_train = responses_train[None, ...]
                # correct the shape of the responses for a session that was exported incorrectly
                if data_key != '3653663964522':
                    warnings.warn("Pickle file with invalid response shape detected")

            responses_test = responses_test.transpose((2, 0, 1))
            responses_train = responses_train.transpose((2, 0, 1))

            if time_bins_sum is not None:  # then average over given time bins
                responses_train = (np.mean if avg else np.sum)(responses_train[:, :, time_bins_sum], axis=-1)
                responses_test = (np.mean if avg else np.sum)(responses_test[:, :, time_bins_sum], axis=-1)

        if image_frac[i] < 1:
            if randomize_image_selection:
                image_selection_seed = int(image_selection_seed * image_frac[i])
            idx_out = get_fraction_of_training_images(image_ids=training_image_ids, fraction=image_frac[i],
                                                      seed=image_selection_seed)
            training_image_ids = training_image_ids[idx_out]

            responses_train = responses_train[idx_out]

        train_idx = np.isin(training_image_ids, all_train_ids)

        val_idx = np.isin(training_image_ids, all_validation_ids)

        responses_val = responses_train[val_idx]
        responses_train = responses_train[train_idx]

        validation_image_ids = training_image_ids[val_idx]
        training_image_ids = training_image_ids[train_idx]

        train_loader = get_cached_loader(training_image_ids, responses_train, batch_size=batch_size, image_cache=cache)
        val_loader = get_cached_loader(validation_image_ids, responses_val, batch_size=batch_size, image_cache=cache)
        test_loader = get_cached_loader(testing_image_ids,
                                        responses_test,
                                        batch_size=None,
                                        shuffle=None,
                                        image_cache=cache,
                                        repeat_condition=testing_image_ids)

        dataloaders["train"][data_key] = train_loader
        dataloaders["validation"][data_key] = val_loader
        dataloaders["test"][data_key] = test_loader

    if store_data_info and not os.path.exists(stats_path):

        in_name, out_name = next(iter(list(dataloaders["train"].values())[0]))._fields

        session_shape_dict = get_dims_for_loader_dict(dataloaders["train"])
        n_neurons_dict = {k: v[out_name][1] for k, v in session_shape_dict.items()}
        in_shapes_dict = {k: v[in_name] for k, v in session_shape_dict.items()}
        input_channels = {k: v[in_name][1] for k, v in session_shape_dict.items()}

        for data_key in session_shape_dict:
            data_info[data_key] = dict(input_dimensions=in_shapes_dict[data_key],
                                       input_channels=input_channels[data_key],
                                       output_dimension=n_neurons_dict[data_key],
                                       img_mean=img_mean,
                                       img_std=img_std)

        stats_path_base = str(Path(stats_path).parent)
        if not os.path.exists(stats_path_base):
            os.mkdir(stats_path_base)
        with open(stats_path, "wb") as pkl:
            pickle.dump(data_info, pkl)

    return dataloaders if not return_data_info else data_info


##Image Cache for Saliency
class ImageCache:
    """
    A simple cache which loads images into memory given a path to the directory where the images are stored.
    Images need to be present as 2D .npy arrays
    """

    def __init__(self, path=None, sal_path=None, subsample=1, crop=0, scale=1, img_mean=None, img_std=None,maps_mean = None, maps_std = None,
                 transform=True, normalize=True, filename_precision=6, logarithm = True, gradient = False, include_all = False):

        """
        path: str - pointing to the directory, where the individual .npy files are present
        subsample: int - amount of downsampling
        crop:  the expected input is a list of tuples, the specify the exact cropping from all four sides
                i.e. [(crop_left, crop_right), (crop_top, crop_down)]
        scale: - the scale factor to upsample or downsample images via interpolation
        img_mean: - mean luminance across all images
        img_std: - std of the luminance across all images
        transform: - whether to apply a transformation to an image
        normalize: - whether to standardized inputs by the mean and variance
        filename_precision: - amount leading zeros of the files in the specified folder
        """

        self.cache = {}
        self.path = path
        self.sal_path = sal_path
        self.subsample = subsample
        self.crop = crop
        self.scale = scale
        self.img_mean = img_mean
        self.img_std = img_std

        self.maps_mean = maps_mean
        self.maps_std = maps_std

        self.transform = transform
        self.normalize = normalize
        self.leading_zeros = filename_precision
        self.logarithm = logarithm
        self.gradient = gradient
        self.include_all = include_all

    def __len__(self):
        return len([file for file in os.listdir(self.path) if file.endswith('.npy')])

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, item):
        item = item.tolist() if isinstance(item, Iterable) else item
        return [self[i] for i in item] if isinstance(item, Iterable) else self.update(item)

    def update(self, key):
        if key in self.cache:
            return self.cache[key]
        else:
            filename = os.path.join(self.path, str(key).zfill(self.leading_zeros) + '.npy')
            image = np.load(filename)

            image = self.transform_image(image) if self.transform else image
            image = self.normalize_image(image) if self.normalize else image
            image = torch.tensor(image).to(torch.float)

            filename_sal = os.path.join(self.sal_path, str(key).zfill(self.leading_zeros) + '.npy')
            sal_map = np.load(filename_sal)

            if (self.logarithm==False):
                sal_map = np.exp(sal_map)

            sal_map = self.transform_image(sal_map) if self.transform else sal_map

            sal_map = self.normalize_maps(sal_map) if self.normalize else sal_map

            if (self.gradient==True):
                sal_map = sal_map.reshape((sal_map.shape[1], sal_map.shape[2]))

                sx = ndimage.sobel(sal_map, axis=0, mode='constant')
                # Get y-gradient in "sy"
                sy = ndimage.sobel(sal_map, axis=1, mode='constant')

                sx = sx.reshape((1, sx.shape[0], sx.shape[1]))
                sy = sy.reshape((1, sy.shape[0], sy.shape[1]))

                sobelx = torch.tensor(sx).to(torch.float)
                sobely = torch.tensor(sy).to(torch.float)

                if (self.include_all == True):
                    sal_map = torch.tensor(sal_map).to(torch.float)
                    image_concat = torch.cat((image, sal_map, sobelx, sobely), 0)
                else:
                    image_concat = torch.cat((image, sobelx, sobely), 0)
                image = image_concat

                self.cache[key] = image

                return image
            else:
                sal_map = torch.tensor(sal_map).to(torch.float)

                image_concat = torch.cat((image, sal_map, sal_map), 0)
                image = image_concat

                self.cache[key] = image

                return image


    def transform_image(self, image):
        """
        applies transformations to the image: downsampling, cropping, rescaling, and dimension expansion.
        """
        if len(image.shape) == 2:
            h, w = image.shape
            rescale_fn = lambda x, s: rescale(x,
                                              s,
                                              mode='reflect',
                                              multichannel=False,
                                              anti_aliasing=False,
                                              preserve_range=True).astype(x.dtype)
            image = image[self.crop[0][0]:h - self.crop[0][1]:self.subsample,
                    self.crop[1][0]:w - self.crop[1][1]:self.subsample]
            image = image if self.scale == 1 else rescale_fn(image, self.scale)
            image = image[None,]
            return image

        elif len(image.shape) == 3:
            h, w = image.shape[:2]
            rescale_fn = lambda x, s: rescale(x, s, mode='reflect', multichannel=True, anti_aliasing=False,
                                              preserve_range=True).astype(x.dtype)
            image = image[self.crop[0][0]:h - self.crop[0][1]:self.subsample,
                    self.crop[1][0]:w - self.crop[1][1]:self.subsample, ...]
            image = image if self.scale == 1 else rescale_fn(image, self.scale)
            image = image[None,].permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Image shape has to be two dimensional (grayscale) or three dimensional "
                             f"(color, with w x h x c). got image shape {image.shape}")
        return image

    def normalize_image(self, image):
        """
        standarizes image
        """
        image = (image - self.img_mean) / self.img_std
        return image

    def normalize_maps(self, image):
        """
        standarizes maps
        """
        image = (image - self.maps_mean) / self.maps_std
        return image


    @property
    def cache_size(self):
        return len(self.cache)

    @property
    def loaded_images(self):
        print('Loading images ...')
        items = [int(file.split('.')[0]) for file in os.listdir(self.path) if file.endswith('.npy')]
        images = torch.stack([self.update(item) for item in items])
        return images

    @property
    def loaded_maps(self):
        print('Loading images ...')
        items = [int(file.split('.')[0]) for file in os.listdir(self.sal_path) if file.endswith('.npy')]
        images = torch.stack([self.update(item) for item in items])
        return images


    def zscore_images(self, update_stats=True):
        """
        zscore images in cache
        """
        images = self.loaded_images

        if (images.shape[1] == 3):
            img_mean = images[:,0,:,:].mean()
            img_std = images[:,0,:,:].std()

            maps_mean = images[:,1,:,:].mean()
            maps_std = images[:,1,:,:].std()


            grad_mean = images[:,2,:,:].mean()
            grad_std = images[:,2,:,:].std()


            for key in self.cache:
                self.cache[key][0, :, :] = (self.cache[key][0, :, :] - img_mean) / img_std
                self.cache[key][1, :, :] = (self.cache[key][1, :, :] - maps_mean) / maps_std
                self.cache[key][2, :, :] = (self.cache[key][2, :, :] - grad_mean) / grad_std

        if (images.shape[1] == 4):

            img_mean = images[:, 0, :, :].mean()
            img_std = images[:, 0, :, :].std()

            maps_mean = images[:, 1, :, :].mean()
            maps_std = images[:, 1, :, :].std()

            grad_mean = images[:, 2, :, :].mean()
            grad_std = images[:, 2, :, :].std()

            grad2_mean = images[:, 2, :, :].mean()
            grad2_std = images[:, 2, :, :].std()

            for key in self.cache:
                self.cache[key][0, :, :] = (self.cache[key][0, :, :] - img_mean) / img_std
                self.cache[key][1, :, :] = (self.cache[key][1, :, :] - maps_mean) / maps_std
                self.cache[key][2, :, :] = (self.cache[key][2, :, :] - grad_mean) / grad_std
                self.cache[key][2, :, :] = (self.cache[key][2, :, :] - grad2_mean) / grad2_std

        if update_stats:
            self.img_mean = np.float32(img_mean.item())
            self.img_std = np.float32(img_std.item())

            self.maps_mean = np.float32(maps_mean.item())
            self.maps_std = np.float32(maps_std.item())

            self.grad_mean = np.float32(grad_mean.item())
            self.grad_std = np.float32(grad_std.item())

            if (images.shape[1] == 4):
                self.grad2_mean = np.float32(grad2_mean.item())
                self.grad2_std = np.float32(grad2_std.item())















