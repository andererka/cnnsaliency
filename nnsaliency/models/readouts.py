import torch

from torch import nn
from nnfabrik.utility.nn_helpers import get_io_dims, get_module_output, set_random_seed, get_dims_for_loader_dict
from collections import OrderedDict, Iterable
import numpy as np
import warnings
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import ModuleDict
from neuralpredictors.constraints import positive
from neuralpredictors.layers.cores import DepthSeparableConv2d, Core2d, Stacked2dCore
from neuralpredictors import regularizers
from neuralpredictors.layers.readouts import PointPooled2d, FullGaussian2d, SpatialXFeatureLinear, RemappedGaussian2d, AttentionReadout


from neuralpredictors.layers.legacy import Gaussian2d


class MultiplePointPooled2d(torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, pool_steps, pool_kern, bias, init_range, gamma_readout):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, PointPooled2d(
                in_shape,
                n_neurons,
                pool_steps=pool_steps,
                pool_kern=pool_kern,
                bias=bias,
                init_range=init_range)
                            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleGaussian2d(torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma_range, bias, gamma_readout):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, Gaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma_range=init_sigma_range,
                bias=bias)
                            )
        self.gamma_readout = gamma_readout

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultiReadout:

    def forward(self, *args, data_key=None, **kwargs):
        if data_key is None and len(self) == 1:
            data_key = list(self.keys())[0]
        return self[data_key](*args, **kwargs)

    def regularizer(self, data_key):
        return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleSpatialXFeatureLinear(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_noise, bias, normalize, gamma_readout, constrain_pos=False):
        # super init to get the _module attribute
        super().__init__()
        for k in n_neurons_dict:
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]
            self.add_module(k, SpatialXFeatureLinear(
                in_shape=in_shape,
                outdims=n_neurons,
                init_noise=init_noise,
                bias=bias,
                normalize=normalize,
                constrain_pos=constrain_pos
            )
                            )
        self.gamma_readout = gamma_readout

    def regularizer(self, data_key):
        return self[data_key].l1(average=False) * self.gamma_readout


class MultipleFullGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, shared_match_ids, gamma_grid_dispersion=0):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))
            elif share_grid:
                shared_grid = {
                    'match_ids': shared_match_ids[k],
                    'shared_grid': None if i == 0 else self[k0].shared_grid
                }

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, FullGaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                init_mu_range=init_mu_range,
                init_sigma=init_sigma,
                bias=bias,
                gauss_type=gauss_type,
                grid_mean_predictor=grid_mean_predictor,
                shared_features=shared_features,
                shared_grid=shared_grid,
                source_grid=source_grid
            )
                            )
        self.gamma_readout = gamma_readout
        self.gamma_grid_dispersion = gamma_grid_dispersion

    def regularizer(self, data_key):
        if hasattr(FullGaussian2d, 'mu_dispersion'):
            return self[data_key].feature_l1(average=False) * self.gamma_readout \
                   + self[data_key].mu_dispersion() * self.gamma_grid_dispersion
        else:
            return self[data_key].feature_l1(average=False) * self.gamma_readout


class MultipleRemappedGaussian2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shape_dict, n_neurons_dict, remap_layers, remap_kernel, max_remap_amplitude,
                 init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, shared_match_ids, ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            source_grid = None
            shared_grid = None
            shared_transform = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))

            elif share_grid:
                shared_grid = {
                    'match_ids': shared_match_ids[k],
                    'shared_grid': None if i == 0 else self[k0].shared_grid
                }

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            self.add_module(k, RemappedGaussian2d(
                in_shape=in_shape,
                outdims=n_neurons,
                remap_layers=remap_layers,
                remap_kernel=remap_kernel,
                max_remap_amplitude=max_remap_amplitude,
                init_mu_range=init_mu_range,
                init_sigma=init_sigma,
                bias=bias,
                gauss_type=gauss_type,
                grid_mean_predictor=grid_mean_predictor,
                shared_features=shared_features,
                shared_grid=shared_grid,
                source_grid=source_grid,
            )
                            )
        self.gamma_readout = gamma_readout

class SaliencyShifter(MultiReadout, torch.nn.ModuleDict):
    def __init__(self, core, in_shapes_dict, in_shapes_dict2,  n_neurons_dict, remap_layers, remap_kernel, max_remap_amplitude,
                 init_mu_range, init_sigma, bias, gamma_readout,
                 gauss_type, grid_mean_predictor, grid_mean_predictor_type, source_grids,
                 share_features, share_grid, shared_match_ids, ):
        # super init to get the _module attribute
        super().__init__()
        ##defining a dictionary for output:

        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape_remap = in_shapes_dict[k][1:]     ## for saliency remapper
            in_shape = get_module_output(core, in_shapes_dict2[k])[1:]          ##for image

            n_neurons = n_neurons_dict[k]

            source_grid = source_grids
            shared_grid = None
            shared_transform = None
            if grid_mean_predictor is not None:
                if grid_mean_predictor_type == 'cortex':
                    source_grid = source_grids[k]
                else:
                    raise KeyError('grid mean predictor {} does not exist'.format(grid_mean_predictor_type))

            elif share_grid:
                shared_grid = {
                    'match_ids': shared_match_ids[k],
                    'shared_grid': None if i == 0 else self[k0].shared_grid
                }

            if share_features:
                shared_features = {
                    'match_ids': shared_match_ids[k],
                    'shared_features': None if i == 0 else self[k0].shared_features
                }
            else:
                shared_features = None

            y = RemappedGaussian2d_Sal(in_shape_remap=in_shape_remap, in_shape=in_shape,
                                                outdims=n_neurons,
                                                remap_layers=remap_layers,
                                                remap_kernel=remap_kernel,
                                                max_remap_amplitude=max_remap_amplitude,
                                                init_mu_range=init_mu_range,
                                                init_sigma=init_sigma,
                                                bias=bias,
                                                gauss_type=gauss_type,
                                                grid_mean_predictor=grid_mean_predictor,
                                                shared_features=shared_features,
                                                shared_grid=shared_grid,
                                                source_grid=source_grid, )


            self.add_module(k, y)

        self.gamma_readout = gamma_readout




class MultipleAttention2d(MultiReadout, torch.nn.ModuleDict):
    def __init__(
        self,
        core,
        in_shape_dict,
        n_neurons_dict,
        attention_layers,
        attention_kernel,
        bias,
        gamma_readout,
    ):
        # super init to get the _module attribute
        super().__init__()
        k0 = None
        for i, k in enumerate(n_neurons_dict):
            k0 = k0 or k
            in_shape = get_module_output(core, in_shape_dict[k])[1:]
            n_neurons = n_neurons_dict[k]

            self.add_module(
                k,
                AttentionReadout(
                    in_shape=in_shape,
                    outdims=n_neurons,
                    attention_layers=attention_layers,
                    attention_kernel=attention_kernel,
                    bias=bias
                ),
            )
        self.gamma_readout = gamma_readout




class RemappedGaussian2d_Sal(FullGaussian2d):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned. In addition, there is an image dependent remapping of neurons
    locations.
    For most parameters see:  FullGaussian2d
    Args:
        remap_layers (int): number of layers of the remapping network
        remap_kernel (int): conv kernel size of the remapping network
        max_remap_amplitude (int): maximal amplitude of remapping (factor on output of remapping network)
    """
    def __init__(
            self, *args, in_shape_remap, in_shape, remap_layers=2, remap_kernel=3, max_remap_amplitude=0.2, **kwargs
    ):

        super().__init__(*args, in_shape, **kwargs)

        channels, width, height = in_shape_remap
        remapper = nn.Sequential()
        for i in range(remap_layers - 1):
            remapper.add_module(
                f"conv{i}", nn.Conv2d(channels, channels, remap_kernel, padding=True)
            )
            remapper.add_module(f"norm{i}", nn.BatchNorm2d(channels))
            remapper.add_module(f"nonlin{i}", nn.ELU())
        else:
            remapper.add_module(
                f"conv{remap_layers}",
                nn.Conv2d(channels, 2, remap_kernel, padding=True),
            )
            remapper.add_module(f"norm{remap_layers}", nn.BatchNorm2d(2))
            remapper.add_module(f"nonlin{remap_layers}", nn.Tanh())
        self.remap_field = remapper
        self.max_remap_amplitude = max_remap_amplitude

        self.in_shape_remap = in_shape_remap
        self.in_shape = in_shape


    @staticmethod
    def init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def initialize_remap_field(self):
        self.apply(self.init_conv)


    def forward(self, x, sal, sample=None, shift=None, out_idx=None):


        offset_field = self.remap_field(sal) * self.max_remap_amplitude

        N, c, w, h = x.size()

        c_in, w_in, h_in = self.in_shape
        if (c_in, w_in, h_in) != (c, w, h):
            raise ValueError(
                "the specified feature map dimension is not the readout's expected input dimension"
            )
        feat = self.features.view(1, c, self.outdims)

        bias = self.bias
        outdims = self.outdims

        if self.batch_sample:
            # sample the grid_locations separately per image per batch
            grid = self.sample_grid(
                batch_size=N, sample=sample
            )  # sample determines sampling from Gaussian
        else:
            # use one sampled grid_locations for all images in the batch
            grid = self.sample_grid(batch_size=1, sample=sample).expand(
                N, outdims, 1, 2
            )

        if out_idx is not None:
            if isinstance(out_idx, np.ndarray):
                if out_idx.dtype == bool:
                    out_idx = np.where(out_idx)[0]
            feat = feat[:, :, out_idx]
            grid = grid[:, out_idx]
            if bias is not None:
                bias = bias[out_idx]
            outdims = len(out_idx)

        offsets = F.grid_sample(offset_field, grid, align_corners=self.align_corners)


        grid = grid + offsets.permute(0, 2, 3, 1)
        if shift is not None:
            grid = grid + shift[:, None, None, :]


        y = F.grid_sample(x, grid, align_corners=self.align_corners)


        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        return y

