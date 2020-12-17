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
        self.shift_per_neuron = {}

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

            v = self.add_module(k, RemappedGaussian2d_Sal(in_shape_remap=in_shape_remap,
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
                                              source_grid=source_grid,))
            self.shift_per_neuron[k] = v

        self.gamma_readout = gamma_readout

    def returnDict(self):
        return self.shift_per_neuron



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


class FullGaussian2dSal(nn.Module):
    """
    A readout using a spatial transformer layer whose positions are sampled from one Gaussian per neuron. Mean
    and covariance of that Gaussian are learned.
    Args:
        in_shape (list, tuple): shape of the input feature map [channels, width, height]
        outdims (int): number of output units
        bias (bool): adds a bias term
        init_mu_range (float): initialises the the mean with Uniform([-init_range, init_range])
                            [expected: positive value <=1]. Default: 0.1
        init_sigma (float): The standard deviation of the Gaussian with `init_sigma` when `gauss_type` is
            'isotropic' or 'uncorrelated'. When `gauss_type='full'` initialize the square root of the
            covariance matrix with with Uniform([-init_sigma, init_sigma]). Default: 1
        batch_sample (bool): if True, samples a position for each image in the batch separately
                            [default: True as it decreases convergence time and performs just as well]
        align_corners (bool): Keyword agrument to gridsample for bilinear interpolation.
                It changed behavior in PyTorch 1.3. The default of align_corners = True is setting the
                behavior to pre PyTorch 1.3 functionality for comparability.
        gauss_type (str): Which Gaussian to use. Options are 'isotropic', 'uncorrelated', or 'full' (default).
        grid_mean_predictor (dict): Parameters for a predictor of the mean grid locations. Has to have a form like
                        {
                        'hidden_layers':0,
                        'hidden_features':20,
                        'final_tanh': False,
                        }
        shared_features (dict): Used when the feature vectors are shared (within readout between neurons) or between
                this readout and other readouts. Has to be a dictionary of the form
               {
                    'match_ids': (numpy.array),
                    'shared_features': torch.nn.Parameter or None
                }
                The match_ids are used to match things that should be shared within or across scans.
                If `shared_features` is None, this readout will create its own features. If it is set to
                a feature Parameter of another readout, it will replace the features of this readout. It will be
                access in increasing order of the sorted unique match_ids. For instance, if match_ids=[2,0,0,1],
                there should be 3 features in order [0,1,2]. When this readout creates features, it will do so in
                that order.
        shared_grid (dict): Like `shared_features`. Use dictionary like
               {
                    'match_ids': (numpy.array),
                    'shared_grid': torch.nn.Parameter or None
                }
                See documentation of `shared_features` for specification.
        source_grid (numpy.array):
                Source grid for the grid_mean_predictor.
                Needs to be of size neurons x grid_mean_predictor[input_dimensions]
    """

    def __init__(
            self,
            in_shape,
            in_shape_remap,
            outdims,
            bias,
            init_mu_range=0.1,
            init_sigma=1,
            batch_sample=True,
            align_corners=True,
            gauss_type="full",
            grid_mean_predictor=None,
            shared_features=None,
            shared_grid=None,
            source_grid=None,
            **kwargs,
    ):

        super().__init__()

        # determines whether the Gaussian is isotropic or not
        self.gauss_type = gauss_type

        if init_mu_range > 1.0 or init_mu_range <= 0.0 or init_sigma <= 0.0:
            raise ValueError(
                "either init_mu_range doesn't belong to [0.0, 1.0] or init_sigma_range is non-positive"
            )

        # store statistics about the images and neurons
        self.in_shape_remap = in_shape_remap
        self.in_shape = in_shape
        self.outdims = outdims

        # sample a different location per example
        self.batch_sample = batch_sample

        # position grid shape
        self.grid_shape = (1, outdims, 1, 2)

        # the grid can be predicted from another grid
        self._predicted_grid = False    ##!!!!!
        self._shared_grid = False
        self._original_grid = not self._predicted_grid

        if grid_mean_predictor is None and shared_grid is None:
            self._mu = Parameter(
                torch.Tensor(*self.grid_shape)
            )  # mean location of gaussian for each neuron
        elif grid_mean_predictor is not None and shared_grid is not None:
            raise ConfigurationError(
                "Shared grid_mean_predictor and shared_grid_mean cannot both be set"
            )
        elif grid_mean_predictor is not None:
            self.init_grid_predictor(source_grid=source_grid, **grid_mean_predictor)
        elif shared_grid is not None:
            self.initialize_shared_grid(**(shared_grid or {}))

        if gauss_type == "full":
            self.sigma_shape = (1, outdims, 2, 2)
        elif gauss_type == "uncorrelated":
            self.sigma_shape = (1, outdims, 1, 2)
        elif gauss_type == "isotropic":
            self.sigma_shape = (1, outdims, 1, 1)
        else:
            raise ValueError(f'gauss_type "{gauss_type}" not known')

        self.init_sigma = init_sigma
        self.sigma = Parameter(
            torch.Tensor(*self.sigma_shape)
        )  # standard deviation for gaussian for each neuron

        self.initialize_features(**(shared_features or {}))

        if bias:
            bias = Parameter(torch.Tensor(outdims))
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.init_mu_range = init_mu_range
        self.align_corners = align_corners
        self.initialize()

    @property
    def shared_features(self):
        return self._features

    @property
    def shared_grid(self):
        return self._mu

    @property
    def features(self):
        if self._shared_features:
            return self.scales * self._features[..., self.feature_sharing_index]
        else:
            return self._features

    @property
    def grid(self):
        return self.sample_grid(batch_size=1, sample=False)

    def mu_dispersion(self):
        """
        Returns the standard deviation of the learned positions.
        Is used as a regularizer to push neurons to learn similar positions.
        Returns:
            mu_dispersion(float): average dispersion of the mean 2d-position
        """

        return self._mu.squeeze().std(0).sum()

    def feature_l1(self, average=True):
        """
        Returns the l1 regularization term either the mean or the sum of all weights
        Args:
            average(bool): if True, use mean of weights for regularization
        """
        if self._original_features:
            if average:
                return self._features.abs().mean()
            else:
                return self._features.abs().sum()
        else:
            return 0

    @property
    def mu(self):
        if self._predicted_grid:
            return self.mu_transform(self.source_grid.squeeze()).view(*self.grid_shape)
        elif self._shared_grid:
            if self._original_grid:
                return self._mu[:, self.grid_sharing_index, ...]
            else:
                return self.mu_transform(self._mu.squeeze())[
                    self.grid_sharing_index
                ].view(*self.grid_shape)
        else:
            return self._mu

    def sample_grid(self, batch_size, sample=None):
        """
        Returns the grid locations from the core by sampling from a Gaussian distribution
        Args:
            batch_size (int): size of the batch
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
        """
        with torch.no_grad():
            self.mu.clamp_(
                min=-1, max=1
            )  # at eval time, only self.mu is used so it must belong to [-1,1]
            if self.gauss_type != "full":
                self.sigma.clamp_(
                    min=0
                )  # sigma/variance i    s always a positive quantity

        grid_shape = (batch_size,) + self.grid_shape[1:]

        sample = self.training if sample is None else sample
        if sample:
            norm = self.mu.new(*grid_shape).normal_()
        else:
            norm = self.mu.new(
                *grid_shape
            ).zero_()  # for consistency and CUDA capability

        if self.gauss_type != "full":
            return torch.clamp(
                norm * self.sigma + self.mu, min=-1, max=1
            )  # grid locations in feature space sampled randomly around the mean self.mu
        else:
            return torch.clamp(
                torch.einsum("ancd,bnid->bnic", self.sigma, norm) + self.mu,
                min=-1,
                max=1,
            )  # grid locations in feature space sampled randomly around the mean self.mu

    def init_grid_predictor(
            self, source_grid, hidden_features=20, hidden_layers=0, final_tanh=False
    ):
        self._original_grid = False
        layers = [
            nn.Linear(source_grid.shape[1], hidden_features if hidden_layers > 0 else 2)
        ]

        for i in range(hidden_layers):
            layers.extend(
                [
                    nn.ELU(),
                    nn.Linear(
                        hidden_features, hidden_features if i < hidden_layers - 1 else 2
                    ),
                ]
            )

        if final_tanh:
            layers.append(nn.Tanh())
        self.mu_transform = nn.Sequential(*layers)

        source_grid = source_grid - source_grid.mean(axis=0, keepdims=True)
        source_grid = source_grid / np.abs(source_grid).max()
        self.register_buffer(
            "source_grid", torch.from_numpy(source_grid.astype(np.float32))
        )
        self._predicted_grid = True

    def initialize(self):
        """
        Initializes the mean, and sigma of the Gaussian readout along with the features weights
        """

        if not self._predicted_grid or self._original_grid:
            self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)

        if self.gauss_type != "full":
            self.sigma.data.fill_(self.init_sigma)
        else:
            self.sigma.data.uniform_(-self.init_sigma, self.init_sigma)
        self._features.data.fill_(1 / self.in_shape[0])
        if self._shared_features:
            self.scales.data.fill_(1.0)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def initialize_features(self, match_ids=None, shared_features=None):
        """
        The internal attribute `_original_features` in this function denotes whether this instance of the FullGuassian2d
        learns the original features (True) or if it uses a copy of the features from another instance of FullGaussian2d
        via the `shared_features` (False). If it uses a copy, the feature_l1 regularizer for this copy will return 0
        """
        c, w, h = self.in_shape
        self._original_features = True
        if match_ids is not None:
            assert self.outdims == len(match_ids)

            n_match_ids = len(np.unique(match_ids))
            if shared_features is not None:
                assert shared_features.shape == (
                    1,
                    c,
                    1,
                    n_match_ids,
                ), f"shared features need to have shape (1, {c}, 1, {n_match_ids})"
                self._features = shared_features
                self._original_features = False
            else:
                self._features = Parameter(
                    torch.Tensor(1, c, 1, n_match_ids)
                )  # feature weights for each channel of the core
            self.scales = Parameter(
                torch.Tensor(1, 1, 1, self.outdims)
            )  # feature weights for each channel of the core
            _, sharing_idx = np.unique(match_ids, return_inverse=True)
            self.register_buffer("feature_sharing_index", torch.from_numpy(sharing_idx))
            self._shared_features = True
        else:
            self._features = Parameter(
                torch.Tensor(1, c, 1, self.outdims)
            )  # feature weights for each channel of the core
            self._shared_features = False

    def initialize_shared_grid(self, match_ids=None, shared_grid=None):
        c, w, h = self.in_shape

        if match_ids is None:
            raise ConfigurationError("match_ids must be set for sharing grid")
        assert self.outdims == len(
            match_ids
        ), "There must be one match ID per output dimension"

        n_match_ids = len(np.unique(match_ids))
        if shared_grid is not None:
            assert shared_grid.shape == (
                1,
                n_match_ids,
                1,
                2,
            ), f"shared grid needs to have shape (1, {n_match_ids}, 1, 2)"
            self._mu = shared_grid
            self._original_grid = False
            self.mu_transform = nn.Linear(2, 2)
            self.mu_transform.bias.data.fill_(0.0)
            self.mu_transform.weight.data = torch.eye(2)
        else:
            self._mu = Parameter(
                torch.Tensor(1, n_match_ids, 1, 2)
            )  # feature weights for each channel of the core
        _, sharing_idx = np.unique(match_ids, return_inverse=True)
        self.register_buffer("grid_sharing_index", torch.from_numpy(sharing_idx))
        self._shared_grid = True

    def forward(self, x, sal, sample=None, shift=None, out_idx=None):
        """
        Propagates the input forwards through the readout
        Args:
            x: input data
            sample (bool/None): sample determines whether we draw a sample from Gaussian distribution, N(mu,sigma), defined per neuron
                            or use the mean, mu, of the Gaussian distribution without sampling.
                           if sample is None (default), samples from the N(mu,sigma) during training phase and
                             fixes to the mean, mu, during evaluation phase.
                           if sample is True/False, overrides the model_state (i.e training or eval) and does as instructed
            shift (bool): shifts the location of the grid (from eye-tracking data)
            out_idx (bool): index of neurons to be predicted
        Returns:
            y: neuronal activity
        """
        N, c, w, h = x.size()
        #c_in, w_in, h_in = self.in_shape
        #if (c_in, w_in, h_in) != (c, w, h):
        #    raise ValueError(
        #        "the specified feature map dimension is not the readout's expected input dimension"
        #    )
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

        if shift is not None:
            grid = grid + shift[:, None, None, :]

        y = F.grid_sample(x, grid, align_corners=self.align_corners)
        y = (y.squeeze(-1) * feat).sum(1).view(N, outdims)

        if self.bias is not None:
            y = y + bias
        return y

    def __repr__(self):
        c, w, h = self.in_shape
        r = self.gauss_type + " "
        r += (
                self.__class__.__name__
                + " ("
                + "{} x {} x {}".format(c, w, h)
                + " -> "
                + str(self.outdims)
                + ")"
        )
        if self.bias is not None:
            r += " with bias"
        if self._shared_features:
            r += ", with {} features".format(
                "original" if self._original_features else "shared"
            )

        if self._predicted_grid:
            r += ", with predicted grid"
        if self._shared_grid:
            r += ", with {} grid".format(
                "original" if self._original_grid else "shared"
            )

        for ch in self.children():
            r += "  -> " + ch.__repr__() + "\n"
        return r




class RemappedGaussian2d_Sal(FullGaussian2dSal):
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
            self, *args, remap_layers=2, remap_kernel=3, max_remap_amplitude=0.2, **kwargs
    ):

        super().__init__(*args, **kwargs)
        channels, width, height = self.in_shape_remap
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


