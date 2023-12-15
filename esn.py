"""
porting of https://github.com/gallicch/DeepRC-TF/blob/master/DeepRC.py
in pytorch.

If you use this code in your work, please cite the following paper,
in which the concept of Deep Reservoir Computing has been introduced:

Gallicchio,  C.,  Micheli,  A.,  Pedrelli,  L.: Deep  reservoir  computing:
A  critical  experimental  analysis.    Neurocomputing268,  87–99  (2017).
https://doi.org/10.1016/j.neucom.2016.12.08924.
"""

import torch
from torch import nn
import numpy as np

def sparse_eye_init(M: int) -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse identity matrix for the
    re-scaling of the sparse recurrent kernel in presence of non-zero leakage.
    The neurons are connected according to a ring topology, where each neuron
    receives input only from one neuron and propagates its activation only to
    one other neuron. All the non-zero elements are set to 1.

    :param M: number of hidden units
    :return: dense weight matrix
    """
    dense_shape = torch.Size([M, M])

    # gives the shape of a ring matrix:
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, :] = i
    values = torch.ones(M)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """ Generates an M x N matrix to be used as sparse (input) kernel
    For each row only C elements are non-zero (i.e., each input dimension is
    projected only to C neurons). The non-zero elements are generated randomly
    from a uniform distribution in [-1,1]

    :param M: number of rows
    :param N: number of columns
    :param C: number of nonzero elements
    :return: MxN dense matrix
    """
    dense_shape = torch.Size([M, N])  # shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k = k + 1
    #values = 2 * (2 * np.random.rand(M * C).astype('f') - 1) # isn't this in [-2,2]?
    values = 2 * np.random.rand(M * C).astype('f') - 1
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_recurrent_tensor_init(M: int, C: int = 1, distrib: str = 'uniform') -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse recurrent kernel.
    For each column only C elements are non-zero (i.e., each recurrent neuron
    take sinput from C other recurrent neurons). The non-zero elements are
    generated randomly from a uniform distribution in [-1,1].

    :param M: number of hidden units
    :param C: number of nonzero elements
    :param distrib: initialisation strategy. It can be 'uniform' or 'normal'
    :return: MxM dense matrix
    """
    assert M >= C
    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k = k + 1
    if distrib == 'uniform':
        #values = 2 * (2 * np.random.rand(M * C).astype('f') - 1) # isn't this in [-2,2]?
        values = 2 * np.random.rand(M * C).astype('f') - 1
    elif distrib == 'normal':
        values =  np.random.randn(M * C).astype('f') / np.sqrt(C) # circular law (rescaling)
    else:
        raise ValueError("Invalid distributon <<" + distrib + ">>. Only uniform and normal allowed.")
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales W to have rho(W) = rho_desired

    :param W:
    :param rho_desired:
    :return:
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)


def fast_spectral_rescaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales a W uniformly sampled in (-1,1) to have rho(W) = rho_desired. 
    This method is fast since we don't need to compute the spectrum of W, which is very slow.

    NB: this method works only if W is uniformly sampled in (-1,1). 
    In particular, W must be fully connected!

    :param W: must be a square matrix uniformly sampled in (-1,1), fully connected.
    :param rho_desired:
    :return:
    """
    units = W.shape[0]
    value  = (rho_desired / np.sqrt(units)) * (6/np.sqrt(12))
    W = value * W
    return W


class ReservoirCell(torch.nn.Module):
    def __init__(self, input_size, units, input_scaling=1., spectral_radius=0.99,
                 leaky=1, connectivity_input=10, connectivity_recurrent=10, 
                 bias_bool=True, alpha=0, beta=1, distrib='uniform', nonlin='tanh',
                 effective_rescaling=True, bias_scaling=None):  
        """ Shallow reservoir to be used as cell of a Recurrent Neural Network.

        :param input_size: number of input units
        :param units: number of recurrent neurons in the reservoir
        :param input_scaling: max abs value of a weight in the input-reservoir
            connections. Note that whis value also scales the unitary input bias
        :param spectral_radius: max abs eigenvalue of the recurrent matrix
        :param leaky: leaking rate constant of the reservoir
        :param connectivity_input: number of outgoing connections from each
            input unit to the reservoir
        :param connectivity_recurrent: number of incoming recurrent connections
            for each reservoir unit
        :param bias_bool: boolean variable decides whether to apply a bias or not
            inside the nonlinearity
        :param alpha, beta: scalar values hyperparameters of the ResESN model.
        :param nonlin: string value can be either 'tanh' or 'identity'
        :param effective_rescaling: True or False. If True, then rescale the
            effective recurrent matrix to have a desired spectral radius.
        :param bias_scaling: If None, then rescale the bias vectore with the
            scalar value input_scaling.
        """
        super().__init__()

        self.input_size = input_size
        self.units = units
        self.state_size = units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky = leaky
        self.connectivity_input = connectivity_input
        self.connectivity_recurrent = connectivity_recurrent

        self.kernel = sparse_tensor_init(input_size, self.units,
                                         self.connectivity_input) * self.input_scaling
        self.kernel = nn.Parameter(self.kernel, requires_grad=False)

        W = sparse_recurrent_tensor_init(self.units, C=self.connectivity_recurrent, distrib=distrib)
        # re-scale the weight matrix to control the effective spectral radius
        # of the linearized system
        if effective_rescaling and self.leaky != 1:
            I = sparse_eye_init(self.units)
            W = W * self.leaky + (I * (1 - self.leaky))
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = (W + I * (self.leaky - 1)) * (1 / self.leaky)
        else:
            if distrib == 'normal':
                W = spectral_radius * W # NB: W was already rescaled to 1 (circular law)
            elif distrib == 'uniform' and connectivity_recurrent == units: # fully connected uniform
                W = fast_spectral_rescaling(W, spectral_radius)
            else: # sparse connections uniform
                W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = W
        self.recurrent_kernel = nn.Parameter(self.recurrent_kernel, requires_grad=False)

        if bias_bool:
            if bias_scaling is None:
                self.bias_scaling = self.input_scaling
            else:
                self.bias_scaling = bias_scaling
            # uniform init in [-1, +1] times bias_scaling
            self.bias = (2 * torch.rand(self.units) - 1) * self.bias_scaling
            self.bias = nn.Parameter(self.bias, requires_grad=False)
        else:
            # zero bias
            self.bias = torch.zeros(self.units) 
            self.bias = nn.Parameter(self.bias, requires_grad=False)

        if alpha > 0:
            # random orthogonal matrix
            Q, _ = torch.linalg.qr(2*torch.rand(units,units)-1)
            self.ortho, self.beta = Q, torch.tensor([beta])
            self.ortho = nn.Parameter(self.ortho, requires_grad=False)
            self.beta = nn.Parameter(self.beta, requires_grad=False)
        self.alpha = torch.tensor([alpha])
        self.alpha = nn.Parameter(self.alpha, requires_grad=False)

        self.nonlin = nonlin

    def forward(self, xt, h_prev):
        """ Computes the output of the cell given the input and previous state.

        :param xt:
        :param h_prev: h[t-1]
        :return: ht, ht
        """
        input_part = torch.mm(xt, self.kernel)
        state_part = torch.mm(h_prev, self.recurrent_kernel)

        if self.nonlin == 'identity':
            output = input_part + self.bias + state_part
        elif self.nonlin == 'tanh':
            output = torch.tanh(input_part + self.bias + state_part)
        else:
            raise ValueError("Invalid nonlinearity <<" + self.nonlin + ">>. Only tanh and identity allowed.")
        if self.alpha > 0:
            h_prev = torch.mm(h_prev, self.ortho) # shuffle internal state
            out = h_prev * self.alpha + output * self.beta
            return out, out
        else: 
            leaky_output = h_prev * (1 - self.leaky) + output * self.leaky
            return leaky_output, leaky_output


class ReservoirLayer(torch.nn.Module):
    def __init__(self, input_size, units, input_scaling=1., spectral_radius=0.99,
                 leaky=1, connectivity_input=10, connectivity_recurrent=10, 
                 bias_bool=True, alpha=0, beta=1, distrib='uniform', nonlin='tanh',
                 effective_rescaling=True, bias_scaling=None):
        """ Shallow reservoir to be used as Recurrent Neural Network layer.

        :param input_size: number of input units
        :param units: number of recurrent neurons in the reservoir
        :param input_scaling: max abs value of a weight in the input-reservoir
            connections. Note that whis value also scales the unitary input bias
        :param spectral_radius: max abs eigenvalue of the recurrent matrix
        :param leaky: leaking rate constant of the reservoir
        :param connectivity_input: number of outgoing connections from each
            input unit to the reservoir
        :param connectivity_recurrent: number of incoming recurrent connections
            for each reservoir unit
        """
        super().__init__()
        self.net = ReservoirCell(input_size, units, input_scaling,
                                 spectral_radius, leaky, connectivity_input,
                                 connectivity_recurrent, 
                                 bias_bool, alpha, beta, distrib, nonlin,
                                 effective_rescaling, bias_scaling)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.net.units)

    def forward(self, x, h_prev=None):
        """ Computes the output of the cell given the input and previous state.

        :param x:
        :param h_prev: h[0]
        :return: h, ht
        """

        if h_prev is None:
            h_prev = self.init_hidden(x.shape[0]).to(x.device)

        hs = []
        for t in range(x.shape[1]):
            xt = x[:, t]
            _, h_prev = self.net(xt, h_prev)
            hs.append(h_prev)
        hs = torch.stack(hs, dim=1)
        return hs, h_prev


class DeepReservoir(torch.nn.Module):
    def __init__(self, input_size=1, tot_units=100, n_layers=1, concat=False,
                 input_scaling=1, inter_scaling=1,
                 spectral_radius=0.99, leaky=1,
                 connectivity_recurrent=10,
                 connectivity_input=10,
                 connectivity_inter=10,
                 bias_bool=True,
                 alpha=0,
                 beta=1,
                 distrib='uniform',
                 nonlin='tanh',
                 #
                 effective_rescaling=True, 
                 bias_scaling=None,
                 ):
        """ Deep Reservoir layer.
        The implementation realizes a number of stacked RNN layers using the
        ReservoirCell as core cell. All the reservoir layers share the same
        hyper-parameter values (i.e., same number of recurrent neurons, spectral
        radius, etc. ).

        :param input_size:
        :param tot_units: number of recurrent units.
            if concat == True this is the total number of units
            if concat == False this is the number of units for each
                reservoir level
        :param n_layers:
        :param concat: if True the returned state is given by the
            concatenation of all the states in the reservoir levels
        :param input_scaling: scaling coeff. of the first reservoir layer
        :param inter_scaling: scaling coeff. of all the other levels (> 1)
        :param spectral_radius:
        :param leaky: leakage coefficient of all levels
        :param connectivity_recurrent:
        :param connectivity_input: input connectivity coefficient of the input weight matrix
        :param connectivity_inter: input connectivity coefficient of all the inter-levels weight matrices
        """
        super().__init__()
        self.n_layers = n_layers
        self.tot_units = tot_units
        self.concat = concat
        self.batch_first = True  # DeepReservoir only supports batch_first

        # in case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed,
        # i.e., the number of trainable parameters does not depend on concat
        if concat:
            self.layers_units = np.int(tot_units / n_layers)
        else:
            self.layers_units = tot_units

        input_scaling_others = inter_scaling
        connectivity_input_1 = connectivity_input
        connectivity_input_others = connectivity_inter

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            ReservoirLayer(
                input_size=input_size,
                units=self.layers_units + tot_units % n_layers,
                input_scaling=input_scaling,
                spectral_radius=spectral_radius,
                leaky=leaky,
                connectivity_input=connectivity_input_1,
                connectivity_recurrent=connectivity_recurrent,
                bias_bool=bias_bool,
                alpha=alpha,
                beta=beta,
                distrib=distrib,
                nonlin=nonlin,
                #
                effective_rescaling=effective_rescaling, 
                bias_scaling=bias_scaling)
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concat=True
        last_h_size = self.layers_units + tot_units % n_layers
        for _ in range(n_layers - 1):
            reservoir_layers.append(
                ReservoirLayer(
                    input_size=last_h_size,
                    units=self.layers_units,
                    input_scaling=input_scaling_others,
                    spectral_radius=spectral_radius,
                    leaky=leaky,
                    connectivity_input=connectivity_input_others,
                    connectivity_recurrent=connectivity_recurrent,
                    bias_bool=bias_bool,
                    alpha=alpha,
                    beta=beta,
                    nonlin=nonlin,
                    #
                    effective_rescaling=effective_rescaling, 
                    bias_scaling=bias_scaling
                )
            )
            last_h_size = self.layers_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    def forward(self, X):
        """ compute the output of the deep reservoir.

        :param X:
        :return: hidden states (B, T, F), last state (L, B, F)
        """
        states = []  # list of all the states in all the layers
        states_last = []  # list of the states in all the layers for the last time step
        # states_last is a list because different layers may have different size.

        for res_idx, res_layer in enumerate(self.reservoir):
            [X, h_last] = res_layer(X)
            states.append(X)
            states_last.append(h_last)

        if self.concat:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]
        return states, states_last
