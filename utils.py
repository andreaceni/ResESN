from scipy.integrate import odeint
import numpy as np
import torch
import os
import torchvision
import torchvision.transforms as transforms
from torch import nn
from esn import spectral_norm_scaling


def count_parameters(model):
    """Return total number of parameters and
    trainable parameters of a PyTorch model.
    """
    params = []
    trainable_params = []
    for p in model.parameters():
        params.append(p.numel())
        if p.requires_grad:
            trainable_params.append(p.numel())
    pytorch_total_params = sum(params)
    pytorch_total_trainableparams = sum(trainable_params)
    print('Total params:', pytorch_total_params)
    print('Total trainable params:', pytorch_total_trainableparams)


def n_params(model):
    """Return total number of parameters of the
    LinearRegression model of Scikit-Learn.
    """
    return (sum([a.size for a in model.coef_]) +
            sum([a.size for a in model.intercept_]))


def get_lorenz(N, F, num_batch=128, lag=25, washout=200, window_size=0, serieslen=20):
    # https://en.wikipedia.org/wiki/Lorenz_96_model
    def L96(x, t):
        """Lorenz 96 model with constant forcing"""
        # Setting up vector
        d = np.zeros(N)
        # Loops over indices (with operations and Python underflow indexing handling edge cases)
        for i in range(N):
            d[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
        return d

    dt = 0.01
    t = np.arange(0.0, serieslen+(lag*dt)+(washout*dt), dt)
    dataset = []
    for i in range(num_batch):
        x0 = np.random.rand(N) + F - 0.5 # [F-0.5, F+0.5]
        x = odeint(L96, x0, t)
        dataset.append(x)
    dataset = np.stack(dataset, axis=0)
    dataset = torch.from_numpy(dataset).float()

    if window_size > 0:
        windows, targets = [], []
        for i in range(dataset.shape[0]):
            w, t = get_fixed_length_windows(dataset[i], window_size, prediction_lag=lag)
        windows.append(w)
        targets.append(t)
        return torch.utils.data.TensorDataset(torch.cat(windows, dim=0), torch.cat(targets, dim=0))
    else:
        return dataset


def get_fixed_length_windows(tensor, length, prediction_lag=1):
    assert len(tensor.shape) <= 2
    if len(tensor.shape) == 1:
        tensor = tensor.unsqueeze(-1)

    windows = tensor[:-prediction_lag].unfold(0, length, 1)
    windows = windows.permute(0, 2, 1)

    targets = tensor[length+prediction_lag-1:]
    return windows, targets  # input (B, L, I), target, (B, I)


def get_mackey_glass(lag=1, washout=200, window_size=0):
    """
    Predict the next lag-th item of mackey-glass series
    """
    with open('mackey-glass.csv', 'r') as f:
        dataset = f.readlines()[0]  # single line file

    # 10k steps
    dataset = torch.tensor([float(el) for el in dataset.split(',')]).float()

    if window_size > 0:
        assert washout == 0
        dataset, targets = get_fixed_length_windows(dataset, window_size, prediction_lag=lag)
        # dataset is input, targets is output

        end_train = int(dataset.shape[0] / 2)
        end_val = end_train + int(dataset.shape[0] / 4)
        end_test = dataset.shape[0]

        train_dataset = dataset[:end_train]
        train_target = targets[:end_train]

        val_dataset = dataset[end_train:end_val]
        val_target = targets[end_train:end_val]

        test_dataset = dataset[end_val:end_test]
        test_target = targets[end_val:end_test]
    else:
        end_train = int(dataset.shape[0] / 2)
        end_val = end_train + int(dataset.shape[0] / 4)
        end_test = dataset.shape[0]

        train_dataset = dataset[:end_train-lag]
        train_target = dataset[washout+lag:end_train]

        val_dataset = dataset[end_train:end_val-lag]
        val_target = dataset[end_train+washout+lag:end_val]

        test_dataset = dataset[end_val:end_test-lag]
        test_target = dataset[end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_narma10(lag=0, washout=200):
    """
    Predict the output of a narma10 series
    """
    with open('narma10.csv', 'r') as f:
        dataset = f.readlines()[0:2]  # 2 lines file

    # 10k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    end_train = int(dataset[0].shape[0] / 2)
    end_val = end_train + int(dataset[0].shape[0] / 4)
    end_test = dataset[0].shape[0]

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_narma30(lag=0, washout=200):
    """
    Predict the output of a narma30 series
    """
    with open('narma30.csv', 'r') as f:
        dataset = f.readlines()[0:2]  # 2 lines file

    # 10k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    end_train = int(dataset[0].shape[0] / 2)
    end_val = end_train + int(dataset[0].shape[0] / 4)
    end_test = dataset[0].shape[0]

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_piSineDelay10(lag=0, washout=200, ergodic=False):
    """
    Predict the output of a sin(pi*u[t-10]) series
    given in input u[t]
    """
    if ergodic:
        with open('SineDelay10.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file
    else:
        with open('piSineDelay10.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file

    # 6k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    # firs 4k training, then 1k validation, and 1k test
    end_train = 4000
    end_val = end_train + 1000
    end_test = 6000

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_ctXOR(delay=2, washout=200, lag=0):
    """
    Predict the output of a sign(r[t])*abs(r[t])^degree series
    where r[t] = u[t-delay]*u[t-delay-1]
    """
    if delay==5:
        with open('ctXOR_delay5_degree2.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file
    elif delay==15:
        with open('ctXOR_delay15_degree10.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file
    elif delay==10:
        with open('ctXOR_delay10_degree2.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file      
    elif delay==2:
        with open('ctXOR_delay2_degree2.csv', 'r') as f:
            dataset = f.readlines()[0:2]  # 2 lines file      
    else:
        raise ValueError('Only delays 5, 10, or 15 available.')

    # 6k steps
    dataset[0] = torch.tensor([float(el) for el in dataset[0].split(',')]).float() # input
    dataset[1] = torch.tensor([float(el) for el in dataset[1].split(',')]).float() # target

    # firs 4k training, then 1k validation, and 1k test
    end_train = 4000
    end_val = end_train + 1000
    end_test = 6000

    train_dataset = dataset[0][:end_train-lag]
    train_target = dataset[1][washout+lag:end_train]

    val_dataset = dataset[0][end_train:end_val-lag]
    val_target = dataset[1][end_train+washout+lag:end_val]

    test_dataset = dataset[0][end_val:end_test-lag]
    test_target = dataset[1][end_val+washout+lag:end_test]

    return (train_dataset, train_target), (val_dataset, val_target), (test_dataset, test_target)


def get_mnist_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [57000,3000])

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)

    return train_loader, valid_loader, test_loader



def get_mnist_testing_data(bs_train,bs_test):
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                              train=False,
                                              transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=bs_train,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=bs_test,
                                              shuffle=False)
    
    return train_loader, test_loader
