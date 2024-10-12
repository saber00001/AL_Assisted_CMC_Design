import os
import torch

import math
from abc import ABC, abstractmethod
from math import pi
from typing import List, Union

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import IdentityMCMultiOutputObjective
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex

import pandas as pd

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),
}

import torch
from botorch.exceptions.errors import UnsupportedError
from botorch.test_functions.base import (
    ConstrainedBaseTestProblem,
    MultiObjectiveTestProblem,
)
from botorch.test_functions.synthetic import Branin, Levy
from botorch.utils.sampling import sample_hypersphere, sample_simplex
from botorch.utils.transforms import unnormalize
from scipy.special import gamma
from torch import Tensor
from torch.distributions import MultivariateNormal
import os
import argparse

import botorch
from botorch.utils import standardize

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement
from botorch.utils.transforms import unnormalize
import matplotlib.pyplot as plt
import numpy as np
import random


SEED = 43
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

input_file = './V8_input_M6.xlsx'
output_file = './V8_output_M6.xlsx'

input0 = pd.read_excel(input_file, header=None)
output0 = pd.read_excel(output_file, header=None)

input0_np = input0.values[1:, 1:]
output0_np = output0.values[1:, 1:]

d = input0_np.shape[1]
M = output0_np.shape[1]

input0_max = np.max(input0_np, axis=0)
input0_min = np.min(input0_np, axis=0)
output0_max = np.max(output0_np, axis=0)
output0_min = np.min(output0_np, axis=0)

output0_mean = np.mean(output0_np, axis=0)
ref_point_list = list(output0_mean * 0.9)

ref_point_list[0] = 60.0
ref_point_list[1] = 8.0
ref_point_list[2] = 0.7

ref_point_tensor = torch.tensor(ref_point_list).to(**tkwargs)

input0_min_torch = torch.tensor(input0_min.astype(np.float64)).to(**tkwargs)
input0_max_torch = torch.tensor(input0_max.astype(np.float64)).to(**tkwargs)

output0_min_torch = torch.tensor(output0_min.astype(np.float64)).to(**tkwargs)
output0_max_torch = torch.tensor(output0_max.astype(np.float64)).to(**tkwargs)

standard_input_bounds = torch.zeros(2, d, **tkwargs)
standard_input_bounds[0, :] = input0_min_torch
standard_input_bounds[1, :] = input0_max_torch

standard_output_bounds = torch.zeros(2, M, **tkwargs)
standard_output_bounds[0, :] = output0_min_torch
standard_output_bounds[1, :] = output0_max_torch

x_qparego = torch.tensor(input0_np.astype(np.float64)).to(**tkwargs)
obj_qparego = torch.tensor(output0_np.astype(np.float64)).to(**tkwargs)
nor_x = normalize(x_qparego, standard_input_bounds)
nor_y = normalize(obj_qparego, standard_output_bounds)

n_samples = input0_np.shape[0]
rand_arr = list(range(n_samples))
random.shuffle(rand_arr)

split_index = int(n_samples)

nor_x_train = nor_x[rand_arr[:split_index]]
#nor_x_test = nor_x[rand_arr[split_index:]]
x_train = x_qparego[rand_arr[:split_index]]
#x_test = x_qparego[rand_arr[split_index:]]

nor_y_train = nor_y[rand_arr[:split_index]]
#nor_y_test = nor_y[rand_arr[split_index:]]
y_train = obj_qparego[rand_arr[:split_index]]
#y_test = obj_qparego[rand_arr[split_index:]]

standard_bounds2 = torch.zeros(2, d, **tkwargs)
standard_bounds2[1] = 1

input_scale = torch.tensor((input0_max - input0_min).astype(np.float64)).to(**tkwargs)

def initialize_model(train_x, train_y):
    
    models = []
    for i in range(train_y.shape[-1]):
        models.append(
            SingleTaskGP(
                train_x, train_y[..., i : i + 1], outcome_transform=Standardize(m=1)
            )
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model

from botorch import fit_gpytorch_mll
from sklearn.model_selection import KFold

def find_min_index(tensor_list):
    values = [t.item() for t in tensor_list]
    min_index = values.index(min(values))

    return min_index

gpr_model_list = []
re_list = []
re_list_all = []
for i in range(5, 11):
    
    gpr_model_kold_list = []
    re_fold_list = []
    re_fold_list_all = []
    k = i
    kf = KFold(n_splits=k, shuffle=True)
    count = 0
    
    mll_qnehvi, model_qnehvi = initialize_model(
        nor_x_train, y_train,
    )
    
    for train_index, valid_index in kf.split(nor_x_train):
        
        X_train, X_valid = nor_x_train[train_index], nor_x_train[valid_index]
        Y_train, Y_valid = nor_y_train[train_index], nor_y_train[valid_index]
        y_train_0, y_valid_0 = y_train[train_index], y_train[valid_index]
        
        mll, model = initialize_model(
            X_train, y_train_0,
        )
        
        for a_model, b_model in zip(model_qnehvi.models, model.models):

            b_model.covar_module.base_kernel.lengthscale = a_model.covar_module.base_kernel.lengthscale
            b_model.covar_module.outputscale = a_model.covar_module.outputscale

            if hasattr(a_model.likelihood, 'noise') and hasattr(b_model.likelihood, 'noise'):
                b_model.likelihood.noise = a_model.likelihood.noise.clone()
        
        #if count == 0:
        fit_gpytorch_mll(mll)
        
        gpr_model_kold_list.append(model)
        
        model.eval()
        posterior = model.posterior(X_valid)
        means2 = posterior.mean
        stds2 = posterior.stddev
        
        valid_re = torch.abs((means2 - y_valid_0)/y_valid_0)
        print("k: ", k, "valid_re_ave: ", torch.mean(valid_re, dim=0))
        
        re_fold_list.append(torch.sum(torch.mean(valid_re, dim=0)))
        re_fold_list_all.append(torch.mean(valid_re, dim=0))
        
        for a_model, b_model in zip(model.models, model_qnehvi.models):
            
            b_model.covar_module.base_kernel.lengthscale = a_model.covar_module.base_kernel.lengthscale
            b_model.covar_module.outputscale = a_model.covar_module.outputscale

            if hasattr(a_model.likelihood, 'noise') and hasattr(b_model.likelihood, 'noise'):
                b_model.likelihood.noise = a_model.likelihood.noise.clone()
        
        count = count + 1
    
    print("k: ", k)
    print("re_fold_list: ", re_fold_list)
    min_index = find_min_index(re_fold_list)
    print("re_fold_list[min_index]: ", re_fold_list[min_index])
    
    gpr_model_list.append(gpr_model_kold_list[min_index])
    re_list.append(re_fold_list[min_index])
    re_list_all.append(re_fold_list_all[min_index])

min_index_final = find_min_index(re_list)
best_model = gpr_model_list[min_index_final]

print("min_index_final: ", min_index_final)
print("re_list_all[min_index_final]: ", re_list_all[min_index_final])

split_index = int(n_samples * 0.8)

nor_x_train = nor_x[rand_arr[:split_index]]
nor_x_test = nor_x[rand_arr[split_index:]]
x_train = x_qparego[rand_arr[:split_index]]
x_test = x_qparego[rand_arr[split_index:]]

nor_y_train = nor_y[rand_arr[:split_index]]
nor_y_test = nor_y[rand_arr[split_index:]]
y_train = obj_qparego[rand_arr[:split_index]]
y_test = obj_qparego[rand_arr[split_index:]]


mll_qnehvi, model_qnehvi = initialize_model(
        nor_x_train, y_train,
    )

for a_model, b_model in zip(best_model.models, model_qnehvi.models):
            
    b_model.covar_module.base_kernel.lengthscale = a_model.covar_module.base_kernel.lengthscale
    b_model.covar_module.outputscale = a_model.covar_module.outputscale

    if hasattr(a_model.likelihood, 'noise') and hasattr(b_model.likelihood, 'noise'):
        b_model.likelihood.noise = a_model.likelihood.noise.clone()

model_qnehvi.eval()
posterior = model_qnehvi.posterior(nor_x_test)
means2 = posterior.mean
stds2 = posterior.stddev

print("means2: ", means2)
print("y_test: ", y_test)
test_re = torch.abs((means2 - y_test)/y_test)
print("test_re_ave: ", torch.mean(test_re, dim=0))

BATCH_SIZE = 32
NUM_RESTARTS = 20 #50
RAW_SAMPLES = 128 #1024

def constraint_fn(Z):
    return 8.0 - Z[..., 1]

def optimize_qnehvi_and_get_observation(model, train_x, train_obj, sampler, loop):
    
    """Optimizes the qNEHVI acquisition function, and returns a new candidate and observation."""
    train_x = normalize(train_x, standard_input_bounds)
    
    print("====================================")
    print("ref_point_list: ", ref_point_list)
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point_list,  # use known reference point
        #best_f=train_obj.max(),
        X_baseline=train_x,
        sampler=sampler,
        prune_baseline=True,
        # define an objective that specifies which outcomes are the objectives
        objective=IdentityMCMultiOutputObjective(outcomes=[0, 1, 2]),
        # specify that the constraint is on the last outcome
        constraints=[constraint_fn],
    )
    
    #import sys
    #sys.exit(0)
    
    #inequality_constraints = inequality_constraints.to(**tkwargs)
    '''
    equality_constraints = [
    (torch.tensor([0, 1, 2]), torch.tensor([input_scale[0], input_scale[1], input_scale[2]]).to(**tkwargs), 
    1.6 - standard_input_bounds[0, 0] - standard_input_bounds[0, 1] - standard_input_bounds[0, 2]),
    
    ]
    
    inequality_constraints = [(torch.tensor([3, 5]), torch.tensor([input_scale[3], input_scale[5]]).to(**tkwargs),  
                             -(standard_input_bounds[0, 3] + standard_input_bounds[0, 5])), 
                             (torch.tensor([6]), torch.tensor([input_scale[6]]).to(**tkwargs),  
                             -(standard_input_bounds[0, 6] - 2.5)),
                             (torch.tensor([6]), torch.tensor([-input_scale[6]]).to(**tkwargs),  
                             (standard_input_bounds[0, 6] - 3.0))
                             ]
    '''
    
    equality_constraints = [
    (torch.tensor([0, 1, 2]), torch.tensor([input_scale[0], input_scale[1], input_scale[2]]).to(**tkwargs), 
    1.6 - standard_input_bounds[0, 0] - standard_input_bounds[0, 1] - standard_input_bounds[0, 2]),
    (torch.tensor([3, 4, 5, 6, 7]), torch.tensor([input_scale[3], input_scale[4], input_scale[5], input_scale[6], input_scale[7]]).to(**tkwargs), 
    50.0 - standard_input_bounds[0, 3] - standard_input_bounds[0, 4] - standard_input_bounds[0, 5] - standard_input_bounds[0, 6] - standard_input_bounds[0, 7])
    ] 
    
    inequality_constraints = [(torch.tensor([4, 6]), torch.tensor([input_scale[4], input_scale[6]]).to(**tkwargs),  
                             -(standard_input_bounds[0, 4] + standard_input_bounds[0, 6])),
                             (torch.tensor([7]), torch.tensor([input_scale[7]]).to(**tkwargs),  
                             -(standard_input_bounds[0, 7] - 2.5)),
                             (torch.tensor([7]), torch.tensor([-input_scale[7]]).to(**tkwargs),  
                             (standard_input_bounds[0, 7] - 3.0))
                             ]
    
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds2,
        q=BATCH_SIZE,
        inequality_constraints = inequality_constraints,
        equality_constraints = equality_constraints,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    
    model.eval()
    posterior1 = model.posterior(candidates)
    means1 = posterior1.mean
    stds1 = posterior1.stddev
    
    print("means1: ", means1)
    print("stds1: ", stds1)
    
    nor_means1 = (means1 - output0_min_torch)/(output0_max_torch - output0_min_torch)
    
    indices_to_remove = []
    
    for i in range(means1.shape[0]):
        if (means1[i, 0] < 60.0 or means1[i, 1] < 10.0 or means1[i, 2] < 0.6):
            indices_to_remove.append(i)
    
    print("nor_means1: ", nor_means1)
    
    target = 2.0 * nor_means1[:, 1] + nor_means1[:, 0] + nor_means1[:, 2]
    
    sorted_indices = torch.argsort(target, descending=True)
    sorted_tensor = target[sorted_indices]
    print("Sorted Tensor (Descending):", sorted_tensor)
    print("Sorted Indices (Descending):", sorted_indices)
    
    
    new_x = unnormalize(candidates.detach(), bounds=standard_input_bounds)
    new_x_np = new_x.numpy()
    
    input_column = ['CMC/g', '添加剂2(玉米淀粉)/g',	'添加剂1(木薯淀粉)/g',	'水',	'溶剂1(尿素)/mL',
                 	'PH调节剂2(NaOH)/mL', '尿素/NaOH比例', '溶剂2(氨水)/mL',	'稀释剂(甘油)/mL'	, '合成温度'
    ]
    
    ratio = new_x_np[:, 4]/new_x_np[:, 5]
    #vol_H2O = 50 - (new_x_np[:, 3] + new_x_np[:, 4] + new_x_np[:, 5] + new_x_np[:, 6])
    #new_x_np_insert = np.insert(new_x_np, 3, vol_H2O, axis=1)
    new_x_np_insert = np.insert(new_x_np, 6, ratio, axis=1)
    
    df = pd.DataFrame(new_x_np_insert, columns=input_column)
    df = df.reindex(sorted_indices.detach().numpy())
    #df.set_index(drop=True, inplace=True)
    
    df.iloc[:, 0] = df.iloc[:, 0].astype(float).round(1)
    df.iloc[:, 1] = df.iloc[:, 1].astype(float).round(1)
    df.iloc[:, 2] = df.iloc[:, 2].astype(float).round(1)
    df.iloc[:, 4] = df.iloc[:, 4].astype(float).round(1)
    df.iloc[:, 5] = df.iloc[:, 5].astype(float).round(1)
    df.iloc[:, 7] = df.iloc[:, 7].astype(float).round(1)
    
    df.iloc[:, 6] = df.iloc[:, 4]/df.iloc[:, 5]
    df.iloc[:, 6] = df.iloc[:, 6].astype(float).round(1)
    
    #df = df.astype(float).round(1)
    df.iloc[:, 3] = df.iloc[:, 3].astype(int)
    #df.iloc[:, 8] = df.iloc[:, 8].astype(int)
    df.iloc[:, 8] = df.iloc[:, 8].astype(float).round(1)
    df.iloc[:, 9] = df.iloc[:, 9].astype(int)
    
    index = range(len(df))
    df.insert(0, 'Index', index)
    #df.to_excel('candidate_input.xlsx', header=False, index=False, engine='openpyxl')
    df.to_excel('candidate_input_all.xlsx', index=False, engine='openpyxl')
    
    df.drop(indices_to_remove, inplace=True)
    df.to_excel('candidate_input.xlsx', index=False, engine='openpyxl')
    
    means1_np = means1.detach().numpy()
    
    df = pd.DataFrame(means1_np)
    df = df.reindex(sorted_indices.detach().numpy())
    #df.drop(indices_to_remove, inplace=True)
    index = range(len(df))
    df.insert(0, 'Index', index)
    df.to_excel('candidate_output_all.xlsx', header=False, index=False, engine='openpyxl')
    
    df.drop(indices_to_remove, inplace=True)
    df.to_excel('candidate_output.xlsx', header=False, index=False, engine='openpyxl')
    
    stds1_np = stds1.detach().numpy()
    df = pd.DataFrame(stds1_np)
    df.to_excel('candidate_output_std.xlsx', header=False, index=False, engine='openpyxl')
    