import os
import sys
import errno
import pickle
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as skmetrics
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import torch.nn.utils.prune as prune
import io
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Union
from torch.nn import functional as F
import gzip

class AddGaussianNoise(object):
	def __init__(self, mean=0., std=1.):
		self.std = std
		self.mean = mean
		
	def __call__(self, tensor):
		return tensor + torch.randn(tensor.size()) * self.std + self.mean
	
	def __repr__(self):
		return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_prune_params(model, name='weight') -> List[Tuple[nn.Parameter, str]]:
    # iterate over network layers
    params_to_prune = []
    for _, module in model.named_children():
        for name_, param in module.named_parameters():
            if name in name_:
                params_to_prune.append((module, name))
    return params_to_prune


def get_prune_summary(model, name='weight') -> Dict[str, Union[List[Union[str, float]], float]]:
    num_global_zeros, num_layer_zeros, num_layer_weights = 0, 0, 0
    num_global_weights = 0
    global_prune_percent, layer_prune_percent = 0, 0
    prune_stat = {'Layers': [],
                  'Weight Name': [],
                  'Percent Pruned': [],
                  'Total Pruned': []}
    params_pruned = get_prune_params(model, 'weight')

    for layer, weight_name in params_pruned:

        num_layer_zeros = torch.sum(
            getattr(layer, weight_name) == 0.0).item()
        num_global_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        num_global_weights += num_layer_weights
        layer_prune_percent = num_layer_zeros / num_layer_weights * 100
        prune_stat['Layers'].append(layer.__str__())
        prune_stat['Weight Name'].append(weight_name)
        prune_stat['Percent Pruned'].append(
            f'{num_layer_zeros} / {num_layer_weights} ({layer_prune_percent:.5f}%)')
        prune_stat['Total Pruned'].append(f'{num_layer_zeros}')

    global_prune_percent = num_global_zeros / num_global_weights

    prune_stat['global'] = global_prune_percent
    return prune_stat

def l1_prune(model, amount=0.00, name='weight', verbose=True):
    """
        Prunes the model param by param by given amount
    """
    params_to_prune = get_prune_params(model, name)
    
    for params, name in params_to_prune:
        prune.l1_unstructured(params, name, amount)
        
    if verbose:
        info = get_prune_summary(model, name)
        global_pruning = info['global']
        info.pop('global')
        print(tabulate(info, headers='keys', tablefmt='github'))
        print("Total Pruning: {}%".format(global_pruning * 100))

def produce_mask_from_model_in_place(model):
    # use prune with 0 amount to init mask for the model
    # create mask in-place on model
    if check_mask_object_from_model(model):
        return
    l1_prune(model=model,
                amount=0.00,
                name='weight',
                verbose=False)
    layer_to_masked_positions = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                if weight_params.is_cuda:
                    layer_to_masked_positions[layer] = list(zip(*np.where(weight_params.cpu() == 0)))
                else:
                    layer_to_masked_positions[layer] = list(zip(*np.where(weight_params == 0)))
        
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                for pos in layer_to_masked_positions[layer]:
                    mask[pos] = 0
                    
@torch.no_grad()
def fed_avg(models: List[nn.Module], weight: float, device='cuda:0'):
    """
        models: list of nn.modules(unpruned/pruning removed)
        weights: normalized weights for each model
        cls:  Class of original model
    """
    aggr_model = models[0].__class__().to(device)
    model_params = []
    num_models = len(models)
    for model in models:
        model_params.append(dict(model.named_parameters()))

    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for i in range(num_models):
            weighted_param = torch.mul(
                model_params[i][name].data, weight)
            param.data.copy_(param.data + weighted_param)
    return aggr_model

@torch.no_grad()
def weighted_fedavg(worker_to_weight, worker_to_model, device='cuda:0'):
    """
        weights_to_model: dict of accuracy to model, with accuracy being weight
    """
    benigh_workers = worker_to_weight.keys()
    weights = [worker_to_weight[w] for w in benigh_workers]
    models = [make_prune_permanent(worker_to_model[w]) for w in benigh_workers]

    aggr_model = models[0].__class__().to(device)
    model_params = []
    num_models = len(models)
    for model in models:
        model_params.append(dict(model.named_parameters()))

    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for i in range(num_models):
            weighted_param = torch.mul(
                model_params[i][name].data, weights[i])
            param.data.copy_(param.data + weighted_param)
    return aggr_model

def apply_local_mask(model, mask):
    # apply mask in-place to model
    # direct multiplying instead of adding mask object
    if not mask:
        return
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                weight_params.data.copy_(torch.tensor(np.multiply(weight_params.data, mask[layer])))


def create_model_no_prune(cls, device='cuda:0') -> nn.Module:
    model = cls().to(device)
    return model

def create_model(cls, device='cuda:0') -> nn.Module:
    """
        Returns new model pruned by 0.00 %. This is necessary to create buffer masks
    """
    model = cls().to(device)
    l1_prune(model, amount=0.00, name='weight', verbose=False)
    return model

def copy_model(model: nn.Module, device='cuda:0'):
    """
        Returns a copy of the input model.
        Note: the model should have been pruned for this method to work to create buffer masks and whatnot.
    """
    produce_mask_from_model_in_place(model)
    new_model = create_model(model.__class__, device)
    source_params = dict(model.named_parameters())
    source_buffer = dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_params[name].data)
    for name, buffer_ in new_model.named_buffers():
        buffer_.data.copy_(source_buffer[name].data)
    return new_model


metrics = MetricCollection([
    Accuracy('MULTICLASS', num_classes = 10),
    Precision('MULTICLASS', num_classes = 10),
    Recall('MULTICLASS', num_classes = 10),
    # F1(), torchmetrics.F1 cannot be imported
])


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    optimizer_type: str,
    lr: float = 1e-3,
    device: str = 'cuda:0',
    verbose=True
) -> Dict[str, torch.Tensor]:


    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    elif optimizer_type == "SGD":
        optimizer = torch.optim.SGD(lr=lr, params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()
    num_batch = len(train_dataloader)
    global metrics

    metrics = metrics.to(device)
    model.train(True)
    torch.set_grad_enabled(True)

    losses = []
    progress_bar = tqdm(enumerate(train_dataloader),
                        total=num_batch,
                        disable=not verbose,
                        )

    for batch_idx, batch in progress_bar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        model.zero_grad()

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        output = metrics(y_hat, y)

        progress_bar.set_postfix({'loss': loss.item(),
                                  'acc': output['MulticlassAccuracy'].item()})


    outputs = metrics.compute()
    metrics.reset()
    outputs = {k: [v.item()] for k, v in outputs.items()}
    torch.set_grad_enabled(False)
    outputs['Loss'] = [sum(losses) / len(losses)]
    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


@ torch.no_grad()
def test_by_data_set(
    model: nn.Module,
    data_loader: DataLoader,
    device='cuda:0',
    verbose=True
) -> Dict[str, torch.Tensor]:

    num_batch = len(data_loader)
    model.eval()
    global metrics

    metrics = metrics.to(device)
    progress_bar = tqdm(enumerate(data_loader),
                        total=num_batch,
                        file=sys.stdout,
                        disable=not verbose)
    for batch_idx, batch in progress_bar:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        output = metrics(y_hat, y)

        progress_bar.set_postfix({'acc': output['MulticlassAccuracy'].item()})


    outputs = metrics.compute()
    metrics.reset()
    model.train(True)
    outputs = {k: [v.item()] for k, v in outputs.items()}

    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


def get_pruned_amount_by_weights(model):
    if check_mask_object_from_model(model):
        sys.exit("\033[91m" + "Warning - get_pruned_amount_by_weights() is called when the model has mask." + "\033[0m")
    total_params_count = get_num_total_model_params(model)
    total_0_count = 0
    total_nan_count = 0
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                if weight_params.is_cuda:
                    total_0_count += len(list(zip(*np.where(weight_params.cpu() == 0))))
                    total_nan_count += len(torch.nonzero(torch.isnan(weight_params.cpu().view(-1))))
                else:
                    total_0_count += len(list(zip(*np.where(weight_params == 0))))
                    total_nan_count += len(torch.nonzero(torch.isnan(weight_params.view(-1))))
    if total_nan_count > 0:
        sys.exit("nan bug")
    return total_0_count / total_params_count

def get_pruned_amount(model):
    if check_mask_object_from_model(model):
        return get_pruned_amount_by_mask(model)
    return get_pruned_amount_by_weights(model)

def get_pruned_amount_by_mask(model):
    if not check_mask_object_from_model(model):
        sys.exit("\033[91m" + "Warning - mask object not found." + "\033[0m")
    total_params_count = get_num_total_model_params(model)
    total_0_count = 0
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                if mask.is_cuda:
                    total_0_count += len(list(zip(*np.where(mask.cpu() == 0))))
                else:
                    total_0_count += len(list(zip(*np.where(mask == 0))))
    return total_0_count / total_params_count

def sum_over_model_params(model):
    layer_to_model_sig_row = {}
    layer_to_model_sig_col = {}

    row_dim, col_dim = -1, -1
    for layer, module in model.named_children():
        if 'conv' in layer:
            row_dim, col_dim = 2, 3
        elif 'fc' in layer:
            row_dim, col_dim = 1, 0
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                if weight_params.is_cuda:
                    layer_to_model_sig_row[layer] = torch.sum(weight_params.cpu(), dim=row_dim)
                    layer_to_model_sig_col[layer] = torch.sum(weight_params.cpu(), dim=col_dim)
                else:
                    layer_to_model_sig_row[layer] = torch.sum(weight_params, dim=row_dim)
                    layer_to_model_sig_col[layer] = torch.sum(weight_params, dim=col_dim)

    return layer_to_model_sig_row, layer_to_model_sig_col


def get_num_total_model_params(model):
    total_num_model_params = 0
    # not including bias
    for layer_name, params in model.named_parameters():
        if 'weight' in layer_name:
            total_num_model_params += params.numel()
    return total_num_model_params    

def get_model_sig_sparsity(model, model_sig):
    total_num_model_params = get_num_total_model_params(model)
    total_num_sig_non_0_params = 0
    for layer, layer_sig in model_sig.items():
        if layer_sig.is_cuda:
            total_num_sig_non_0_params += len(list(zip(*np.where(layer_sig.cpu()!=0))))
        else:
            total_num_sig_non_0_params += len(list(zip(*np.where(layer_sig!=0))))
    return total_num_sig_non_0_params / total_num_model_params

def generate_mask_from_0_weights(model):
    params_to_prune = get_prune_params(model)
    for param, name in params_to_prune:
        weights = getattr(param, name)
        mask_amount = torch.eq(weights.data, 0.00).sum().item()
        prune.l1_unstructured(param, name, amount=mask_amount)
        
def make_prune_permanent(model):
    if check_mask_object_from_model(model):
        params_pruned = get_prune_params(model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
    return model

def check_mask_object_from_model(model):
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                return True
    return False


def get_trainable_model_weights(model):
    """
    Args:
        model (_torch model_): NN Model

    Returns:
        layer_to_param _dict_: you know!
    """
    layer_to_param = {} 
    for layer_name, param in model.named_parameters():
        if 'weight' in layer_name:
            layer_to_param[layer_name.split('.')[0]] = param.cpu().detach().numpy()
    return layer_to_param

def calc_mask_from_model_with_mask_object(model):
    layer_to_mask = {}
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                layer_to_mask[layer] = mask
    return layer_to_mask

def calc_mask_from_model_without_mask_object(model):
    layer_to_mask = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                layer_to_mask[layer] = np.ones_like(weight_params.cpu())
                layer_to_mask[layer][weight_params.cpu() == 0] = 0
    return layer_to_mask


def generate_2d_top_magnitude_mask(model_path, percent, check_whole = False, keep_sign = False):

    """
        returns 2d top magnitude mask.
        1. keep_sign == True
            it keeps the sign of the original weight. Used in introduce noise. 
            returns mask with -1, 1, 0.
        2. keep_sign == False
            calculate absolute magitude mask. Used in calculating weight overlapping.
            returns binary mask with 1, 0.
    """
    
    layer_to_mask = {}

    with open(model_path, 'rb') as f:
        nn_layer_to_weights = pickle.load(f)
            
    for layer, param in nn_layer_to_weights.items():
    
        # take abs as we show magnitude values
        abs_param = np.absolute(param)

        mask_2d = np.empty_like(abs_param)
        mask_2d[:] = 0 # initialize as 0

        base_size = abs_param.size if check_whole else abs_param.size - abs_param[abs_param == 0].size

        top_boundary = math.ceil(base_size * percent)
                    
        percent_threshold = -np.sort(-abs_param.flatten())[top_boundary]

        # change top weights to 1
        mask_2d[np.where(abs_param > percent_threshold)] = 1

        # sanity check
        # one_counts = (mask_2d == 1).sum()
        # print(one_counts/param.size)

        layer_to_mask[layer] = mask_2d
        if keep_sign:
            layer_to_mask[layer] *= np.sign(param)

    # sanity check
    # for layer in layer_to_mask:
	#     print((layer_to_mask[layer] == 1).sum()/layer_to_mask[layer].size)

    return layer_to_mask

def calculate_overlapping_mask(model_paths, check_whole, percent, model_validation=False):
    layer_to_masks = []

    for model_path in model_paths:
        layer_to_masks.append(generate_2d_top_magnitude_mask(model_path, percent, check_whole))

    ref_layer_to_mask = layer_to_masks[0]

    for layer_to_mask_iter in range(len(layer_to_masks[1:])):
        layer_to_mask = layer_to_masks[1:][layer_to_mask_iter]
        for layer, mask in layer_to_mask.items():
            ref_layer_to_mask[layer] *= mask
            if check_whole:
                # for debug - when each local model has high overlapping with the last global model, why the overlapping ratio for all local models seems to be low?
                if model_validation: # called by model_validation()
                    print(f"Worker {model_paths[-1].split('/')[-2]}, layer {layer} - overlapping ratio on top {percent:.2%} is {(ref_layer_to_mask[layer] == 1).sum()/ref_layer_to_mask[layer].size/percent:.2%}")
                else:
                    print(f"iter {layer_to_mask_iter + 1}, layer {layer} - overlapping ratio on top {percent:.2%} is {(ref_layer_to_mask[layer] == 1).sum()/ref_layer_to_mask[layer].size/percent:.2%}")
        print()

    return ref_layer_to_mask