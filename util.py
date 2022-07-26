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



def produce_mask_from_model(model):
    # use prune with 0 amount to init mask for the model
    # create mask in-place on model
    l1_prune(model=model,
                amount=0.00,
                name='weight',
                verbose=False)
    layer_to_masked_positions = {}
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                layer_to_masked_positions[layer] = list(zip(*np.where(weight_params.cpu() == 0)))
        
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                for pos in layer_to_masked_positions[layer]:
                    mask[pos] = 0
                    
@torch.no_grad()
def fedavg(models, device):
    aggr_model = models[0].__class__().to(device)
    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for model in models:
            weighted_param = torch.mul(
                dict(model.named_parameters())[name].data, 1/len(models))
            param.data.copy_(param.data + weighted_param)
    return aggr_model    

def fedavg_lotteryfl(models, device):
    # print("Using LotteryFL aggregation style.")
    aggr_model = models[0].__class__().to(device)
    # sum all masks values
    layer_to_mask_sum = {}
    for model in models:
        produce_mask_from_model(model)
        for layer, module in model.named_children():
            for name, mask in module.named_buffers():
                if 'mask' in name:
                    layer_to_mask_sum[layer] = layer_to_mask_sum.get(layer, torch.zeros_like(mask)) + mask
        # revoke mask
        pytorch_make_prune_permanent(model)
    # sum all weight values
    layer_to_weight_sum = {}
    for model in models:
        for name, param in model.named_parameters():
            layer_to_weight_sum[name.split('_')[0]] = layer_to_weight_sum.get(name, torch.zeros_like(param)) + param
    # divide
    for name, param in aggr_model.named_parameters():
        if 'weight' in name:
            # divide by 0 becomes nan, need to replace by 0
            param.data.copy_(torch.nan_to_num(layer_to_weight_sum[name] / layer_to_mask_sum[name.split('.')[0]]))
        else:
            param.data.copy_(layer_to_weight_sum[name])
    return aggr_model

def apply_local_mask(model, mask):
    # apply mask in-place to model
    # direct multiplying instead of adding mask object
    if not mask:
        return
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                weight_params.data.copy_(torch.tensor(np.multiply(weight_params.cpu().data, mask[layer])))


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
    new_model = create_model_no_prune(model.__class__, device)
    source_params = dict(model.named_parameters())
    source_buffer = dict(model.named_buffers())
    for name, param in new_model.named_parameters():
        param.data.copy_(source_params[name].data)
    for name, buffer_ in new_model.named_buffers():
        buffer_.data.copy_(source_buffer[name].data)
    return new_model


metrics = MetricCollection([
    Accuracy(),
    Precision(),
    Recall(),
    # F1(), torchmetrics.F1 cannot be imported
])


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    lr: float = 1e-3,
    device: str = 'cuda:0',
    verbose=True
) -> Dict[str, torch.Tensor]:

    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
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
                                  'acc': output['Accuracy'].item()})


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

        progress_bar.set_postfix({'acc': output['Accuracy'].item()})


    outputs = metrics.compute()
    metrics.reset()
    model.train(True)
    outputs = {k: [v.item()] for k, v in outputs.items()}

    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


def get_pruned_amount_by_weights(model):
    total_params_count = get_num_total_model_params(model)
    total_0_count = 0
    total_nan_count = 0
    for layer, module in model.named_children():
        for name, weight_params in module.named_parameters():
            if 'weight' in name:
                total_0_count += len(list(zip(*np.where(weight_params.cpu() == 0))))
                total_nan_count += len(torch.nonzero(torch.isnan(weight_params.view(-1))))
    if total_nan_count > 0:
        sys.exit("nan bug")
    return total_0_count / total_params_count

def get_pruned_amount_by_mask(model):
    total_params_count = get_num_total_model_params(model)
    total_0_count = 0
    for layer, module in model.named_children():
        for name, mask in module.named_buffers():
            if 'mask' in name:
                total_0_count += len(list(zip(*np.where(mask.cpu() == 0))))
    return total_0_count / total_params_count

    

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
        total_num_sig_non_0_params += len(list(zip(*np.where(layer_sig!=0))))
    return total_num_sig_non_0_params / total_num_model_params

def generate_mask_from_0_weights(model):
    params_to_prune = get_prune_params(model)
    for param, name in params_to_prune:
        weights = getattr(param, name)
        mask_amount = torch.eq(weights.data, 0.00).sum().item()
        prune.l1_unstructured(param, name, amount=mask_amount)
        
def pytorch_make_prune_permanent(model):
    params_pruned = get_prune_params(model, name='weight')
    try:
        for param, name in params_pruned:
            prune.remove(param, name)
    except:
        # if an unpruned model is passed into this function, ignore
        pass
    return model



def custom_save(model, path):
    """
    https://pytorch.org/docs/stable/generated/torch.save.html#torch.save
    Custom save utility function
    Compresses the model using gzip
    Helpfull if model is highly pruned
    """
    bufferIn = io.BytesIO()
    torch.save(model.state_dict(), bufferIn)
    bufferOut = gzip.compress(bufferIn.getvalue())
    with gzip.open(path, 'wb') as f:
        f.write(bufferOut)


def custom_load(path) -> Dict:
    """
    returns saved_dictionary
    """
    with gzip.open(path, 'rb') as f:
        bufferIn = f.read()
        bufferOut = gzip.decompress(bufferIn)
        state_dict = torch.load(io.BytesIO(bufferOut))
    return state_dict


def log_obj(path, obj):
    # pass
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    #
    with open(path, 'wb') as file:
        if isinstance(obj, nn.Module):
            torch.save(obj, file)
        else:
            pickle.dump(obj, file)


class CustomPruneMethod(prune.BasePruningMethod):

    PRUNING_TYPE = 'unstructured'

    def __init__(self, amount, orig_weights):
        super().__init__()
        self.amount = amount
        self.original_signs = self.get_signs_from_tensor(orig_weights)

    def get_signs_from_tensor(self, t: torch.Tensor):
        return torch.sign(t).view(-1)

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        large_weight_mask = t.view(-1).mul(self.original_signs)
        large_weight_mask_ranked = F.relu(large_weight_mask)
        nparams_toprune = int(torch.numel(t) * self.amount)  # get this val
        if nparams_toprune > 0:
            bottom_k = torch.topk(
                large_weight_mask_ranked.view(-1), k=nparams_toprune, largest=False)
            mask.view(-1)[bottom_k.indices] = 0.00
            return mask
        else:
            return mask


def customPrune(module, orig_module, amount=0.1, name='weight'):
    """
        Taken from https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
        Takes: current module (module), name of the parameter to prune (name)

    """
    CustomPruneMethod.apply(module, name, amount, orig_module)
    return module

