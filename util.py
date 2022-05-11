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
import qrcode


@torch.no_grad()
def fed_avg(models, device):
    aggr_model = models[0].__class__().to(device)
    for name, param in aggr_model.named_parameters():
        param.data.copy_(torch.zeros_like(param.data))
        for model in models:
            weighted_param = torch.mul(
                dict(model.named_parameters())[name].data)
            param.data.copy_(param.data + weighted_param)
    return aggr_model


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
        Note: the model should have been pruned for this method to work to create buffer masks and what not.
    """
    new_model = create_model(model.__class__, device)
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
    fast_dev_run=False,
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
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    outputs = {k: [v.item()] for k, v in outputs.items()}
    torch.set_grad_enabled(False)
    outputs['Loss'] = [sum(losses) / len(losses)]
    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


@ torch.no_grad()
def test(
    model: nn.Module,
    test_dataloader: DataLoader,
    device='cuda:0',
    fast_dev_run=False,
    verbose=True,
) -> Dict[str, torch.Tensor]:

    num_batch = len(test_dataloader)
    model.eval()
    global metrics

    metrics = metrics.to(device)
    progress_bar = tqdm(enumerate(test_dataloader),
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
        if fast_dev_run:
            break

    outputs = metrics.compute()
    metrics.reset()
    model.train(True)
    outputs = {k: [v.item()] for k, v in outputs.items()}

    if verbose:
        print(tabulate(outputs, headers='keys', tablefmt='github'))
    return outputs


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
        print("Total Pruning: {}%".format(global_pruning))


"""
Hadamard Mult of Mask and Attributes,
then return zeros
"""


@ torch.no_grad()
def summarize_prune(model: nn.Module, name: str = 'weight') -> tuple:
    """
        returns (pruned_params,total_params)
    """
    num_pruned = 0
    params, num_global_weights, _ = get_prune_params(model)
    for param, _ in params:
        if hasattr(param, name+'_mask'):
            data = getattr(param, name+'_mask')
            num_pruned += int(torch.sum(data == 0.0).item())
    return (num_pruned, num_global_weights)


def get_prune_params(model, name='weight') -> List[Tuple[nn.Parameter, str]]:
    # iterate over network layers
    params_to_prune = []
    for _, module in model.named_children():
        for name_, param in module.named_parameters():
            if name in name_:
                params_to_prune.append((module, name))
    return params_to_prune

def get_prune_amount_by_0_weights_by_0_weights(model):
    num_zeros, num_weights = 0, 0
    params_pruned = get_prune_params(model, 'weight')
    for layer, weight_name in params_pruned:

        num_layer_zeros = torch.sum(
            getattr(layer, weight_name) == 0.0).item()
        num_zeros += num_layer_zeros
        num_layer_weights = torch.numel(getattr(layer, weight_name))
        num_weights += num_layer_weights
    
    return num_zeros / num_weights

def generate_mask_from_0_weights(model):
    params_to_prune = get_prune_params(model)
    for param, name in params_to_prune:
        weights = getattr(param, name)
        mask_amount = torch.eq(weights.data, 0.00).sum().item()
        prune.l1_unstructured(param, name, amount=mask_amount)
        
def extract_mask(model):
    # either use NO-stealing -> chosen
    # or use module.weight_mask
    mask_dict = {}
    for key in model.state_dict().keys():
        if 'mask' in key:
            mask_dict[key] = model.state_dict()[key]
    return mask_dict

def gen_qr(sig):
    
    qr = qrcode.QRCode(
    version=3,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=1,
    border=0,
    )
    qr.add_data(sig)
    qr.make()

    img = qr.make_image(fill_color="black", back_color="white")
    qr_code_array = np.array(img)
    return qr_code_array

def get_qr_max_sim_position(code, mask):
    h,w = code.shape[0],code.shape[1]
    max_sim = 0
    for layer_name in mask:
        mask_ = mask[layer_name].sum((2,3)).numpy() > 0
        mask_ = mask_.astype(float)
        if (mask_.shape[0] - code.shape[0] < 0) or (mask_.shape[1] - code.shape[1] < 0):
            continue
        sim = np.zeros((mask_.shape[0] - code.shape[0] + 1, mask_.shape[1] - code.shape[1] + 1))
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                sim[i,j] = (mask_[i:i+h,j:j+w] == code).mean()

        if np.max(sim) > max_sim:
            max_layer_name = layer_name
            max_sim = np.max(sim)
    mask_ = mask[max_layer_name].sum((2,3)).numpy() > 0
    mask_ = mask_.astype(float)
    sim = np.zeros((mask_.shape[0] - code.shape[0] + 1, mask_.shape[1] - code.shape[1] + 1))
    for i in range(sim.shape[0]):
        for j in range(sim.shape[1]):
            sim[i,j] = (mask_[i:i+h,j:j+w] == code).mean()
    r, c = np.where(sim == np.max(sim))
    start_x = r[0]
    start_y = c[0]
    return max_layer_name, start_x, start_y, h, w

def embed_qr_into_mask(mask, max_name, code, r, c, h, w):
    real_mask = mask[max_name].numpy()[r:r+h, c:c+w].copy()
    real_mask_one = (real_mask == 1).sum()
    real_mask_flat = ((real_mask).sum((2,3)) > 0).astype(float)
    
    for i in range(code.shape[0]):
        for j in range(code.shape[1]):
            if code[i,j] == 1 and real_mask_flat[i,j] == 0:
                _ = np.array([0] * 9)
                _[0] = 1
                new_mask = np.random.permutation(_)
                real_mask[i,j] = new_mask.reshape((3, 3))
                real_mask_flat[i,j] == 1
            elif code[i,j] == 0 and real_mask_flat[i,j] == 1:
                real_mask[i,j] = 0
                real_mask_flat[i,j] == 0

    original_mask = mask[max_name][r:r+h, c:c+w].clone().numpy()
    real_mask[0:9, 0:9] = original_mask[0:9, 0:9]
    real_mask[-9:,:9] = original_mask[-9:,:9]
    real_mask[:9,-9:] = original_mask[:9,-9:]
    real_mask[20:25, 20:25] = original_mask[20:25, 20:25]
    real_mask[-8, 4 * 3 + 9] = 1
    real_mask[6] = original_mask[6]
    real_mask[:, 6] = original_mask[:, 6]

    real_mask_one_new = (real_mask == 1).sum()
    real_mask_flat_new = (real_mask).sum((2,3))
    diff = real_mask_one_new - real_mask_one
    # print(diff)

    if (diff > 0):
        # remove some connections
        real_mask_flat_greater_0 = np.where(real_mask_flat_new > 1)
    else:
        # recover some connections
        pos = np.expand_dims((code == 1), (2, 3)) * np.expand_dims(real_mask_flat == 1, (2,3)) * (real_mask == 0)
        pos = np.where(pos)
        pos = np.stack(pos)
        # print(pos.shape)
        pos = pos[:, np.random.permutation(pos.shape[1])[:(-diff)]]
        # print(pos.shape)
        for i in range(pos.shape[1]):
            p = pos[:, i]
            real_mask[p[0], p[1], p[2], p[3]] = 1
    mask[max_name][r:r+h, c:c+w] = torch.from_numpy(real_mask)
    return mask

def get_prune_amount_by_mask():
    pass

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

