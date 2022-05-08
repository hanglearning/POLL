import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from typing import Dict
import copy
import math
import wandb
from torch.nn.utils import prune
from util import get_prune_summary, get_prune_amount_by_0_weights, l1_prune, get_prune_params, copy_model, extract_mask, gen_qr, get_qr_max_sim_position, embed_qr_into_mask
from util import train as util_train
from util import test as util_test

from copy import copy, deepcopy
from Crypto.PublicKey import RSA
from hashlib import sha256
from Blockchain import Blockchain
import random
import string

# used for signature embedding
letters = string.ascii_lowercase

class Device():
    def __init__(
        self,
        idx,
        args,
        train_loader,
        test_loader,
        init_global_model,
    ):
        self.idx = idx
        self.args = args
        # ticket learning variables
        self._train_loader = train_loader
        self._test_loader = test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.mask_original = None
        self.mask_embedded = None
        # blockchain variables
        self.role = None
        self.device_dict = None
        self.peers = None
        self.stake_book = None
        self.blockchain = Blockchain()
        self.is_online = False
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()
        
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
        
    def assign_peers(self, idx_to_device):
        self.device_dict = idx_to_device
        self.peers = set(idx_to_device.values())
        self.stake_book = {key: [] for key in idx_to_device.keys()}
  
    def set_is_online(self):
        self.is_online = random.random() <= self.args.network_stability
        
    def is_online(self):
        return self.is_online
    
    def resync_chain(self):
        if self.args.resync_verbose:
            print(f"{self.role} {self.idx} is looking for a chain with the highest accumulated stake in the network...")
        top_stake_holders = []
        for peer in self.peers:
            top_stake_holders.append(max(peer.stake_book, key=peer.stake_book.get))
        final_top_stake_holder = self.device_dict[max(set(top_stake_holders), key=top_stake_holders.count)]
        # compare chain difference
        if self.blockchain.get_last_block_hash() == final_top_stake_holder.blockchain.get_last_block_hash():
            if self.args.resync_verbose:
                print(f"{self.role} {self.idx}'s chain not resynced.")
                return False
        else:
            self.blockchain.replace_chain(final_top_stake_holder.blockchain)
            print(f"{self.role} {self.idx}'s chain resynced chain from {final_top_stake_holder.idx}.")
            #TODO - update global model
            return True
    
    def warm_initial_mask(self, model):
        # only do it once at the begining of joining
        # 1. train; 2. prune; 3. embed; 4.prune
        self.train()
        self.prune()
        self.embed_sig()
        self.prune_after_embed()

    
    def train(self):
        """
            Train local model
        """
        losses = []

        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Device={self.idx}, epoch={epoch}, comm_round:{self.blockchain.get_chain_length()+1}")

            metrics = util_train(self.model,
                                 self.train_loader,
                                 self.args.lr,
                                 self.args.dev_device,
                                 self.args.fast_dev_run,
                                 self.args.train_verbose)
            losses.append(metrics['Loss'][0])

            if self.args.fast_dev_run:
                break
        self.losses.extend(losses)
    
    def prune(self):
        curr_model_prune_amount = get_prune_amount_by_0_weights(model=self.model)
        curr_prune_diff = self.blockchain.get_cur_pruning_diff()
        prune_amount = curr_prune_diff - curr_model_prune_amount
        if not self.mask_embedded and prune_amount < self.args.prune_diff:
            # warming mask
            prune_amount = curr_prune_diff + self.args.prune_diff
        l1_prune(model=self.model,
                amount=prune_amount,
                name='weight',
                verbose=self.args.prune_verbose)
        
    def reinit_model_params(self):
        source_params = dict(self.global_init_model.named_parameters())
        for name, param in self.model.named_parameters():
            param.data.copy_(source_params[name].data)
            
    def embed_sig(self):
        # extract mask
        mask = extract_mask(self.model)
        # generate signature
        sig = f"{self.id}_{''.join(random.choice(letters) for i in range(10))}"
        # generate QR Code
        qr_code_array = gen_qr(sig)
        # find max similarity in mask (single layer)
        max_layer_name, start_x, start_y, h, w = get_qr_max_sim_position(qr_code_array, mask)
        # embed qr code into mask
        mask_with_embedding = embed_qr_into_mask(mask, max_layer_name, qr_code_array, start_x, start_y, h, w)
        # apply mask to the model
        
        pass
        
        
    def update(self) -> None:
        """
            Interface to Server
        """
        print(f"\n----------Device:{self.idx} Update---------------------")

        print(f"Evaluating Global model ")
        metrics = self.eval(self.model)
        accuracy = metrics['Accuracy'][0]
        print(f'Global model accuracy: {accuracy}')

        prune_rate = get_prune_summary(model=self.model,
                                       name='weight')['global']
        print('Global model prune percentage: {}'.format(prune_rate))
           
        if self.cur_prune_rate < self.args.target_spar:
            if accuracy > self.eita:
                self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                          self.args.target_spar)
                if self.cur_prune_rate > prune_rate:
                    l1_prune(model=self.model,
                             amount=self.cur_prune_rate - prune_rate,
                             name='weight',
                             verbose=self.args.prune_verbose)
                    self.prune_rates.append(self.cur_prune_rate)
                else:
                    self.prune_rates.append(prune_rate)
                # reinitialize model with init_params
                source_params = dict(self.init_global_model.named_parameters())
                for name, param in self.model.named_parameters():
                    param.data.copy_(source_params[name].data)

                self.model = self.model
                self.eita = self.eita_hat

            else:
                self.eita *= self.alpha
                self.model = self.model
                self.prune_rates.append(prune_rate)
        else:
            if self.cur_prune_rate > prune_rate:
                l1_prune(model=self.model,
                         amount=self.cur_prune_rate-prune_rate,
                         name='weight',
                         verbose=self.args.prune_verbose)
                self.prune_rates.append(self.cur_prune_rate)
            else:
                self.prune_rates.append(self.prune_rates)
            self.model = self.model

        print(f"\nTraining local model")
        self.train(self.elapsed_comm_rounds)

        print(f"\nEvaluating Trained Model")
        metrics = self.eval(self.model)
        print(f'Trained model accuracy: {metrics["Accuracy"][0]}')

        wandb.log({f"{self.idx}_cur_prune_rate": self.cur_prune_rate})
        wandb.log({f"{self.idx}_eita": self.eita})
        wandb.log(
            {f"{self.idx}_percent_pruned": self.prune_rates[-1]})

        for key, thing in metrics.items():
            if(isinstance(thing, list)):
                wandb.log({f"{self.idx}_{key}": thing[0]})
            else:
                wandb.log({f"{self.idx}_{key}": thing})

        if (self.elapsed_comm_rounds+1) % self.args.save_freq == 0:
            self.save(self.model)

        self.elapsed_comm_rounds += 1


    @torch.no_grad()
    def download(self, global_model, init_global_model, *args, **kwargs):
        """
            Download global model from server
        """
        self.model = global_model
        self.init_global_model = init_global_model

        params_to_prune = get_prune_params(self.model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

        params_to_prune = get_prune_params(self.init_global_model)
        for param, name in params_to_prune:
            weights = getattr(param, name)
            masked = torch.eq(weights.data, 0.00).sum().item()
            # masked = 0.00
            prune.l1_unstructured(param, name, amount=int(masked))

    def eval(self, model):
        """
            Eval self.model
        """
        eval_score = util_test(model,
                               self.test_loader,
                               self.args.dev_device,
                               self.args.fast_dev_run,
                               self.args.test_verbose)
        self.accuracies.append(eval_score['Accuracy'][0])
        return eval_score

    def save(self, *args, **kwargs):
        pass

    def upload(self, *args, **kwargs) -> Dict[nn.Module, float]:
        """
            Upload self.model
        """
        upload_model = copy_model(model=self.model, device=self.args.dev_device)
        params_pruned = get_prune_params(upload_model, name='weight')
        for param, name in params_pruned:
            prune.remove(param, name)
        return {
            'model': upload_model,
            'acc': self.accuracies[-1]
        }
