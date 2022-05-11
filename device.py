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
from util import get_prune_summary, get_prune_amount_by_0_weights, l1_prune, get_prune_params, copy_model, fed_avg
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
        self._mask = {}
        self._train_loader = train_loader
        self._test_loader = test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.model_sig = None
        # blockchain variables
        self.role = None
        self.device_dict = None
        self.peers = None
        self.stake_book = None
        self.blockchain = Blockchain()
        self.is_online = False
        # for lotters
        self._lotter_transaction = None
        # for validators
        self.associated_lotters = set()
        self.verified_transactions = set()
        self.lotter_to_vote = {}
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
    
    ######## lotters method ########
    def warm_initial_mask(self):
        # only do it once at the begining of joining
        # 1. train; 2. prune;
        self.train()
        self.prune()
    
    def regular_ticket_learning(self):
        # only do it once at the begining of joining
        # 1. prune; 2. reinit; 3. train;
        self.prune()
        self.reinit_model_params()
        self.train()

    
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
        if self._mask:
            # warming mask
            curr_model_prune_amount = get_prune_amount_by_0_weights(model=self.model)
            prune_amount = curr_model_prune_amount + self.args.prune_diff
        else:
            # apply local mask to global model weights
            for layer, module in self.model.named_children():
                for name, weight_params in module.named_parameters():
                    if 'weight' in name:
                        weight_params.data.copy_(torch.tensor(np.multiply(weight_params.data, self._mask[layer])))
            # use prune to "produce" current mask
            masked_prune_amount = get_prune_amount_by_0_weights(model=self.model)
            l1_prune(model=self.model,
                amount=masked_prune_amount,
                name='weight',
                verbose=False)
            curr_prune_diff = self.blockchain.get_cur_pruning_diff()
            prune_amount = curr_prune_diff - masked_prune_amount
        
        # real prune
        l1_prune(model=self.model,
                amount=prune_amount,
                name='weight',
                verbose=self.args.prune_verbose)
        # update local mask
        for layer, module in self.model.named_children():
                for name, mask in module.named_buffers():
                    if 'mask' in name:
                        self._mask[layer] = mask
                        
    def reinit_model_params(self):
        source_params = dict(self.init_global_model.named_parameters())
        for name, param in self.model.named_parameters():
            param.data.copy_(source_params[name].data)
            
    def create_model_sig(self):
        
        # find all 1s in mask layer by layer
        def populate_layer_dict():
            layer_to_num_picked_ones = {}
            for layer, module in self.model.named_children():
                for name, mask in module.named_buffers():
                    if 'mask' in name:
                        layer_to_num_picked_ones[layer] = 0
            return layer_to_num_picked_ones
        
        def get_ones_postions():
            total_num_ones = 0
            layer_to_one_positions = {}
            layer_to_num_ones = {}
            for layer, module in self.model.named_children():
                for name, mask in module.named_buffers():
                    if 'mask' in name:
                        # get all 1 positions
                        # https://stackoverflow.com/questions/8081545/how-to-convert-list-of-tuples-to-multiple-lists
                        one_positions = list(zip(*np.where(mask==1)))
                        layer_to_one_positions[layer] = one_positions
                        layer_to_num_ones[layer] = len(one_positions)
                        total_num_ones += len(one_positions)
            return total_num_ones, layer_to_one_positions, layer_to_num_ones
        
        # determine how many ones to pick for signature layer by layer, randomly
        def nums_elems_to_pick_from_layers(total_num_ones, layer_to_num_ones, layer_to_num_picked_ones):
            while total_num_ones > 0:
                picked_layer, num_ones = random.choice(list(layer_to_num_ones.items()))
                if num_ones == 0:
                    continue
                num_ones_to_pick = min(total_num_ones, random.randint(1, num_ones))
                layer_to_num_picked_ones[picked_layer] += num_ones_to_pick
                layer_to_num_ones[picked_layer] -= num_ones_to_pick
                if layer_to_num_ones[picked_layer] == 0:
                    del layer_to_num_ones[picked_layer]
                total_num_ones -= num_ones_to_pick
            return layer_to_num_picked_ones
        
        # randomly pick ones by the numbers calculated in nums_elems_to_pick_from_layers()
        def pick_elems_positions_from_layers(layer_to_one_positions, layer_to_num_picked_ones):
            layer_to_picked_ones_positions = {}
            for layer, num_picks in layer_to_num_picked_ones.items():
                layer_to_picked_ones_positions[layer] = random.sample(layer_to_one_positions[layer], num_picks)
            return layer_to_picked_ones_positions
        
        def create_sig_mask_with_disturbs(layer_to_picked_ones_positions, layer_to_picked_distrubs_positions):
            layer_to_keeped_ones_positions = {layer: set(layer_to_picked_ones_positions[layer]) - set(layer_to_picked_distrubs_positions[layer]) for layer in layer_to_picked_ones_positions}
            # create the signature mask
            sig_mask = populate_layer_dict()
            for layer, module in self.model.named_children():
                for name, mask in module.named_buffers():
                    if 'mask' in name:
                        sig_mask_layer = np.zeros_like(mask)
                        for position in layer_to_keeped_ones_positions[layer]:
                            sig_mask_layer[position] = 1
                        for position in layer_to_picked_distrubs_positions[layer]:
                            sig_mask_layer[position] = random.uniform(0,2)
                    sig_mask[layer] = sig_mask_layer
            return sig_mask
        
        def create_signature(sig_mask_with_disturbs):
            layer_to_sig = {}
            for layer, module in self.model.named_children():
                for name, weight_params in module.named_parameters():
                    if 'weight' in name:
                        layer_to_sig[layer] = np.multiply(weight_params, sig_mask_with_disturbs[layer])
            # print(sum([len(list(zip(*np.where(layer_to_sig[layer] != 0)))) for layer in layer_to_sig]))
            return layer_to_sig
        
        # pick the ones for signature
        layer_to_num_picked_ones = populate_layer_dict()
        total_num_ones, layer_to_one_positions, layer_to_num_ones = get_ones_postions()
        total_num_ones = int(total_num_ones * self.args.sig_portion)
        layer_to_num_picked_ones = nums_elems_to_pick_from_layers(total_num_ones, layer_to_num_ones, layer_to_num_picked_ones)
        layer_to_picked_ones_positions = pick_elems_positions_from_layers(layer_to_one_positions, layer_to_num_picked_ones)
        
        # disturb sig_threshold portion of weights
        layer_to_num_picked_disturbs = populate_layer_dict()
        total_num_disturbs = int(total_num_ones * (1 - self.args.sig_threshold))
        layer_to_num_picked_disturbs = nums_elems_to_pick_from_layers(total_num_disturbs, layer_to_num_picked_ones, layer_to_num_picked_disturbs)
        layer_to_picked_distrubs_positions = pick_elems_positions_from_layers(layer_to_picked_ones_positions, layer_to_num_picked_disturbs)
        sig_mask = create_sig_mask_with_disturbs(layer_to_picked_ones_positions, layer_to_picked_distrubs_positions)
        model_sig = create_signature(sig_mask)
        
        self.model_sig = model_sig
        
        return model_sig
    
    def make_transaction(self):
        lotter_transaction = {
            'idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'model' : self.model,
            'model_signature' : self.model_sig
        }
        lotter_transaction['tx_signature'] = self.sign_msg(sorted(lotter_transaction.items()))
        self._lotter_transaction = lotter_transaction
    
    def asso_validators(self, validators):
        num_validators_to_send = len(validators) * self.args.validator_portion
        validators = random.shuffle(validators)
        for validator in validators:
            if validator.is_online:
                validator.associate_with_lotter(self)
                num_validators_to_send -= 1
                if num_validators_to_send == 0:
                    break
                
    ### Validators ###
    def associate_with_lotter(self, lotter):
        self.associated_lotters.add(lotter)
        
    def verify_lotter_tx(self):
        for lotter in self.associated_lotters:
            tx_before_signed = copy.deepcopy(lotter._lotter_transaction)
            del tx_before_signed["tx_signature"]
            modulus = lotter._lotter_transaction['rsa_pub_key']["modulus"]
            pub_key = lotter._lotter_transaction['rsa_pub_key']["pub_key"]
            signature = lotter._lotter_transaction["rsa_pub_key"]
            # verify
            hash = int.from_bytes(sha256(str(sorted(tx_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
            hashFromSignature = pow(signature, pub_key, modulus)
            if hash == hashFromSignature:
                self.verified_transactions.add(lotter._lotter_transaction)
            else:
                print(f"Signature of tx from lotter {lotter._lotter_transaction['idx']} is invalid.")
                # TODO - record to black list
                
    def verify_model_sig_positions(self):    
        def verify_model_sig_positions_by_layer(verified_tx):
            lotter_idx = verified_tx['idx']
            model = verified_tx['model']
            model_signature = verified_tx['model_signature']
            for layer, module in model.named_children():
                for name, weight_params in module.named_parameters():
                    if 'weight' in name:
                        unpruned_positions = list(zip(*np.where(weight_params[layer] != 0)))
                        model_sig_positions = list(zip(*np.where(model_signature[layer] != 0)))
                        for pos in model_sig_positions:
                            if pos not in unpruned_positions:
                                print(f"Model signature from lotter {lotter_idx} is invalid.")
                                return False
            return True
        passed_model_sig_tx = []
        for verified_tx in self.verified_transactions:
            if verify_model_sig_positions_by_layer(verified_tx):                    
                passed_model_sig_tx.append(verified_tx)
        self.verified_transactions = passed_model_sig_tx

    def validate_model_accuracy(self):
        
        def avg_individual_model(models_list, by_how_many):
            new_models_list = deepcopy(models_list)
            for idx, model in new_models_list.items():
                for name, param in model.named_parameters():
                    param.data.copy_(torch.mul(param, 1/by_how_many))
            return new_models_list        
            
        
        models_list = {}
        for verified_tx in self.verified_transactions:
            models_list[verified_tx['idx']] = verified_tx['model']
        base_aggr_model = fed_avg(models_list, self.args.dev_device)
        model_divides = avg_individual_model(models_list, len(models_list) - 1)
        
        base_aggr_model_acc = 
                
        
                     
        
    ### General ###
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}
    
    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature
    
        
        
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
