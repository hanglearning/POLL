# TODO - complete validate_chain(), check check_chain_validity() on VBFL

# TODO - tx and msg sig check, finish after 5/19

from matplotlib.pyplot import contour
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import numpy as np
import os
from typing import Dict
import math
import wandb
from torch.nn.utils import prune
from util import *
# from util import get_prune_summary, get_pruned_amount_by_weights, l1_prune, get_prune_params, copy_model, fedavg, fedavg_workeryfl, test_by_data_set, AddGaussianNoise, get_model_sig_sparsity, get_num_total_model_params, get_pruned_amount_from_mask, produce_mask_from_model, apply_local_mask, pytorch_make_prune_permanent
from util import train as util_train
from util import test_by_data_set

import pandas as pd
pd.set_option('display.max_columns', None)

from copy import copy, deepcopy
from Crypto.PublicKey import RSA
from hashlib import sha256
from Block import Block
from Blockchain import Blockchain
import random
import string
import collections 
from statistics import mean

# used for signature embedding
letters = string.ascii_lowercase

class Device():
    def __init__(
        self,
        idx,
        is_malicious,
        args,
        train_loader,
        test_loader,
        user_labels,
        global_test_loader,
        init_global_model,
    ):
        self.idx = idx
        self.args = args
        # ticket learning variables
        self._reinit = False
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._user_labels = user_labels
        self.global_test_loader = global_test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.model_sig = None
        # blockchain variables
        self.role = None
        self._is_malicious = is_malicious
        self.has_appended_block = False
        # self.device_dict = None
        self.peers = None
        self.stake_book = None
        self.blockchain = Blockchain(args.diff_base, args.diff_incre, args.diff_freq, args.target_spar)
        self._received_blocks = []
        self._black_list = {}
        self._resync_to = None
        # for workers
        self._worker_tx = None
        # for validators
        self._associated_workers = set()
        self._associated_validators = set()
        self._validator_txs = []
        self._verified_worker_txs = {}
        self._received_validator_txs = {} # worker_id_to_corresponding_validator_txes
        self._verified_validator_txs = set()
        self._final_ticket_model = None
        self._used_worker_txes = {} # models used in final ticket model
        self._unused_worker_txes = {} # models NOT used in final ticket model
        self._dup_used_worker_txes = {}
        self.produced_block = None
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
        # self.device_dict = idx_to_device
        self.peers = set(idx_to_device.values())
        self.peers.remove(self)
        self.stake_book = {key: 0 for key in idx_to_device.keys()}
  
    def is_online(self):
        return random.random() <= self.args.network_stability
    
    def resync_chain(self, comm_round, idx_to_device, online_devices_list, online_validators):
        if comm_round == 1:
            return False
        if self._resync_to:
            # _resync_to specified to the last round's picked winning validator
            resync_to_device = idx_to_device[self._resync_to]
            if self.validate_chain(resync_to_device.blockchain) and self._resync_to in [d.idx for d in online_devices_list]:
                # update chain
                self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                print(f"\n{self.role} {self.idx}'s chain is resynced from last round's picked winning validator {self._resync_to}.")
                # update stake_book
                self.stake_book = resync_to_device.stake_book
                self._resync_to = None
                return True                
        online_devices_list = copy(online_devices_list)
        if self.stake_book:
            # resync chain from online validators having the highest recorded stake
            self.stake_book = {validator: stake for validator, stake in sorted(self.stake_book.items(), key=lambda x: x[1], reverse=True)}
            for device_idx, stake in self.stake_book.items():
                device = idx_to_device[device_idx]
                if device.role != "validator":
                    continue
                if device in online_devices_list:
                    # compare chain difference
                    if self.blockchain.get_last_block_hash() == device.blockchain.get_last_block_hash():
                        return
                    else:
                        # validate chain
                        if not self.validate_chain(device.blockchain):
                            continue
                        # update chain
                        self.blockchain.replace_chain(device.blockchain.chain)
                        print(f"\n{self.role} {self.idx}'s chain is resynced from {device.idx}, who picked {idx_to_device[device.idx].blockchain.get_last_block().produced_by}'s block.")
                        # update stake_book
                        self.stake_book = device.stake_book
                        return True
        else:
            # sync chain by randomly picking a device when join in the middle (or got offline in the first comm round)
            while online_devices_list:
                picked_device = random.choice(online_devices_list)
                # validate chain
                if not self.validate_chain(picked_device.blockchain):
                    online_devices_list.remove(picked_device)
                    continue
                self.blockchain.replace_chain(picked_device.blockchain)
                self.stake_book = picked_device.stake_book
                print(f"{self.role} {self.idx}'s chain resynced chain from {device.idx}.")
                return True
        if self.args.resync_verbose:
            print(f"{self.role} {self.idx}'s chain not resynced.")
        return False
            
    def post_resync(self):
        # update global model from the new block
        self.model = copy_model(self.blockchain.get_last_block().global_ticket_model, self.args.dev_device)            
    
    def validate_chain(self, chain_to_check):
        # shall use check_block_when_resyncing(block, last_block)
        return True
        
    def verify_tx_sig(self, tx):
        # assume all tx signatures are valid to speed up execution
        # return True
        tx_before_signed = copy(tx)
        del tx_before_signed["tx_sig"]
        modulus = tx['rsa_pub_key']["modulus"]
        pub_key = tx['rsa_pub_key']["pub_key"]
        signature = tx["tx_sig"]
        # verify
        hash = int.from_bytes(sha256(str(tx_before_signed).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature     
    
    ######## workers method ########
    
    def ticket_learning(self, comm_round):
        # 1. prune; 2. reinit; 3. train; 4. introduce noice;
        print()
        self.prune()
        self.reinit_params()
        self.train()
        # self.test_indi_accuracy(comm_round)
        # if malicious, introduce noise
        if self._is_malicious:
            self.poison_model()
        
    def prune(self):
        
        identity = "malicious" if self._is_malicious else "legit"

        already_pruned_amount = round(get_pruned_amount_by_weights(model=self.model), 2)
        print(f"worker {self.idx} {identity} is pruning.\nCurrent pruned amount:{already_pruned_amount:.2%}")
        curr_prune_diff = self.blockchain.get_cur_pruning_diff()
        
        if curr_prune_diff > already_pruned_amount:
            self._reinit = True
        
        print(f"Current pruning difficulty: {curr_prune_diff}")
        
        if curr_prune_diff:
            l1_prune(model=self.model,
                    amount=curr_prune_diff,
                    name='weight',
                    verbose=self.args.prune_verbose)
            
            if not self.args.prune_verbose:
                print(f"After pruning, pruned amount:{get_pruned_amount_by_mask(self.model):.2%}")
        else:
            print("Prune skipped.")
    
                        
    def reinit_params(self):
        # reinit if prune_diff increased
        if self._reinit:
            source_params = dict(self.init_global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name.split("_")[0]].data)
            print(f"worker {self.idx} has reinitialized its parameters.")
            self._reinit = False
        else:
            print(f"worker {self.idx} did NOT reinitialize its parameters.")
            
    def train(self):
        print(f"worker {self.idx} with labels {self._user_labels} is training for {self.args.epochs} epochs...")
        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Device={self.idx}, epoch={epoch}, comm_round:{self.blockchain.get_chain_length()+1}")
            metrics = util_train(self.model,
                                 self._train_loader,
                                 self.args.optimizer,
                                 self.args.lr,
                                 self.args.dev_device,
                                 self.args.train_verbose)

            
    def poison_model(self):
        for layer, module in self.model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    noise = self.args.noise_variance * torch.randn(weight_params.size())
                    # variance_of_noise = torch.var(noise)
                    weight_params.add_(noise.to(self.args.dev_device))
        print(f"Device {self.idx} has poisoned its model.")
            
    def create_model_sig(self):
        # TODO - new model signature 
        # model_sig sparsity has to meet the current pruning difficulty, and not detemrined by the worker's own model's pruned amount, as the worker's model will not be recorded in the final block, so a worker can just provide a very small model_sig in terms of dimension to increase the success of passing the model_sig check in the final block. Also this discourages a worker to prune too quickly, because then its model_sig will provide more real model weights
        # spar(model_sig) >= (1-curr_pruining_diff)*sig_portion
        # find all 1s in mask layer by layer
        def populate_layer_dict():
            layer_to_num_picked_ones = {}
            for layer, module in self.model.named_children():
                for name, mask in module.named_buffers():
                    if 'mask' in name:
                        layer_to_num_picked_ones[layer] = 0
            return layer_to_num_picked_ones
        
        def get_ones_postions():
            # total_num_ones = 0
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
                        # total_num_ones += len(one_positions)
            return layer_to_one_positions, layer_to_num_ones
        
        # determine how many ones to pick for signature layer by layer, randomly
        def nums_elems_to_pick_from_layers(total_num_sig_params, layer_to_num_ones, layer_to_num_picked_ones):
            while total_num_sig_params > 0:
                picked_layer, num_ones = random.choice(list(layer_to_num_ones.items()))
                if num_ones == 0:
                    continue
                num_ones_to_pick = min(total_num_sig_params, random.randint(1, num_ones))
                layer_to_num_picked_ones[picked_layer] += num_ones_to_pick
                layer_to_num_ones[picked_layer] -= num_ones_to_pick
                if layer_to_num_ones[picked_layer] == 0:
                    del layer_to_num_ones[picked_layer]
                total_num_sig_params -= num_ones_to_pick
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
        layer_to_one_positions, layer_to_num_ones = get_ones_postions()
        model_params_count = get_num_total_model_params(self.model)
        total_num_sig_params = int(model_params_count * (1 - self.blockchain.get_cur_pruning_diff()) * self.args.sig_portion)
        layer_to_num_picked_ones = nums_elems_to_pick_from_layers(total_num_sig_params, layer_to_num_ones, layer_to_num_picked_ones)
        layer_to_picked_ones_positions = pick_elems_positions_from_layers(layer_to_one_positions, layer_to_num_picked_ones)
        
        # disturb sig_threshold portion of weights
        layer_to_num_picked_disturbs = populate_layer_dict()
        total_num_disturbs = int(total_num_sig_params * (1 - self.args.sig_threshold))
        layer_to_num_picked_disturbs = nums_elems_to_pick_from_layers(total_num_disturbs, layer_to_num_picked_ones, layer_to_num_picked_disturbs)
        layer_to_picked_distrubs_positions = pick_elems_positions_from_layers(layer_to_picked_ones_positions, layer_to_num_picked_disturbs)
        sig_mask = create_sig_mask_with_disturbs(layer_to_picked_ones_positions, layer_to_picked_distrubs_positions)
        model_sig = create_signature(sig_mask)
        
        self.model_sig = model_sig     
        get_model_sig_sparsity(self.model, model_sig)   
        return model_sig
    
    def make_worker_tx(self):
                
        worker_tx = {
            'worker_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'model' : pytorch_make_prune_permanent(self.model),
            'm_sig' : self.model_sig, # TODO
            'm_sig_sig': self.sign_msg(self.model_sig)
        }
        worker_tx['tx_sig'] = self.sign_msg(str(worker_tx))
        self._worker_tx = worker_tx
    
    def asso_validators(self, validators):
        n_validators_to_send = int(len(validators) * self.args.v_portion)
        random.shuffle(validators)
        for validator in validators:
            if validator.is_online:
                validator.associate_with_worker(self)
                self._associated_validators.add(validator)
                n_validators_to_send -= 1
                if n_validators_to_send == 0:
                    print(f"{self.role} {self.idx} associated with validators {[v.idx for v in self._associated_validators]}")
                    break
                
    ### Validators ###
    def associate_with_worker(self, worker):
        self._associated_workers.add(worker)
        
    def receive_and_verify_worker_tx_sig(self):
        for worker in self._associated_workers:
            if self.verify_tx_sig(worker._worker_tx):
                # TODO - check sig of model_sig
                # if self.args.mal_vs and self._is_malicious:
                #     continue
                self._verified_worker_txs[worker.idx] = worker._worker_tx
            else:
                print(f"Signature of tx from worker {worker['idx']} is invalid.")

    def form_validator_tx(self, worker_idx, model_vote=0, shapley_diff_rewards=0, indi_acc_rewards=0):
        validator_tx = {
            '1. validator_idx': self.idx,
            '2. worker_idx': worker_idx,
            '3. worker_model': self._verified_worker_txs[worker_idx]['model'], # will be removed in final block
            '4. worker_m_sig': self._verified_worker_txs[worker_idx]['m_sig'], # will be preserved to record validator's participation, also can be used to verify if a an unsed model has been used and vice versa
            '5. l_sign(worker_m_sig)': self._verified_worker_txs[worker_idx]['m_sig_sig'], # the worker makes sure that no other validators can temper with worker_m_sig
            '6. worker_rsa': self._verified_worker_txs[worker_idx]['rsa_pub_key'],
            '7. validator_vote': model_vote,
            '8. shapley_diff_rewards': shapley_diff_rewards,
            '9. indi_acc_rewards': indi_acc_rewards, # negtive rewards for 8 and 9 do not make sense
            '10. v_sign(vote_or_rewards + worker_sign(l_m_sig))': self.sign_msg(model_vote + self._verified_worker_txs[worker_idx]['m_sig_sig']), # the validator makes sure no other validators can temper with vote'
            'rsa_pub_key': self.return_rsa_pub_key() 
        } 
        validator_tx['tx_sig'] = self.sign_msg(str(validator_tx)) # will be removed in final block
        return validator_tx

    '''
    def shapley_value_validation_VBFL(self, comm_round, idx_to_device):

        
        # 1. Validator one-epoch update from the latest global model and get training accuracy
        # 2. Test worker's model on its training data
        # 3. Compare with threshold

        # 1. Validator one-epoch update from the latest global model and get training accuracy
        temp_validator_model = deepcopy(self.model)
        # one epoch of training
        util_train(temp_validator_model,
                    self._train_loader,
                    self.args.optimizer,
                    self.args.lr,
                    self.args.dev_device,
                    self.args.train_verbose)
        # get validator's accuracy
        self.validator_local_accuracy = test_by_data_set(temp_validator_model,
                self._train_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]

        # 2. Test worker's model on its training data
        worker_idx_to_acc = {}
        for worker_idx, worker_tx in self._verified_worker_txs.items():
            worker_model = worker_tx['model']
            worker_idx_to_acc[worker_idx] = test_by_data_set(worker_model,
                self._train_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        
        # 3. Compare with threshold and vote
        print("validator_local_accuracy", self.validator_local_accuracy)
        for worker_idx, eval_acc in worker_idx_to_acc.items():
            identity = "malicious" if idx_to_device[worker_idx]._is_malicious else "legit"
            print("For worker", worker_idx, identity, f"{eval_acc} - {self.validator_local_accuracy} = {eval_acc - self.validator_local_accuracy}")
            wandb.log({"comm_round": comm_round, f"v_{self.idx}_{self._user_labels}_to_w_{worker_idx}_{idx_to_device[worker_idx]._user_labels}_{identity}": eval_acc - self.validator_local_accuracy})
    '''

    ''' Validation Methods '''

    # helper functions
    def return_worker_to_acc_top_to_low(self, worker_idx_to_model):
        # Test worker's model on its training data
        worker_idx_to_acc = {}
        for worker_idx, worker_model in worker_idx_to_model.items():
            worker_idx_to_acc[worker_idx] = test_by_data_set(worker_model,
                self._train_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        worker_to_acc_top_to_low = {w_idx: acc for w_idx, acc in sorted(worker_idx_to_acc.items(), key=lambda item: item[1], reverse=True)}
        return worker_to_acc_top_to_low

    def filter_by_acc_z_score(self, worker_to_acc, idx_to_device):
        df = pd.DataFrame({'worker_idx': worker_to_acc.keys(), 'acc': worker_to_acc.values()})
        # is_legit_list used to debug and choose the best z_counts
        df['zscore'] = (df.acc - df.acc.mean())/df.acc.std()
        is_legit_list = []
        for iter in range(len(df)):
            worker_idx = int(df[iter:iter+1].iloc[0]['worker_idx'])
            is_legit_list.append(not idx_to_device[worker_idx]._is_malicious)
        df.insert(df.shape[1], "is_legit", is_legit_list)
        df.sort_values(by=['worker_idx'])
        print(df) # debug
        selected_models = df[(df.zscore > -self.args.z_counts) & (df.zscore < self.args.z_counts)]
        unselected_models = df[~df.apply(tuple,1).isin(selected_models.apply(tuple,1))]

        return selected_models, unselected_models

    def shapley_value_get_base_acc(self, models):
        # used in shapley value based validation (validation 1 and 2)
        base_aggr_model = fedavg(models, self.args.dev_device)
        base_aggr_model_acc = test_by_data_set(base_aggr_model,
                               self._train_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['Accuracy'][0]
        return base_aggr_model_acc

    def exclude_one_model_and_fedavg_rest(self, worker_idx_to_model):
        # used in shapley value based validation (validation 1 and 2)
        if len(worker_idx_to_model) == 1:
            sys.exit("This shouldn't happen in exclude_one_model().")
        excluding_one_models_list = {}
        for idx, model in worker_idx_to_model.items():
            tmp_models_list = copy(worker_idx_to_model)
            del tmp_models_list[idx]
            excluding_one_models_list[idx] = fedavg(list(tmp_models_list.values()), self.args.dev_device)
        return excluding_one_models_list

    def validate_model_sig(self):
        # Assume workers are honest in providing their model signatures by summing up rows and columns. This attack is so easy to spot.
        # self.validate_model_sig() worker_model_sig = worker_tx['m_sig'], and check worker_tx['m_m_sig'] by worker's rsa key, easy, skip
        return True

    def validate_model_sparsity(self):
        worker_idx_to_model = {}
        for worker_idx, worker_tx in self._verified_worker_txs.items():
            worker_model = worker_tx['model']
            # validate model spasity
            pruned_amount = round(get_pruned_amount_by_weights(worker_model), 2)
            prune_diff = self.blockchain.get_cur_pruning_diff()
            if round(pruned_amount, 2) < round(prune_diff, 1):
                # skip model below the current pruning difficulty
                print(f"worker {worker_idx}'s prune amount {pruned_amount} is less than current blockchain's prune difficulty {prune_diff}. Model skipped.")
                continue
            # if self.args.mal_vs and self._is_malicious:
                # drop legitimate model
                # even malicious validator cannot pass the model with invalid sparsity because it'll be detected by other validators
                # continue
            
            worker_idx_to_model[worker_idx] = worker_model
        return worker_idx_to_model

    def model_structure_validation(self):
        # self.validate_model_sig()
        worker_idx_to_model = self.validate_model_sparsity()
        return worker_idx_to_model

    # base validatation_method
    def validate_model(self, idx_to_device):

        # validate model signature and sparsity
        worker_idx_to_model = self.model_structure_validation()
            
        if not worker_idx_to_model:
            print(f"{self.role} {self.idx} either has not received any worker tx, or did not pass the verification of any worker tx.")
            return
        
        if len(worker_idx_to_model) == 1:
            # vote = 0, due to lack comparing models
            worker_idx = list(worker_idx_to_model.keys())[0]
            self._validator_txs = [self.form_validator_tx(worker_idx)]
            return
        else:
            if self.args.validation_method == 1:
                self.shapley_value_validation(idx_to_device, worker_idx_to_model)
            elif self.args.validation_method == 2:
                self.filter_valuation(idx_to_device, worker_idx_to_model)
            elif self.args.validation_method == 3:
                self.assumed_attack_level_validation(worker_idx_to_model)
            elif self.args.validation_method == 4:
                self.greedy_soup_validation(worker_idx_to_model)


    # validation_method == 1, malicious validators flip votes (but they probably do not want to do that)
    def shapley_value_validation(self, idx_to_device, worker_idx_to_model):
        
        validator_txes = []
        
        # validate model accuracy
        base_aggr_model_acc = self.shapley_value_get_base_acc(list(worker_idx_to_model.values()))

        user_labels_counter = [] # used for debugging the validation scheme

        # worker_idx_to_weighted_model = avg_individual_model(worker_idx_to_model, len(worker_idx_to_model) - 1)
        
        excluding_one_models_list = self.exclude_one_model_and_fedavg_rest(worker_idx_to_model)
        identity = "malicious" if self._is_malicious else "legit"

        # test each model
        print(f"\nValidator {self.idx} {identity} with base model acc {round(base_aggr_model_acc, 2)} and labels {self._user_labels} starts validating models.")
        
        for worker_idx, model in excluding_one_models_list.items():
            exclu_aggr_model_acc = test_by_data_set(model,
                            self._train_loader,
                            self.args.dev_device,
                            self.args.test_verbose)['Accuracy'][0]
            
            acc_difference = base_aggr_model_acc - exclu_aggr_model_acc

            malicious_worker = idx_to_device[worker_idx]._is_malicious
            judgement = "right"

            if acc_difference > 0:
                model_vote = 1
                inc_or_dec = "decreased"
                if malicious_worker:
                    judgement = "WRONG"
            elif acc_difference == 0:
                # added =0 because if n_class so small, in first few rounds the diff could = 0, so good or bad is undecided
                model_vote = 0
                inc_or_dec = "CAN NOT DECIDE"
                if malicious_worker:
                    judgement = "WRONG"
            else:
                model_vote = -1
                inc_or_dec = "increased"
                if not malicious_worker:
                    judgement = "WRONG"

            print(f"Excluding worker {worker_idx}'s ({idx_to_device[worker_idx]._user_labels}) model, the accuracy {inc_or_dec} by {round(abs(acc_difference), 2)} - voted {model_vote} - Judgement {judgement}.")

            # turn off validation, pass all models, and mal_vs will be turned off
            if self.args.pass_all_models:
                self.args.mal_vs = 0
                model_vote = 1

            # if malicious validator, disturb vote
            if self.args.mal_vs and self._is_malicious:
                model_vote = 1 if model_vote == 0 else model_vote * -1
                # acc_difference = acc_difference * -1 # negative rewards do not make sense

            user_labels_counter.extend(list(idx_to_device[worker_idx]._user_labels))
            
            # form validator tx for this worker tx (and model)
            validator_tx = self.form_validator_tx(worker_idx, model_vote=model_vote, shapley_diff_rewards=acc_difference)
            validator_txes.append(validator_tx)

        # debug the validation mechanism
        if self.args.debug_validation:
            user_labels_counter_dict = dict(collections.Counter(user_labels_counter))
            print("Worker labels total count", {k: v for k, v in sorted(user_labels_counter_dict.items(), key=lambda item: item[1], reverse=True)})
            print("Unique labels", len(user_labels_counter_dict))
        
        self._validator_txs = validator_txes

    # validation_method == 2, malicious validators flip votes (but they probably do not want to do that)
    def filter_valuation(self, idx_to_device, worker_idx_to_model):

        top_models_count = round(len(worker_idx_to_model) * (1 - self.args.assumed_attack_level))

        def prepare_vote_1_models(model_df):
            # return the models to be used mapped to its accuracy
            model_to_indi_acc = {}
            selected_worker_idx_to_model = {}
            for iter in range(min(top_models_count, len(model_df))):
                worker_idx = int(model_df[iter:iter+1].iloc[0]['worker_idx'])
                acc = model_df[iter:iter+1].iloc[0]['acc']
                model_to_indi_acc[worker_idx_to_model[worker_idx]] = acc
                selected_worker_idx_to_model[worker_idx] = worker_idx_to_model[worker_idx]
            return model_to_indi_acc, selected_worker_idx_to_model
        
        def prepare_vote_0_models(model_df):
            worker_to_indi_acc = {}
            for iter in range(min(top_models_count, len(model_df))):
                worker_idx = int(model_df[iter:iter+1].iloc[0]['worker_idx'])
                acc = model_df[iter:iter+1].iloc[0]['acc']
                worker_to_indi_acc[worker_idx] = acc
            return worker_to_indi_acc

             
        validator_txes = []

        worker_to_acc_top_to_low = self.return_worker_to_acc_top_to_low(worker_idx_to_model)

        df_selected_models, df_unselected_models = self.filter_by_acc_z_score(worker_to_acc_top_to_low, idx_to_device)

        
        if self.args.mal_vs and self._is_malicious:
            # use the unselected_models
            v1_model_to_indi_acc, v1_selected_worker_idx_to_model = prepare_vote_1_models(df_unselected_models)
            v0_worker_to_indi_acc = prepare_vote_0_models(df_selected_models)
        else:
            v1_model_to_indi_acc, v1_selected_worker_idx_to_model = prepare_vote_1_models(df_selected_models)
            v0_worker_to_indi_acc = prepare_vote_0_models(df_unselected_models)

        # for vote=1 models, calcuate shaply value accuracy difference for rewards_method_1
        base_aggr_model_acc = self.shapley_value_get_base_acc(list(v1_model_to_indi_acc.keys()))
        excluding_one_models_list = self.exclude_one_model_and_fedavg_rest(v1_selected_worker_idx_to_model)
        
        for worker_idx, model in excluding_one_models_list.items():
            exclu_aggr_model_acc = test_by_data_set(model,
                            self._train_loader,
                            self.args.dev_device,
                            self.args.test_verbose)['Accuracy'][0]
            
            acc_difference = base_aggr_model_acc - exclu_aggr_model_acc
            validator_tx = self.form_validator_tx(worker_idx, model_vote=1, shapley_diff_rewards=acc_difference, indi_acc_rewards=v1_model_to_indi_acc[worker_idx_to_model[worker_idx]])
            validator_txes.append(validator_tx)

        # for vote=0 models, shapley_diff_rewards=0
        for worker_idx, acc in v0_worker_to_indi_acc.items():
            validator_tx = self.form_validator_tx(worker_idx, model_vote=0, indi_acc_rewards=acc)

        self._validator_txs = validator_txes
            

    # validation_method == 3, malicious validators flip votes and reverse rewards (but they probably do not want to do that)
    def assumed_attack_level_validation(self, worker_idx_to_model):

        pos_vote_models_count = round(len(worker_idx_to_model) * (1 - self.args.assumed_attack_level))
        
        worker_to_acc_top_to_low = self.return_worker_to_acc_top_to_low(worker_idx_to_model)
        worker_ranked_acc_top_to_low = list(worker_to_acc_top_to_low.keys())
        worker_ranked_acc_low_to_top = worker_ranked_acc_top_to_low[::-1]

        validator_txes = []

        for worker_iter in range(len(worker_ranked_acc_top_to_low)):
            worker_idx = worker_ranked_acc_top_to_low[worker_iter]
            reversed_worker_idx = worker_ranked_acc_low_to_top[worker_iter]
            if worker_iter < pos_vote_models_count:
                # positive vote
                if self.args.mal_vs and self._is_malicious:
                    vote = 0
                    indi_acc_rewards=worker_to_acc_top_to_low[reversed_worker_idx]
                else:
                    vote = 1
                    indi_acc_rewards=worker_to_acc_top_to_low[worker_idx]
            else:
                # negative vote
                if self.args.mal_vs and self._is_malicious:
                    vote = 1
                    indi_acc_rewards=worker_to_acc_top_to_low[reversed_worker_idx]
                else:
                    vote = 0
                    indi_acc_rewards=worker_to_acc_top_to_low[worker_idx]
            validator_tx = self.form_validator_tx(worker_idx, model_vote = vote, indi_acc_rewards=indi_acc_rewards)
            validator_txes.append(validator_tx)
        self._validator_txs = validator_txes

    # validation_method == 4, malicious validators flip votes and assign bad models top rewards (but they probably do not want to do that)
    def greedy_soup_validation(self, worker_idx_to_model):

        worker_to_acc_top_to_low = self.return_worker_to_acc_top_to_low(worker_idx_to_model)

        validator_txes = []
        worker_idx_to_vote = {}
        worker_idx_to_diff_acc = {}

        ingredients = []
        last_acc = 0

        for worker_idx in worker_to_acc_top_to_low.keys():
            model = worker_idx_to_model[worker_idx]
            new_aggr_models = copy(ingredients)
            new_aggr_models.append(model)
            
            new_aggr_model = fedavg(new_aggr_models, self.args.dev_device)
            to_compare_acc = test_by_data_set(new_aggr_model,
                                self._train_loader,
                                self.args.dev_device,
                                self.args.test_verbose)['Accuracy'][0]
            
            if to_compare_acc >= last_acc:
                ingredients.append(model)
                worker_idx_to_diff_acc[worker_idx] = to_compare_acc - last_acc
                last_acc = to_compare_acc
                worker_idx_to_vote[worker_idx] = 1
            else:
                # worker_idx_to_vote[worker_idx] = 0
                worker_idx_to_vote[worker_idx] = -1
                worker_idx_to_diff_acc[worker_idx] = to_compare_acc - last_acc

        if self.args.mal_vs and self._is_malicious:
            votes_from_0_to_1 = {w: v for w, v in sorted(worker_idx_to_vote.items(), key=lambda item: item[1])}
            worker_iter = 0
            for worker_idx, vote in votes_from_0_to_1.items():
                vote = 1 if vote == 0 else 0
                worker_idx_to_diff_acc[worker_idx] *= -1
                validator_tx = self.form_validator_tx(worker_idx, model_vote = vote, shapley_diff_rewards= worker_idx_to_diff_acc[worker_idx],indi_acc_rewards=worker_to_acc_top_to_low.values()[worker_iter])
                validator_txes.append(validator_tx)
                worker_iter += 1
        else:
            for worker_idx, vote in worker_idx_to_vote.items():
                validator_tx = self.form_validator_tx(worker_idx, model_vote = vote, shapley_diff_rewards= worker_idx_to_diff_acc[worker_idx],indi_acc_rewards=worker_to_acc_top_to_low[worker_idx])
                validator_txes.append(validator_tx)
        self._validator_txs = validator_txes
    
    ''' Validation Methods '''

    def exchange_and_verify_validator_tx(self, validators):
        # key: worker_idx, value: transactions from its associated validators
        self._received_validator_txs = {}
        # exchange among validators
        for validator in validators:
            # if validator == self:
            #     continue - it's okay to have itself
            for tx in validator._validator_txs:
                tx = deepcopy(tx)
                if not self.verify_tx_sig(tx):
                    continue
                    # TODO - record to black list
                # if self.args.mal_vs and self._is_malicious:
                #     # randomly drop tx
                #     if random.random() < 0.5:
                #         continue
                if tx['2. worker_idx'] in self._received_validator_txs:
                    self._received_validator_txs[tx['2. worker_idx']].append(tx)
                else:
                    self._received_validator_txs[tx['2. worker_idx']] = [tx]

    def validator_no_exchange_tx(self):
        self._received_validator_txs = {}
        for tx in self._validator_txs:
            self._received_validator_txs[tx['2. worker_idx']] = [tx]

    def produce_global_model(self):
        # TODO - cross check model sparsity from other validators' transactions	
        # TODO - change _received_validator_txs to _verified_validator_txs
        final_models_to_fedavg = []
        used_worker_txes = {} # worker_idx to its validator tx, used txes in block, verify participating validators and worker's model_sig
        dup_used_worker_txes = {} # identify participating validators and reward info, only verify participating validators
        unused_worker_txes = {} # unused txes in block, verify participating validators and model_sig

        # sum up model votes
        worker_to_votes = {}
        pos_model_votes = 0 # for validation_method == 4
        for worker_idx, corresponding_validators_txes in self._received_validator_txs.items():
            model_votes = sum([validator_tx['7. validator_vote'] for validator_tx in corresponding_validators_txes])
            pos_model_votes += 1 if model_votes >= 0 else 0
            worker_to_votes[worker_idx] = model_votes
            # participating_validators = participating_validators.union(set([validator_tx['1. validator_idx'] for validator_tx in corresponding_validators_txes])) - do in block process
        
        # sort votes by decreasing order, and 
        workers_votes_high_to_low = [w_idx for w_idx, votes in sorted(worker_to_votes.items(), key=lambda item: item[1], reverse=True)]

        if self.args.validation_method == 4:
            top_models_count = pos_model_votes
        else:
            # calculate how many models to choose by # of unique workers times the assumed_attack_level
            top_models_count = round(len(self._received_validator_txs) * self.args.agg_models_portion)
            #determine top voted models

        # choose models for final aggregation
        chosen_workers = workers_votes_high_to_low[:top_models_count]
        for worker_idx in chosen_workers:
            corresponding_validators_txes = self._received_validator_txs[worker_idx]
            # worker_shap_diff_rewards = sum([validator_tx['8. shapley_diff_rewards'] for validator_tx in corresponding_validators_txes])
            # worker_indi_acc_rewards = sum([validator_tx['9. indi_acc_rewards'] for validator_tx in corresponding_validators_txes]) - do those in block processing
            if self.idx in set([validator_tx['1. validator_idx'] for validator_tx in corresponding_validators_txes]):
                for validator_tx in corresponding_validators_txes:
                    if self.idx == validator_tx['1. validator_idx']:
                        # if this validator has this worker's transaction, it prefers its own transaction
                        chosen_tx = validator_tx
                        break
            else:
                # random.choice is necessary because in this design one worker can send different txs to different validators. may change to use stake_book to determine which validator's tx to pick
                chosen_tx = random.choice(corresponding_validators_txes)
            final_models_to_fedavg.append(chosen_tx['3. worker_model'])
            used_worker_txes[worker_idx] = chosen_tx
            # record other unused transactions to reward other participating validators
            # validators do not check if workers send different txes
            corresponding_validators_txes.remove(chosen_tx)
            dup_used_worker_txes[worker_idx] = corresponding_validators_txes

        # for unused transactions, also record them to identify global participating validators
        unchosen_workers = workers_votes_high_to_low[top_models_count:]

        for worker_idx in unchosen_workers:
            unused_worker_txes[worker_idx] = self._received_validator_txs[worker_idx]


        if final_models_to_fedavg:
            self._final_ticket_model = fedavg(final_models_to_fedavg, self.args.dev_device)
        else:
            # no local models have passed the validation, use the latest global model
            # take caution of shallow copy
            self._final_ticket_model = self.model
        # print(self.args.epochs, get_pruned_amount_by_weights(self._final_ticket_model))
        # print()
        self._used_worker_txes = used_worker_txes
        self._dup_used_worker_txes = dup_used_worker_txes
        self._unused_worker_txes = unused_worker_txes
        # no way to ensure that the validator chooses the model and model_sig within the same tx, but if a worker doesn't send different models, there would be no issue. Validator also is responsible to check for the model_sig. If invalid, should drop before the block. Once model_sig found invalid in the block, the whole block becomes invalid and no one will be rewarded, so validators are not incentived to select a model_sig from a different validator_tx that has the same model 


    def remove_model_and_vtx_sig(self):
        
        def remove_from_single_tx(v_tx):
            del v_tx['3. worker_model']
            del v_tx['tx_sig']
        
        for validator_tx in self._used_worker_txes.values():
            remove_from_single_tx(validator_tx)

        for validator_txs in self._dup_used_worker_txes.values():
            for validator_tx in validator_txs:
                remove_from_single_tx(validator_tx)

        for validator_txs in self._unused_worker_txes.values():
            for validator_tx in validator_txs:
                remove_from_single_tx(validator_tx)

    def check_validation_performance(self, block, idx_to_device, comm_round):
        incorrect_pos = 0
        incorrect_neg = 0
        for pos_voted_worker_idx in list(block.used_worker_txes.keys()):
            if idx_to_device[pos_voted_worker_idx]._is_malicious:
                incorrect_pos += 1
        for neg_voted_worker_idx in list(block.unused_worker_txes.keys()):
            if not idx_to_device[neg_voted_worker_idx]._is_malicious:
                incorrect_neg += 1
        print(f"{incorrect_pos} / {len(block.used_worker_txes)} are malicious but used.")
        print(f"{incorrect_neg} / {len(block.unused_worker_txes)} are legit but not used.")
        incorrect_rate = (incorrect_pos + incorrect_neg)/(len(block.used_worker_txes) + len(block.unused_worker_txes))
        try:
            false_positive_rate = incorrect_pos/len(block.used_worker_txes)
        except ZeroDivisionError:
            # no models were voted positively
            false_positive_rate = -0.1
        print(f"False positive rate: {false_positive_rate:.2%}")    
        print(f"Incorrect rate: {incorrect_rate:.2%}")
        # record validation mechanism performance
        wandb.log({"comm_round": comm_round, f"{self.idx}_block_false_positive_rate": round(false_positive_rate, 2)})
        wandb.log({"comm_round": comm_round, f"{self.idx}_block_incorrect_rate": round(incorrect_rate, 2)})

    
    def produce_block(self):
        
        def sign_block(block_to_sign):
            block_to_sign.block_signature = self.sign_msg(str(block_to_sign.__dict__))
      
        last_block_hash = self.blockchain.get_last_block_hash()

        # remove model and validator signature from validator txs
        self.remove_model_and_vtx_sig()

        block = Block(last_block_hash, self._final_ticket_model, self._used_worker_txes, self._dup_used_worker_txes, self._unused_worker_txes, self.idx, self.return_rsa_pub_key())
        sign_block(block)

        self.produced_block = block

        return block
        
    def broadcast_block(self, online_devices_list, block):
        for device in online_devices_list:
            device._received_blocks.append(block)
        
    ### General ###
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}
    
    def sign_msg(self, msg):
        # TODO - sorted migjt be a bug when signing. need to change in VBFL as well
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature

    def verify_block_sig(self, block):
        # assume block signature is not disturbed
        # return True
        block_to_verify = copy(block)
        block_to_verify.block_signature = None
        modulus = block.validator_rsa_pub_key["modulus"]
        pub_key = block.validator_rsa_pub_key["pub_key"]
        signature = block.block_signature
        # verify
        hash = int.from_bytes(sha256(str(block_to_verify.__dict__).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature
    
    def pick_wining_block(self, idx_to_device):
        
        
        if not self._received_blocks:
            print(f"\n{self.idx} has not received any block. Resync chain next round.")
            return
            
        while self._received_blocks:
            # TODO - check while logic when blocks are not valid

            # when all validators have the same stake, workers pick randomly, a validator favors its own block. This most likely happens only in the 1st comm round

            if len(set(self.stake_book.values())) == 1:
                if self.role == 'worker':
                    picked_block = random.choice(self._received_blocks)
                    self._received_blocks.remove(picked_block)
                    if self.verify_block_sig(picked_block):
                        winning_validator = picked_block.produced_by
                    else:
                        continue
                if 'validator' in self.role:
                    winning_validator = self.idx
                    picked_block = self.produced_block                
                print(f"\n{self.role} {self.idx} {self._user_labels} picks {winning_validator}'s {idx_to_device[winning_validator]._user_labels} block.")
                return picked_block
            else:
                self.stake_book = {validator: stake for validator, stake in sorted(self.stake_book.items(), key=lambda x: x[1], reverse=True)}
                received_validators_to_blocks = {block.produced_by: block for block in self._received_blocks}
                for validator, stake in self.stake_book.items():
                    if validator in received_validators_to_blocks:
                        picked_block = received_validators_to_blocks[validator]
                        winning_validator = validator
                        print(f"\n{self.role} {self.idx} ({self._user_labels}) picks {winning_validator}'s ({idx_to_device[winning_validator]._user_labels}) block.")
                        return picked_block           
        
        print(f"\n{self.idx}'s received blocks are not valid. Resync chain next round.")
        return None # all validators are in black list, resync chain

    def check_last_block_hash_match(self, block):
        if not self.blockchain.get_last_block_hash():
            return True
        else:
            last_block_hash = self.blockchain.get_last_block_hash()
            if block.previous_block_hash == last_block_hash:
                return True
        return False

    def check_block_when_resyncing(self, block, last_block):
        # 1. check block signature
        if not self.verify_block_sig(block):
            return False
        # 2. check last block hash match
        if block.previous_block_hash != last_block.compute_hash():
            return False
        # 3. check POLL
        if not self.proof_of_workery_learning(block):
            return False
        # block checked
        return True

    def check_block_when_appending(self, winning_block):
        # 1. check last block hash match
        if not self.check_last_block_hash_match(winning_block):
            print(f"{self.role} {self.idx}'s last block hash conflicts with {winning_block.produced_by}'s block. Resync to its chain next round.")
            self._resync_to = winning_block.produced_by
            return False
        # 2. check POLL
        if not self.proof_of_workery_learning(winning_block):
            print("POLL check failed.")
            return False
        # block checked
        return True

    def proof_of_workery_learning(self, block):
        ''' TODO
        1. Check global sparsity meets pruning difficulty
        2. Check model signature
        3. Record participating validators by verifying pos, dub_pos and neg voted txes by verifying '8 v_sign' in validator_tx (validator verifies other validators don't cheat on workers' signatures)

        promote the idea of proof-of-useful-learning -
        1. neural network training SGD is random, and the model evolvement is represented in the stake book
        2. network pruning is also random given SGD is random, and it can be verified by checking global sparsity and model signature
        3. when chain resync, PoW re-syncs to the longest chain, while we resync to the chain of the current highest stake holder validator. Not easy to hack. 
        (4. role-switching is a protection, since resyncing can only resync to validator.)
        '''
        return True

        
    def append_and_process_block(self, winning_block):

        self.blockchain.chain.append(copy(winning_block))
        
        block = self.blockchain.get_last_block()
        to_reward_workers = list(block.used_worker_txes.keys())
        
        # for unused (duplicated) transactions, reverify participating validators
        pass
        
        # get participating validators
                
        # update stake info
        # workers
        if self.args.reward_method == 1:
            reward_type = '8. shapley_diff_rewards' 
        elif self.args.reward_method == 2:
            reward_type = '9. indi_acc_rewards'
        this_round_worker_rewards = []
        for to_reward_worker in to_reward_workers:
            reward = block.used_worker_txes[to_reward_worker][reward_type]    
            reward += sum([validator_tx[reward_type] for validator_tx in block.dup_used_worker_txes[to_reward_worker]])
            this_round_worker_rewards.append(reward)
        
        # validator
        winning_validator = block.produced_by
        participating_validators = set([validator_tx['1. validator_idx'] for validator_tx in list(block.used_worker_txes.values())])
        for worker_idx, corresponding_txes in list(block.dup_used_worker_txes.items()) + list(block.unused_worker_txes.items()):
            participating_validators = participating_validators.union(set([validator_tx['1. validator_idx'] for validator_tx in corresponding_txes]))
        
        this_round_worker_rewards.sort(reverse=True)
        if len(this_round_worker_rewards) == 0:
            winning_rewards = 0
            participating_rewards = 0
        elif len(this_round_worker_rewards) == 1:
            winning_rewards = this_round_worker_rewards[0]
            participating_rewards = this_round_worker_rewards[0]
        elif len(this_round_worker_rewards) == 2:
            winning_rewards = this_round_worker_rewards[1]
            participating_rewards = this_round_worker_rewards[1]
        else:
            winning_rewards = this_round_worker_rewards[1]
            this_round_worker_rewards.remove(winning_rewards)
            this_round_worker_rewards.pop(0)
            participating_rewards = mean(this_round_worker_rewards)

        for validator in participating_validators:
            if validator == winning_validator:
                self.stake_book[validator] += winning_rewards
            else:
                self.stake_book[validator] += participating_rewards
        
        # used to record if a block is produced by a malicious device
        self.has_appended_block = True
        # update global ticket model
        self.model = deepcopy(block.global_ticket_model)
        
    def test_indi_accuracy(self, comm_round):
        indi_acc = test_by_data_set(self.model,
                self._test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        
        print(f"\n{self.role}_{self.idx}", "indi_acc", round(indi_acc, 2))
        
        wandb.log({"comm_round": comm_round, f"{self.idx}_indi_acc": round(indi_acc, 2)})
        
        return indi_acc

    def test_global_accuracy(self, comm_round):
        global_acc = test_by_data_set(self.model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        print(f"\nglobal_acc: {global_acc}")

        wandb.log({"comm_round": comm_round, "global_acc": round(global_acc, 2)})

        return global_acc