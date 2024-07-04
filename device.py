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
# from util import get_prune_summary, get_pruned_amount_by_weights, l1_prune, get_prune_params, copy_model, fedavg, fedavg_workeryfl, test_by_data_set, AddGaussianNoise, get_model_sig_sparsity, get_num_total_model_params, get_pruned_amount_from_mask, produce_mask_from_model, apply_local_mask, make_prune_permanent
from util import train as util_train
from util import test_by_data_set
from pathlib import Path
from sklearn.cluster import KMeans

from copy import copy, deepcopy
from Crypto.PublicKey import RSA
from hashlib import sha256
from Block import Block
from Blockchain import Blockchain
import random
import string
from collections import deque
from statistics import mean

from collections import defaultdict

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
        init_global_model_path
    ):
        self.idx = idx
        self.args = args
        # ticket learning variables
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._user_labels = user_labels
        self.global_test_loader = global_test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.model_path = init_global_model_path
        # blockchain variables
        self.role = None
        self._is_malicious = is_malicious
        self.has_appended_block = False
        # self.device_dict = None
        self.peers = None
        self.blockchain = Blockchain()
        self._received_blocks = {}
        self._resync_to = None # record the last round's picked winning validator to resync chain
        # CELL
        self.cur_prune_rate = 0.00
        self.eita_hat = self.args.eita
        self.eita = self.eita_hat
        self.alpha = self.args.alpha
        self.prune_rates = []
        # for workers
        self._worker_tx = None
        self._associated_validators = set()
        self._local_mask = None
        # for validators
        self._validator_tx = None
        self._associated_workers = set() # ideally, all workers should be associated with a validator
        self._worker_txs = []
        self._verified_worker_txs = {} # signature verified
        self._validator_txs = []
        self._received_validator_txs = {} # worker_id_to_corresponding_validator_txes
        self._verified_validator_txs = {}
        self._final_global_model = None
        self.produced_block = None
        self._iden_benigh_workers = None
        self._pouw_book = {}
        # init key pair
        self.modulus = None
        self.private_key = None
        self.public_key = None
        self.generate_rsa_key()

    @staticmethod
    def assign_ranks(a): # ChatGPT's code
        # Sort the dictionary by value in descending order, with stable sorting to preserve order for ties
        sorted_items = sorted(a.items(), key=lambda item: item[1], reverse=True)
        
        # Initialize rank and create a dictionary for the ranks
        rank = 1
        b = {}
        
        # Iterate through the sorted items
        for i, (key, value) in enumerate(sorted_items):
            # If it's not the first item and the value is the same as the previous value, keep the same rank
            if i > 0 and value == sorted_items[i-1][1]:
                b[key] = rank
            else:
                rank = i + 1  # The rank is the current index + 1
                b[key] = rank
        
        # Transform the ranks to assign the highest number to the highest rank
        max_rank = max(b.values())
        for key in b:
            b[key] = max_rank - b[key] + 1
        
        return b
        
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self.modulus = keyPair.n
        self.private_key = keyPair.d
        self.public_key = keyPair.e
        
    def assign_peers(self, idx_to_device):
        # self.device_dict = idx_to_device
        self.peers = set(idx_to_device.values())
        self.peers.remove(self)
        self.pouw_book = {key: 0 for key in idx_to_device.keys()}
  
    def is_online(self):
        if 1 - get_pruned_amount_by_weights(self.model) <= self.args.target_pruned_sparsity:
            return False
        return random.random() <= self.args.network_stability
    
    def resync_chain(self, comm_round, idx_to_device, online_devices_list):
        if comm_round == 1:
            return False
        if self._resync_to:
            # _resync_to specified to the last round's picked winning validator
            resync_to_device = idx_to_device[self._resync_to]
            if self.validate_chain(resync_to_device.blockchain) and self._resync_to in [d.idx for d in online_devices_list]:
                # update chain
                self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                print(f"\n{self.role} {self.idx}'s chain is resynced from last round's picked winning validator {self._resync_to}.")
                # update pouw_book
                self.pouw_book = resync_to_device.pouw_book
                self._resync_to = None
                return True                
        online_devices_list = copy(online_devices_list)
        if self.pouw_book:
            # resync chain from online validators having the highest recorded pouw
            self.pouw_book = {validator: pouw for validator, pouw in sorted(self.pouw_book.items(), key=lambda x: x[1], reverse=True)}
            for device_idx, pouw in self.pouw_book.items():
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
                        # update pouw_book
                        self.pouw_book = device.pouw_book
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
                self.pouw_book = picked_device.pouw_book
                print(f"{self.role} {self.idx}'s chain resynced chain from {device.idx}.")
                return True
        if self.args.resync_verbose:
            print(f"{self.role} {self.idx}'s chain not resynced.")
        return False
            
    def post_resync(self):
        # update global model from the new block
        self.model = deepcopy(self.blockchain.get_last_block().global_model_in_block)
    
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

    def save_model_weights_to_log(self, comm_round, epoch, global_model=False):
        L_or_M = "M" if self._is_malicious else "L"
        model_save_path = f"{self.args.log_dir}/models_weights/{L_or_M}_{self._user_labels}_{self.idx}"
        Path(model_save_path).mkdir(parents=True, exist_ok=True)
        trainable_model_weights = get_trainable_model_weights(self.model)

        # apply mask just in case
        layer_to_mask = calc_mask_from_model_with_mask_object(self.model)
        if not layer_to_mask:
            layer_to_mask = calc_mask_from_model_without_mask_object(self.model)
        for layer in trainable_model_weights:
            trainable_model_weights[layer] *= np.array(torch.Tensor(layer_to_mask[layer]).cpu())

        if global_model:
            self.model_path = f"{model_save_path}/R{comm_round}.pkl"
            with open(self.model_path, 'wb') as f:
                pickle.dump(trainable_model_weights, f)
        else:
            self.last_local_model_path = f"{model_save_path}/R{comm_round}_E{epoch}.pkl"
            with open(self.last_local_model_path, 'wb') as f:
                pickle.dump(trainable_model_weights, f)
    
    def poison_model(self):
        layer_to_mask = calc_mask_from_model_with_mask_object(self.model) # introduce noise to unpruned weights
        for layer, module in self.model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    # noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * torch.from_numpy(layer_to_mask[layer]).to(self.args.dev_device)
                    noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * layer_to_mask[layer].to(self.args.dev_device)
                    weight_params.add_(noise.to(self.args.dev_device))
        print(f"Device {self.idx} poisoned the whole network with variance {self.args.noise_variance}.") # or should say, unpruned weights?

    def ticket_learning_CELL(self, comm_round):
        # adapoted CELL pruning method
        # BUG OR FEATURE?? every worker prunes on the same global model! - Guess it's a feature, because each worker has different pruning rate
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n----------{L_or_M} Worker:{self.idx} CELL Update---------------------")

        metrics = self.eval_model_by_test(self.model)
        accuracy = metrics['MulticlassAccuracy'][0]
        print(f'Global model local accuracy before pruning and training: {accuracy}')

        # global model prune percentage
        prune_rate = get_prune_summary(model=self.model, name='weight')['global']
           
        if self.cur_prune_rate < self.args.prune_threshold:
            if accuracy > self.eita:
                self.cur_prune_rate = min(self.cur_prune_rate + self.args.prune_step,
                                          self.args.prune_threshold)
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

                self.eita = self.eita_hat

            else:
                self.eita *= self.alpha
                self.prune_rates.append(prune_rate)
        else:
            if self.cur_prune_rate > prune_rate:
                l1_prune(model=self.model,
                         amount=self.cur_prune_rate-prune_rate,
                         name='weight',
                         verbose=self.args.prune_verbose)
                self.prune_rates.append(self.cur_prune_rate)
            else:
                self.prune_rates.append(prune_rate)

        print(f"\nTraining local model")
        self.train(comm_round)

        ticket_acc = self.eval_model_by_test(self.model)["MulticlassAccuracy"][0]
        print(f'Trained model accuracy: {ticket_acc}')

        wandb.log({f"{self.idx}_cur_prune_rate": self.cur_prune_rate}) # worker's prune rate, but not necessarily the same as _percent_pruned because when < validation_threshold, no prune and use the whole model
        # wandb.log({f"{self.idx}_eita": self.eita}) - I don't care logging it
        wandb.log(
            {f"{self.idx}_percent_pruned": self.prune_rates[-1]}) # model sparsity at this moment

        # save last local model
        self.save_model_weights_to_log(comm_round, self.args.epochs)

        if self._is_malicious:
            # poison the last local model
            self.poison_model()
            poinsoned_acc = self.eval_model_by_test(self.model)["MulticlassAccuracy"][0]
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {ticket_acc - poinsoned_acc}.')
            # overwrite the last local model
            self.save_model_weights_to_log(comm_round, self.args.epochs)
            ticket_acc = poinsoned_acc
        
        wandb.log({f"{self.idx}_ticket_local_acc": ticket_acc, "comm_round": comm_round})

    def model_learning_max(self, comm_round):
        
        # train to max accuracy
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} Train to Max Acc Update ---------------------")

        # apply local mask first - abandoned, because the global model should not be consistent with the old mask
        # apply_local_mask(self.model, self._local_mask)

        max_epoch = self.args.epochs # 500 is arbitrary, stronger hardware can have more
        # if self.is_malicious:
        #     max_epoch = 20 # undertrain
        #     if self.idx == 11:
        #         max_epoch = 0 # freerider
        
        epoch = 0
        max_model_epoch = epoch
        max_model = copy_model(self.model, self.args.dev_device)

        g = copy_model(self.model, self.args.dev_device)

        # init max_acc as the initial global model acc on local training set
        max_acc = test_by_data_set(self.model,
                    self._train_loader,
                    self.args.dev_device,
                    self.args.test_verbose)['MulticlassAccuracy'][0]

        layer_to_weights = get_trainable_model_weights(g)
        eds = {l:[] for l in layer_to_weights.keys()}
        css = {l:[] for l in layer_to_weights.keys()}
        pcs = {l:[] for l in layer_to_weights.keys()}
        accs = []
        while epoch < max_epoch and max_acc != 1.0:
        # while epoch < max_epoch:
            if self.args.train_verbose:
                print(f"Client={self.idx}, epoch={epoch + 1}")

            metrics = util_train(self.model,
                                self._train_loader,
                                self.args.optimizer,
                                self.args.lr,
                                self.args.dev_device,
                                self.args.train_verbose)
            acc = metrics['MulticlassAccuracy'][0]
            accs.append(acc)
            # print(epoch + 1, acc)
            if acc > max_acc:
                print(self.idx, "epoch", epoch, acc)
                max_model = copy_model(self.model, self.args.dev_device)
                max_acc = acc
                # max_model_epoch = epoch + 1
                max_model_epoch = epoch

            epoch += 1

            # evaluate similarity
            # ed = compare_nn_euclidean_distance(g, self.model)
            # cs = compare_nn_cosine_similarity(g, self.model)
            # pc = compare_nn_pearson_correlation(g, self.model)
            # for layer in ed.keys():
            #     eds[layer].append(ed[layer])
            #     css[layer].append(cs[layer])
            #     pcs[layer].append(pc[layer])

        print(f"Client {self.idx} trained for {epoch} epochs with max training acc {max_acc} arrived at epoch {max_model_epoch}.")

        self.model = max_model
        self.max_train_acc = max_acc

        # self.is_malicious = True

        self.save_model_weights_to_log(comm_round, max_model_epoch)
        
        wandb.log({f"{self.idx}_local_max_acc_after_training": self.max_train_acc, "comm_round": comm_round})    

    def train(self):
        """
            Train NN
        """
        losses = []

        for epoch in range(1, self.args.epochs + 1):
            if self.args.train_verbose:
                print(
                    f"Worker={self.idx}, epoch={epoch}")

            metrics = util_train(self.model,
                                 self._train_loader,
                                 self.args.optimizer,
                                 self.args.lr,
                                 self.args.dev_device,
                                 self.args.train_verbose)
            losses.append(metrics['Loss'][0])


    def prune_model(self, comm_round):
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} starts pruning ---------------------")

        init_model_acc = self.eval_model_by_train(self.model)["MulticlassAccuracy"][0]
        accs = [init_model_acc]
        models = deque([copy_model(self.model, self.args.dev_device)])
        # print("Initial pruned model accuracy", init_model_acc)

        prune_step = 0.05 # 5% pruning step
        prune_amount = 0 - prune_step # prune 0% at the beginning to create mask object for poison_model function 
        # three prune options
        # 1. if init_model_acc < self.max_train_acc (last round accuracy), prune from the begining
        # 2. if init_model_acc > self.max_train_acc, prune on top of the last pruning rate
        if init_model_acc > self.max_train_acc:
            prune_amount = self.cur_prune_rate - prune_step

        # model prune percentage
        # before_prune_rate = get_prune_summary(model=self.model, name='weight')['global'] # prune_rate = 0s/total_params = 1 - sparsity

        
        def is_decreasing(arr, check_last_n=2):
            if len(arr) < check_last_n:
                return False

            for i in range(-check_last_n, -1):
                if arr[i] <= arr[i + 1]:
                    return False

            return True
        
        last_pruned_model = copy_model(self.model, self.args.dev_device)
        while True:
            prune_amount += prune_step
            # if prune_amount > self.args.max_prune_step: # prevent some devices to prune too aggressively
            #     break
            pruned_model = copy_model(self.model, self.args.dev_device)
            l1_prune(model=pruned_model,
                        amount=prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)
            model_acc = self.eval_model_by_train(pruned_model)["MulticlassAccuracy"][0]
            # print("Intermediate pruned model accuracy", model_acc)

            # (1) prune until accuracy starts to decline
            # if len(model_weights) > 1:
            #     model_weights.popleft()

            #     # Check convergence
            #     if is_decreasing(accs):
            #         break

            
            # (2) prune until the accuracy increases or the accuracy drop exceeds the threshold
            if model_acc > accs[-1]:
                # use the current model
                self.model = copy_model(pruned_model, self.args.dev_device)
                self.max_train_acc = model_acc
                break

            if init_model_acc - model_acc > self.args.prune_acc_drop_threshold or 1 - prune_amount <= self.args.target_pruned_sparsity:
                # revert to the last pruned model
                print("1 - prune_amount", 1 - prune_amount, "target_pruned_sparsity", self.args.target_pruned_sparsity)
                print("init_model_acc - model_acc > self.args.prune_acc_drop_threshold", init_model_acc, model_acc, self.args.prune_acc_drop_threshold)
                self.model = copy_model(last_pruned_model, self.args.dev_device)
                self.max_train_acc = accs[-1]
                break
            
            accs.append(model_acc)
            last_pruned_model = copy_model(pruned_model, self.args.dev_device)

        after_pruning_rate = get_pruned_amount_by_mask(self.model) # prune_rate = 0s/total_params = 1 - sparsity
        after_pruning_acc = self.eval_model_by_train(self.model)["MulticlassAccuracy"][0]
        self.cur_prune_rate = after_pruning_rate

        print(f"Model sparsity: {1 - after_pruning_rate:.2f}")
        print(f"Pruned model before and after accuracy: {init_model_acc:.2f}, {after_pruning_acc:.2f}")
        print(f"Pruned amount: {after_pruning_rate - before_prune_rate:.2f}")

        if self._is_malicious:
            # poison the last local model
            self.poison_model()
            poinsoned_acc = self.eval_model_by_train(self.model)['MulticlassAccuracy'][0]
            accs.append(poinsoned_acc)
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {after_pruning_acc - poinsoned_acc}.')
            # self.max_train_acc = poinsoned_acc  

        # save lateste pruned model
        self.save_model_weights_to_log(comm_round, self.args.epochs)

        # save mask
        self._local_mask = calc_mask_from_model_with_mask_object(self.model)

            
    def make_worker_tx(self):
                
        worker_tx = {
            'worker_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'model' : make_prune_permanent(self.model),
            # 'model_path' : self.last_local_model_path, # in reality could be IPFS
            # TODO - sum over tows and columns as model sig
        }
        worker_tx['tx_sig'] = self.sign_msg(str(worker_tx))
        self._worker_tx = worker_tx
    
    def broadcast_worker_tx(self, online_devices_list):
        # worker broadcast tx, imagine the tx is broadcasted to all devices volunterring to become validators
        # see receive_and_verify_worker_tx_sig()
        print(f"Worker {self.id} is broadcasting worker transaction to validators.")
        return
        # validators = [d for d in online_devices_list if d.role == "validator"]
        # random.shuffle(validators)
        # for validator in validators:
        #     if validator.is_online():
        #         validator.associate_with_worker(self)
        #         self._associated_validators.add(validator)
        #         print(f"{self.role} {self.idx} has broadcasted to validators {[v.idx for v in self._associated_validators]}")
                
    ### Validators ###

    def receive_and_verify_worker_tx_sig(self, online_workers):
        for worker in online_workers:
            if worker == self:
                continue
            if self.verify_tx_sig(worker._worker_tx):
                print(f"Worker {self.idx} has received and verified the signature of the tx from worker {worker.idx}.")
                self._verified_worker_txs[worker.idx] = worker._worker_tx
            else:
                print(f"Signature of tx from worker {worker['idx']} is invalid.")

    def receive_and_verify_validator_tx_sig(self, online_validators):
        for validator in online_validators:
            if validator == self:
                continue
            if self.verify_tx_sig(validator._validator_tx):
                print(f"Validator {self.idx} has received and verified the signature of the tx from validator {validator.idx}.")
                self._verified_validator_txs[validator.idx] = validator._validator_tx
            else:
                print(f"Signature of tx from worker {validator['idx']} is invalid.")


    ''' Validation Methods '''

    def fedavg(
        self,
        iden_benigh_models,
        *args,
        **kwargs
    ):
        weight_per_worker = 1/len(iden_benigh_models)

        aggr_model = fed_avg(
            models=iden_benigh_models,
            weight=weight_per_worker,
            device=self.args.dev_device
        )

        # if self.args.CELL:
        #     pruned_percent = get_prune_summary(aggr_model, name='weight')['global']
        #     # pruned by the earlier zeros in the model
        #     l1_prune(aggr_model, amount=pruned_percent, name='weight')

        # if self.args.overlapping_prune:
        #     # apply mask object to aggr_model. Otherwise won't work in lowOverlappingPrune()
        #     l1_prune(model=aggr_model,
        #             amount=0.00,
        #             name='weight',
        #             verbose=False)
        
        return aggr_model


    # base validatation_method
    # def validate_models(self, comm_round, idx_to_device):

    #     # validate model sparsity
    #     # worker_idx_to_model = self.model_structure_validation()
    #     worker_idx_to_model = {}
    #     for worker_idx, worker_tx in self._verified_worker_txs.items():
    #         worker_idx_to_model[worker_idx] = worker_tx['model_path']

    #     # get layers
    #     layers = []
    #     for layer_name, param in self.init_global_model.named_parameters():
    #         if 'weight' in layer_name:
    #             layers.append(layer_name.split('.')[0])
    #     num_layers = len(layers)

    #     # 2 groups of models and treat the higher center group as legitimate
    #     layer_to_ratios = {l:[] for l in layers} # in the order of worker
    #     worker_to_points = {}
    #     # worker_to_layer_to_ratios = {c: {l: [] for l in layers} for c in idx_to_last_local_model_path.keys()}
    #     for worker_idx, worker_model_path in worker_idx_to_model.items():
    #         layer_to_mask = calculate_overlapping_mask([self.model_path, worker_model_path], self.args.check_whole, self.args.overlapping_threshold, model_validation = True) # change overlapping threshold to just check about overlapped params cosine similarity
    #         for layer, mask in layer_to_mask.items():
    #             # overlapping_ratio = round((mask == 1).sum()/mask.size, 3)
    #             overlapping_ratio = (mask == 1).sum()/mask.size
    #             layer_to_ratios[layer].append(overlapping_ratio)
    #             # worker_to_layer_to_ratios[worker_idx][layer] = overlapping_ratio
    #         worker_to_points[worker_idx] = 0

    #     # group workers based on ratio
    #     kmeans = KMeans(n_clusters=2, random_state=0) 
    #     for layer, ratios in layer_to_ratios.items():
            
    #         kmeans.fit(np.array(ratios).reshape(-1,1))
    #         labels = list(kmeans.labels_)
                   
    #         center0 = kmeans.cluster_centers_[0]
    #         center1 = kmeans.cluster_centers_[1]

    #         benigh_center_group = 0
    #         if center0 < center1:
    #             benigh_center_group = 1

    #         benigh_group = []
    #         workers_in_order = list(worker_idx_to_model.keys())
    #         for worker_iter in range(len(workers_in_order)):
    #             worker_idx = workers_in_order[worker_iter]
    #             if labels[worker_iter] == benigh_center_group:
    #                 benigh_group.append(worker_idx)

    #         for worker_iter in benigh_group:
    #             worker_to_points[worker_iter] += 1
        
    #     self._iden_benigh_workers = [worker_idx for worker_idx in worker_to_points if worker_to_points[worker_idx] > num_layers * 0.5] # was >=

    #     # evaluate validation
    #     false_positive = 0
    #     for benigh_client_idx in self._iden_benigh_workers:
    #         if idx_to_device[benigh_client_idx]._is_malicious:
    #             false_positive += 1
    #     try:
    #         correct_rate = 1 - false_positive/self.args.n_malicious
    #     except ZeroDivisionError:
    #         correct_rate = 1
    #     print(f"Validator {self.idx} selects {self._iden_benigh_workers} as benigh workers.")
    #     print(f"{false_positive} in {self.args.n_malicious} identified wrong. Correct rate - {correct_rate:.2%}")
    #     wandb.log({f"validator_{self.idx}_correct_rate": correct_rate, "comm_round": comm_round})


    # validate model by comparing the masks and euclidean distance of the unpruned weights
    def validate_models(self, comm_round, idx_to_device):
        # at this moment, both models have no mask objects, and may or may not have been pruned
       
        # # get validator's mask
        # validator_layer_to_weights = get_mask(self.model) - mask based validation not good

        # get validator's weights
        validator_layer_to_weights = get_trainable_model_weights(self.model)
       
        def compare_nn_euclidean_distance(nn1, nn2):

            nn1_net = np.array([])
            nn2_net = np.array([])

            for layer in nn1.keys():
                nn1_net = np.concatenate([nn1_net, nn1[layer].flatten()])
                nn2_net = np.concatenate([nn2_net, nn2[layer].flatten()])

            return np.linalg.norm(nn1_net - nn2_net)
        
        worker_to_ed = {}
        worker_to_acc = {}
        for widx, wtx in self._verified_worker_txs.items():
            worker_model = wtx['model']
            worker_layer_to_weights = get_trainable_model_weights(worker_model)        
            # calculate euclidean distance of weights between validator and worker's models
            worker_to_ed[widx] = compare_nn_euclidean_distance(worker_layer_to_weights, validator_layer_to_weights)
            # calculate accuracy by validator's local dataset
            worker_to_acc[widx] = test_by_data_set(worker_model, self._train_loader, self.args.dev_device)["MulticlassAccuracy"][0]

        # based on the euclidean distance, decide if the worker is benigh or not by kmeans = 2
        kmeans = KMeans(n_clusters=2, random_state=0)
        worker_eds = list(worker_to_ed.values())
        kmeans.fit(np.array(worker_eds).reshape(-1,1))
        labels = list(kmeans.labels_)
        center0 = kmeans.cluster_centers_[0]
        center1 = kmeans.cluster_centers_[1]

        benigh_center_group = 1
        if center0 < center1:
            benigh_center_group = 0
        
        workers_in_order = list(worker_to_ed.keys())
        for worker_iter in range(len(workers_in_order)):
            worker_idx = workers_in_order[worker_iter]
            if labels[worker_iter] == benigh_center_group:
                self.benigh_worker_to_acc[worker_idx] = worker_to_acc[worker_idx]
            else:
                self.malicious_worker_to_acc[worker_idx] = worker_to_acc[worker_idx]        
    
    # def validator_no_exchange_tx(self):
    #     self._received_validator_txs = {}
    #     for tx in self._validator_txs:
    #         self._received_validator_txs[tx['2. worker_idx']] = [tx]

    def make_validator_tx(self):
         
        validator_tx = {
            'validator_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'benigh_worker_to_acc' : self.benigh_worker_to_acc,
            'malicious_worker_to_acc' : self.malicious_worker_to_acc
        }
        validator_tx['tx_sig'] = self.sign_msg(str(validator_tx))
        self._validator_tx = validator_tx


    def broadcast_validator_tx(self, online_validators):
        print(f"Validator {self.id} is broadcasting validator transaction to other validators.")
        return


    def produce_global_model_and_reward(self):

        pouw_ranks = assign_ranks(self._pouw_book)

        # aggregate votes - normalize by the rank of useful work of both the worker and the validator
        worker_idx_to_votes = defaultdict(int)
        for validator_idx, validator_tx in self._verified_validator_txs.items():
            for worker_idx in validator_tx['benigh_worker_to_acc'].keys():
                worker_idx_to_votes[worker_idx] += 1 * (pouw_ranks[worker_idx] / len(pouw_ranks)) * (pouw_ranks[validator_idx] / len(pouw_ranks))
            for worker_idx in validator_tx['malicious_worker_to_acc'].keys():
                worker_idx_to_votes[worker_idx] -= 1 * (pouw_ranks[worker_idx] / len(pouw_ranks)) * (pouw_ranks[validator_idx] / len(pouw_ranks))
        
        # for the vote of the validator itself, no normalization needed as an incentive
        for worker_idx in self.benigh_worker_to_acc.keys():
            worker_idx_to_votes[worker_idx] += 1
        for worker_idx in self.malicious_worker_to_acc.keys():
            worker_idx_to_votes[worker_idx] -= 1
        
        # select local models if their votes > 0 for FedAvg
        acc_weight_to_benigh_models = {}
        for worker_idx, votes in worker_idx_to_votes.items():
            # malicious validator's behaviors - flip votes and acc
            if worker_idx in self._verified_worker_txs and ((votes < 0 and self._is_malicious) or (votes > 0 and not self._is_malicious)):
                worker_acc = self.benigh_worker_to_acc[worker_idx] if worker_idx in self.benigh_worker_to_acc else self.malicious_worker_to_acc[worker_idx]
                if self._is_malicious:
                    worker_acc = 1 - worker_acc
                acc_weight_to_benigh_models[worker_acc] = self._verified_worker_txs[worker_idx]['model']
                # reward the workers by their accuracy (not granted yet)
                self._device_to_ungranted_pouw[worker_idx] += worker_acc

        # add its own model
        acc_weight_to_benigh_models[self.max_train_acc] = self.model

        # produce final global model
        # prune with its current pruning rate? abandoned
        self._final_global_model = self.weighted_fedavg(acc_weight_to_benigh_models)

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

        block = Block(last_block_hash, self._final_global_model, self._device_to_ungranted_pouw, self.idx, self.return_rsa_pub_key())
        sign_block(block)

        self.produced_block = block

        return block
        
    def broadcast_block(self, v_idx, online_devices_list, block):
        for device in online_devices_list:
            device._received_blocks[v_idx] = block
        
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

            # when all validators have the same pouw, workers pick randomly, a validator favors its own block. This most likely happens only in the 1st comm round

            if len(set(self.pouw_book.values())) == 1:
                if self.role == 'worker':
                    picked_validator = random.choice(self._received_blocks.keys())
                    picked_block = self._received_blocks[picked_validator]
                    self._received_blocks.remove(picked_validator)
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
                self.pouw_book = {validator: pouw for validator, pouw in sorted(self.pouw_book.items(), key=lambda x: x[1], reverse=True)}
                received_validators_to_blocks = {block.produced_by: block for block in self._received_blocks}
                for validator, pouw in self.pouw_book.items():
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
        # block checked
        return True

    def check_block_when_appending(self, winning_block):
        # check last block hash match
        if not self.check_last_block_hash_match(winning_block):
            print(f"{self.role} {self.idx}'s last block hash conflicts with {winning_block.produced_by}'s block. Resync to its chain next round.")
            self._resync_to = winning_block.produced_by
            return False
        # block checked
        return True

    def proof_of_lottery_learning(self, block):
        ''' TODO
        1. Check global sparsity meets pruning difficulty
        2. Check model signature
        3. Record participating validators by verifying pos, dub_pos and neg voted txes by verifying '8 v_sign' in validator_tx (validator verifies other validators don't cheat on workers' signatures)

        promote the idea of proof-of-useful-learning -
        1. neural network training SGD is random, and the model evolvement is represented in the pouw book
        2. network pruning is also random given SGD is random, and it can be verified by checking global sparsity and model signature
        3. when chain resync, PoW re-syncs to the longest chain, while we resync to the chain of the current highest pouw holder validator. Not easy to hack. 
        (4. role-switching is a protection, since resyncing can only resync to validator.)
        '''
        return True

        
    def append_and_process_block(self, winning_block, comm_round):

        self.blockchain.chain.append(copy(winning_block))
        
        block = self.blockchain.get_last_block()

        # grant pouw to workers
        for worker, reward in block.worker_to_reward.items():
            self.pouw_book[worker] += reward
        
        # used to record if a block is produced by a malicious device
        self.has_appended_block = True
        # update global model
        self.model = deepcopy(block.global_model_in_block)
        # save global model weights and update path
        self.save_model_weights_to_log(comm_round, 0, global_model=True)
        
    def test_indi_accuracy(self, comm_round):
        indi_acc = test_by_data_set(self.model,
                self._test_loader,
                self.args.dev_device,
                self.args.test_verbose)['MulticlassAccuracy'][0]
        
        print(f"\n{self.role}_{self.idx}", "indi_acc", round(indi_acc, 2))
        
        wandb.log({"comm_round": comm_round, f"{self.idx}_indi_acc": round(indi_acc, 2)})
        
        return indi_acc

    def test_global_accuracy(self, comm_round):
        global_acc = test_by_data_set(self.model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['MulticlassAccuracy'][0]
        print(f"\nglobal_acc: {global_acc}")

        wandb.log({"comm_round": comm_round, "global_acc": round(global_acc, 2)})

        return global_acc

    def eval_model_by_test(self, model):
        """
            Eval self.model by test dataset - containing all samples corresponding to the device's training labels
        """
        eval_score = test_by_data_set(model,
                               self._test_loader,
                               self.args.dev_device,
                               self.args.test_verbose)
        return eval_score

    def eval_model_by_train(self, model):
        """
            Eval self.model by test dataset - containing all samples corresponding to the device's training labels
        """
        eval_score = test_by_data_set(model,
                               self._train_loader,
                               self.args.dev_device,
                               self.args.test_verbose)
        return eval_score