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
# from util import get_prune_summary, get_pruned_amount_by_weights, l1_prune, get_prune_params, copy_model, fedavg, fedavg_lotteryfl, test_by_data_set, AddGaussianNoise, get_model_sig_sparsity, get_num_total_model_params, get_pruned_amount_from_mask, produce_mask_from_model, apply_local_mask, pytorch_make_prune_permanent
from util import train as util_train
from util import test_by_data_set

from copy import copy, deepcopy
from Crypto.PublicKey import RSA
from hashlib import sha256
from Block import Block
from Blockchain import Blockchain
import random
import string

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
        # for lotters
        self._lotter_tx = None
        # for validators
        self._associated_lotters = set()
        self._associated_validators = set()
        self._validator_txs = []
        self._verified_lotter_txs = {}
        self._received_validator_txs = {} # lotter_id_to_corresponding_validator_txes
        self._verified_validator_txs = set()
        self._final_ticket_model = None
        self._pos_voted_txes = {} # models used in final ticket model
        self._participating_validators = set()
        self._neg_voted_txes = {} # models NOT used in final ticket model
        self._dup_pos_voted_txes = {}
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
    
    ######## lotters method ########
    
    def ticket_learning(self, comm_round):
        # 1. prune; 2. reinit; 3. train; 4. introduce noice;
        print()
        self.prune()
        self.reinit_params()
        self.train()
        # self.test_accuracy(comm_round)
        # if malicious, introduce noise
        if self._is_malicious:
            self.poison_model()
        
    def prune(self):
        
        identity = "malicious" if self._is_malicious else "legit"

        already_pruned_amount = round(get_pruned_amount_by_weights(model=self.model), 2)
        print(f"Lotter {self.idx} {identity} is pruning.\nCurrent pruned amount:{already_pruned_amount:.2%}")
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
            print(f"Lotter {self.idx} has reinitialized its parameters.")
            self._reinit = False
        else:
            print(f"Lotter {self.idx} did NOT reinitialize its parameters.")
            
    def train(self):
        print(f"Lotter {self.idx} with labels {self._user_labels} is training for {self.args.epochs} epochs...")
        for epoch in range(self.args.epochs):
            if self.args.train_verbose:
                print(
                    f"Device={self.idx}, epoch={epoch}, comm_round:{self.blockchain.get_chain_length()+1}")
            metrics = util_train(self.model,
                                 self._train_loader,
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
        # model_sig sparsity has to meet the current pruning difficulty, and not detemrined by the lotter's own model's pruned amount, as the lotter's model will not be recorded in the final block, so a lotter can just provide a very small model_sig in terms of dimension to increase the success of passing the model_sig check in the final block. Also this discourages a lotter to prune too quickly, because then its model_sig will provide more real model weights
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
    
    def make_lotter_tx(self):
                
        lotter_tx = {
            'lotter_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'model' : pytorch_make_prune_permanent(self.model),
            'm_sig' : self.model_sig, # TODO
            'm_sig_sig': self.sign_msg(self.model_sig)
        }
        lotter_tx['tx_sig'] = self.sign_msg(str(lotter_tx))
        self._lotter_tx = lotter_tx
    
    def asso_validators(self, validators):
        n_validators_to_send = int(len(validators) * self.args.validator_portion)
        random.shuffle(validators)
        for validator in validators:
            if validator.is_online:
                validator.associate_with_lotter(self)
                self._associated_validators.add(validator)
                n_validators_to_send -= 1
                if n_validators_to_send == 0:
                    print(f"{self.role} {self.idx} associated with validators {[v.idx for v in self._associated_validators]}")
                    break
                
    ### Validators ###
    def associate_with_lotter(self, lotter):
        self._associated_lotters.add(lotter)
        
    def receive_and_verify_lotter_tx_sig(self):
        for lotter in self._associated_lotters:
            if self.verify_tx_sig(lotter._lotter_tx):
                # TODO - check sig of model_sig
                # if self.args.malicious_validators and self._is_malicious:
                #     continue
                self._verified_lotter_txs[lotter.idx] = lotter._lotter_tx
            else:
                print(f"Signature of tx from lotter {lotter['idx']} is invalid.")

    def form_validator_tx(self, lotter_idx, model_vote):
        validator_tx = {
            '1. validator_idx': self.idx,
            '2. lotter_idx': lotter_idx,
            '3. lotter_model': self._verified_lotter_txs[lotter_idx]['model'], # will be removed in final block
            '4. lotter_m_sig': self._verified_lotter_txs[lotter_idx]['m_sig'], # will be preserved to record validator's participation, also can be used to verify if a an unsed model has been used and vice versa
            '5. l_sign(lotter_m_sig)': self._verified_lotter_txs[lotter_idx]['m_sig_sig'], # the lotter makes sure that no other validators can temper with lotter_m_sig
            '6. lotter_rsa': self._verified_lotter_txs[lotter_idx]['rsa_pub_key'],
            '7. validator_vote': model_vote,
            '8. v_sign(validator_vote + lotter_sign(l_m_sig))': self.sign_msg(model_vote + self._verified_lotter_txs[lotter_idx]['m_sig_sig']), # the validator makes sure no other validators can temper with vote'
            'rsa_pub_key': self.return_rsa_pub_key() 
        } 
        validator_tx['tx_sig'] = self.sign_msg(str(validator_tx)) # will be removed in final block
        return validator_tx

    def validate_models_and_init_validator_tx(self, idx_to_device):

        # Assume lotters are honest in providing their model signatures by summing up rows and columns. This attack is so easy to spot.
        
        def exclude_one_model(models_list):
            if len(models_list) == 1:
                sys.exit("This shouldn't happen in exclude_one_model().")
            excluding_one_models_list = {}
            for idx, model in models_list.items():
                tmp_models_list = copy(models_list)
                del tmp_models_list[idx]
                excluding_one_models_list[idx] = fedavg(list(tmp_models_list.values()), self.args.dev_device)
            return excluding_one_models_list
        
        # validate model sparsity
        lotter_idx_to_model = {}
        for lotter_idx, lotter_tx in self._verified_lotter_txs.items():
            lotter_model = lotter_tx['model']
            lotter_model_sig = lotter_tx['m_sig']
            # TODO - check lotter_tx['m_m_sig'] by lotter's rsa key, easy, skip
            # validate model spasity
            pruned_amount = round(get_pruned_amount_by_weights(lotter_model), 2)
            prune_diff = self.blockchain.get_cur_pruning_diff()
            if round(pruned_amount, 2) < round(prune_diff, 1):
                # skip model below the current pruning difficulty
                print(f"Lotter {lotter_idx}'s prune amount {pruned_amount} is less than current blockchain's prune difficulty {prune_diff}. Model skipped.")
                continue
            # if self.args.malicious_validators and self._is_malicious:
                # drop legitimate model
                # even malicious validator cannot pass the model with invalid sparsity because it'll be detected by other validators
                # continue
            
            lotter_idx_to_model[lotter_idx] = lotter_model
            
        if not lotter_idx_to_model:
            print(f"{self.role} {self.idx} either has not received any lotter tx, or did not pass the verification of any lotter tx.")
            return
        
        # validate model accuracy
        base_aggr_model = fedavg(list(lotter_idx_to_model.values()), self.args.dev_device)
        base_aggr_model_acc = test_by_data_set(base_aggr_model,
                               self._train_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['Accuracy'][0]

        validator_txes = []
        if len(lotter_idx_to_model) == 1:
            # vote = 0, due to lack comparing models
            lotter_idx = list(lotter_idx_to_model.keys())[0]
            model_vote = 0
            validator_txes.append(self.form_validator_tx(lotter_idx, model_vote))
        else:
            # lotter_idx_to_weighted_model = avg_individual_model(lotter_idx_to_model, len(lotter_idx_to_model) - 1)
            
            excluding_one_models_list = exclude_one_model(lotter_idx_to_model)
            
            identity = "malicious" if self._is_malicious else "legit"

            # test each model
            print(f"\nValidator {self.idx} {identity} with base model acc {round(base_aggr_model_acc, 2)} and labels {self._user_labels} starts validating models.")
            
            for lotter_idx, model in excluding_one_models_list.items():
                this_model_acc = test_by_data_set(model,
                                self._train_loader,
                                self.args.dev_device,
                                self.args.test_verbose)['Accuracy'][0]
                
                acc_difference = base_aggr_model_acc - this_model_acc

                malicious_lotter = idx_to_device[lotter_idx]._is_malicious
                judgement = "right"

                if acc_difference > 0:
                    model_vote = 1
                    inc_or_dec = "decreased"
                    if malicious_lotter:
                        judgement = "WRONG"
                elif acc_difference == 0:
                    # added =0 because if n_class so small, in first few rounds the diff could = 0, so good or bad is undecided
                    model_vote = 0
                    inc_or_dec = "CAN NOT DECIDE"
                    if malicious_lotter:
                        judgement = "WRONG"
                else:
                    model_vote = -1
                    inc_or_dec = "increased"
                    if not malicious_lotter:
                        judgement = "WRONG"
            
                # disturb vote
                if self.args.malicious_validators and self._is_malicious:
                    model_vote = model_vote * -1

                print(f"Excluding lotter {lotter_idx}'s ({idx_to_device[lotter_idx]._user_labels}) model, the accuracy {inc_or_dec} by {round(abs(acc_difference), 2)} - voted {model_vote} - Judgement {judgement}.")
                
                # form validator tx for this lotter tx (and model)
                validator_tx = self.form_validator_tx(lotter_idx, model_vote)
                validator_txes.append(validator_tx)
        
        self._validator_txs = validator_txes
        
    def exchange_and_verify_validator_tx(self, validators):
        # key: lotter_idx, value: transactions from its associated validators
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
                # if self.args.malicious_validators and self._is_malicious:
                #     # randomly drop tx
                #     if random.random() < 0.5:
                #         continue
                if tx['2. lotter_idx'] in self._received_validator_txs:
                    self._received_validator_txs[tx['2. lotter_idx']].append(tx)
                else:
                    self._received_validator_txs[tx['2. lotter_idx']] = [tx]
                    
    
    def produce_global_model(self):
        # TODO - check again model sparsity from other validators' transactions
        # TODO - change _received_validator_txs to _verified_validator_txs
        final_models_to_fedavg = []
        pos_voted_txes = {} # lotter_idx to its validator tx, used txes in block, verify participating validators and model_sig
        duplicated_pos_voted_txes = {} # verify participating validators
        neg_voted_txes = {} # unused txes in block, verify participating validators and model_sig
        participating_validators = set()
        for lotter_idx, corresponding_validators_txes in self._received_validator_txs.items():
            neg_voted_txes[lotter_idx] = []
            # validators do not check if lotters send different txes
            duplicated_pos_voted_txes[lotter_idx] = []
            # if lotter_idx in self._black_list: # gave up blacklist
            #     if self._black_list[lotter_idx] >= self.args.kick_out_rounds:
            #         continue
            model_votes = sum([validator_tx['7. validator_vote'] for validator_tx in corresponding_validators_txes])
            participating_validators = participating_validators.union(set([validator_tx['1. validator_idx'] for validator_tx in corresponding_validators_txes]))
            if model_votes >= 0:
                # random.choice is necessary because in this design one lotter can send different txs to different validators. can change to use stake_book to determine which validator's tx to pick
                chosen_tx = random.choice(corresponding_validators_txes)
                corresponding_validators_txes.remove(chosen_tx)
                duplicated_pos_voted_txes[lotter_idx].extend(corresponding_validators_txes)
                final_models_to_fedavg.append(chosen_tx['3. lotter_model'])
                pos_voted_txes[lotter_idx] = chosen_tx
                del neg_voted_txes[lotter_idx]
            else:
                neg_voted_txes[lotter_idx].extend(corresponding_validators_txes)
                del duplicated_pos_voted_txes[lotter_idx]
        # self._final_ticket_model = fedavg_lotteryfl(final_models_to_fedavg, self.args.dev_device)
        self._final_ticket_model = fedavg(final_models_to_fedavg, self.args.dev_device)
        # print(self.args.epochs, get_pruned_amount_by_weights(self._final_ticket_model))
        # print()
        self._pos_voted_txes = pos_voted_txes
        self._dup_pos_voted_txes = duplicated_pos_voted_txes
        self._neg_voted_txes = neg_voted_txes
        # no way to ensure that the validator chooses the model and model_sig within the same tx, but if a lotter doesn't send different models, there would be no issue. Validator also is responsible to check for the model_sig. If invalid, should drop before the block. Once model_sig found invalid in the block, the whole block becomes invalid and no one will be rewarded, so validators are not incentived to select a model_sig from a different validator_tx that has the same model 
        self._participating_validators = participating_validators

    def remove_model_and_vtx_sig(self):
        
        def remove_from_single_tx(v_tx):
            del v_tx['3. lotter_model']
            del v_tx['tx_sig']
        
        for validator_tx in self._pos_voted_txes.values():
            remove_from_single_tx(validator_tx)

        for validator_txs in self._dup_pos_voted_txes.values():
            for validator_tx in validator_txs:
                remove_from_single_tx(validator_tx)

        for validator_txs in self._neg_voted_txes.values():
            for validator_tx in validator_txs:
                remove_from_single_tx(validator_tx)

    def check_validation_performance(self, block, idx_to_device, comm_round):
        incorrect_pos = 0
        incorrect_neg = 0
        for pos_voted_lotter_idx in list(block.pos_voted_txes.keys()):
            if idx_to_device[pos_voted_lotter_idx]._is_malicious:
                incorrect_pos += 1
        for neg_voted_lotter_idx in list(block.neg_voted_txes.keys()):
            if not idx_to_device[neg_voted_lotter_idx]._is_malicious:
                incorrect_neg += 1
        print(f"{incorrect_pos} / {len(block.pos_voted_txes)} are malicious but used.")
        print(f"{incorrect_neg} / {len(block.neg_voted_txes)} are legit but not used.")
        incorrect_rate = (incorrect_pos + incorrect_neg)/(len(block.pos_voted_txes) + len(block.neg_voted_txes))
        print(f"Incorrect rate: {incorrect_rate:.2%}")
        # record validation mechanism performance
        wandb.log({"comm_round": comm_round, f"{self.idx}_block_incorrect_rate": round(incorrect_rate, 2)})

    
    def produce_block(self):
        
        def sign_block(block_to_sign):
            block_to_sign.block_signature = self.sign_msg(str(block_to_sign.__dict__))
      
        last_block_hash = self.blockchain.get_last_block_hash()

        # remove model and validator signature from validator txs
        self.remove_model_and_vtx_sig()

        block = Block(last_block_hash, self._final_ticket_model, self._pos_voted_txes, self._dup_pos_voted_txes, self._neg_voted_txes, self._participating_validators, self.idx, self.return_rsa_pub_key())
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

            # when all validators have the same stake, lotters pick randomly, a validator favors its own block. This most likely happens only in the 1st comm round

            if len(set(self.stake_book.values())) == 1:
                if self.role == 'lotter':
                    picked_block = random.choice(self._received_blocks)
                    self._received_blocks.remove(picked_block)
                    if self.verify_block_sig(picked_block):
                        winning_validator = picked_block.produced_by
                    else:
                        continue
                if self.role == 'validator':
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
        if not self.proof_of_lottery_learning(block):
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
        if not self.proof_of_lottery_learning(winning_block):
            print("POLL check failed.")
            return False
        # block checked
        return True

    def proof_of_lottery_learning(self, block):
        ''' TODO
        1. Check global sparsity meets pruning difficulty
        2. Check model signature
        3. Record participating validators by verifying pos, dub_pos and neg voted txes by verifying '8 v_sign' in validator_tx

        promote the idea of proof-of-useful-learning -
        1. neural network training SGD is random, and the model evolvement is represented in the stake book
        2. network pruning is also random given SGD is random, and it can be verified by checking global sparsity and model signature
        3. when chain resync, PoW re-syncs to the longest chain, while we resync to the chain of the current highest stake holder validator. Not easy to hack
        '''
        return True

        
    def append_and_process_block(self, winning_block):

        self.blockchain.chain.append(copy(winning_block))
        
        block = self.blockchain.get_last_block()
        to_reward_lotters = list(block.pos_voted_txes.keys())
        
        # for unused (duplicated) positive transactions - votes >= 0, reverify participating validators
        pass
        
        # get participating validators - TODO finish the code in POLL(), iterate over all used and unused validator idxes
        # temporarily use block.participating_validators
                
        # update stake info
        for to_reward_lotter in to_reward_lotters:
            self.stake_book[to_reward_lotter] += self.args.lotter_reward
        
        winning_validator = block.produced_by
        for validator in block.participating_validators:
            if validator == winning_validator:
                self.stake_book[validator] += self.args.win_val_reward
            else:
                self.stake_book[validator] += self.args.validator_reward
        
        # used to record if a block is produced by a malicious device
        self.has_appended_block = True
        # update global ticket model
        self.model = deepcopy(block.global_ticket_model)
        
    def test_accuracy(self, comm_round):
        global_acc = test_by_data_set(self.model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        indi_acc = test_by_data_set(self.model,
                self._test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        
        print(f"\n{self.role}", self.idx,"\nglobal_acc", round(global_acc, 2), "indi_acc", round(indi_acc, 2))
        
        wandb.log({"comm_round": comm_round, f"{self.idx}_global_acc": round(global_acc, 2), f"{self.idx}_indi_acc": round(indi_acc, 2)})
        
        return global_acc, indi_acc
        