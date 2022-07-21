# TODO - validate chain, check check_chain_validity() on VBFL

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
from util import get_prune_summary, get_pruned_amount_by_0_weights, l1_prune, get_prune_params, copy_model, fedavg, fedavg_lotteryfl, test_by_data_set, AddGaussianNoise, get_model_sig_sparsity, get_num_total_model_params, get_pruned_amount_from_mask, produce_mask_from_model, apply_local_mask, pytorch_make_prune_permanent
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
        self._mask = {}
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._user_labels = user_labels
        self.global_test_loader = global_test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.model_sig = None
        # blockchain variables
        self.role = None
        self.is_malicious = is_malicious
        # self.device_dict = None
        self.peers = None
        self.stake_book = None
        self.blockchain = Blockchain(args.diff_base, args.diff_incre, args.diff_freq, args.target_spar)
        self._received_blocks = []
        self._black_list = {}
        # for lotters
        self._lotter_tx = None
        # for validators
        self._associated_lotters = set()
        self._validator_txs = None
        self._verified_lotter_txs = {}
        self._received_validator_txs = {} # lotter_id_to_corresponding_validator_txes
        self._verified_validator_txs = set()
        self._final_ticket_model = None
        self._pos_votes_txes = {} # models used in final ticket model
        self._participating_validators = set()
        self._neg_votes_txes = {} # models NOT used in final ticket model
        self._dup_pos_votes_txes = {}
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
    
    def resync_chain(self, comm_round, idx_to_device, online_devices_list):
        if comm_round == 1:
            return False
        online_devices_list = copy(online_devices_list)
        online_devices_list.remove(self)
        if self.stake_book:
            # resync chain from the recorded device that has the highest stake and online
            self.stake_book = {validator: stake for validator, stake in sorted(self.stake_book.items(), key=lambda x: x[1], reverse=True)}
            for device_idx, stake in self.stake_book.items():
                device = idx_to_device[device_idx]
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
                        print(f"{self.role} {self.idx}'s chain is resynced from {device.idx}, who picked {idx_to_device[device.idx].blockchain.get_last_block().produced_by}'s block.")
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
        return True
    
    def verify_digital_sig(self, msg, modulus, key):
        return True
        
    def verify_tx_sig(self, tx):
        # tx_before_signed = copy(tx)
        # del tx_before_signed["tx_sig"]
        # modulus = tx['rsa_pub_key']["modulus"]
        # pub_key = tx['rsa_pub_key']["pub_key"]
        # signature = tx["tx_sig"]
        # # verify
        # hash = int.from_bytes(sha256(str(sorted(tx_before_signed.items())).encode('utf-8')).digest(), byteorder='big')
        # hashFromSignature = pow(signature, pub_key, modulus)
        # if hash == hashFromSignature:
        #     return True
        # return False
        return True
            
    
    ######## lotters method ########
    
    def ticket_learning(self):
        print()
        # 1. prune; 2. reinit; 3. train; 4. introduce noice;
        self.prune()
        self.reinit_params()
        self.train()
        # if malicious, introduce noise
        if self.is_malicious:
            self.poison_model()
    
    # def warm_mask(self):
    #     # for new comers - an extra train at the begining
    #     self.train()
        
    def prune(self):
        print(f"Lotter {self.idx} is pruning.\nCurrent pruned amount:{get_pruned_amount_by_0_weights(model=self.model):.2%}")
        if not self._mask:
            # new lotter
            # warm_mask - train then prune
            # self.warm_mask()
            # vs directly prune
            pass
        else:
            # apply local mask to global model weights
            apply_local_mask(self.model, self._mask)
        already_pruned_amount = get_pruned_amount_by_0_weights(model=self.model)
        curr_diff_increiculty = self.blockchain.get_cur_pruning_diff()
        amount_to_prune = max(already_pruned_amount, curr_diff_increiculty)
        
        if amount_to_prune:
            l1_prune(model=self.model,
                    amount=amount_to_prune,
                    name='weight',
                    verbose=self.args.prune_verbose)
            
        if not self.args.prune_verbose:
            print(f"After prunign, pruned amount:{get_pruned_amount_by_0_weights(model=self.model):.2%}")
            
        # update local mask
        for layer, module in self.model.named_children():
            for name, mask in module.named_buffers():
                if 'mask' in name:
                    self._mask[layer] = mask
                        
    def reinit_params(self):
        source_params = dict(self.init_global_model.named_parameters())
        for name, param in self.model.named_parameters():
            param.data.copy_(source_params[name].data)
        print(f"Lotter {self.idx} has reinitialized its parameters.")
            
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
        def malicious_worker_add_noise_to_weights(m):
            with torch.no_grad():
                if hasattr(m, 'weight'):
                    noise = self.noise_variance * torch.randn(m.weight.size())
                    variance_of_noise = torch.var(noise)
                    m.weight.add_(noise.to(self.dev))
                    self.variance_of_noises.append(float(variance_of_noise))
        # check if masks are disturbed as well
        self.model.apply(malicious_worker_add_noise_to_weights)
        print(f"Device {self.idx} has poisoned its model.")
            
    def create_model_sig(self):
        
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
            'm_sig' : self.model_sig,
            'm_sig_sig': self.sign_msg(self.model_sig)
        }
        lotter_tx['tx_sig'] = self.sign_msg(sorted(lotter_tx.items()))
        self._lotter_tx = lotter_tx
    
    def asso_validators(self, validators):
        n_validators_to_send = int(len(validators) * self.args.validator_portion)
        random.shuffle(validators)
        for validator in validators:
            if validator.is_online:
                validator.associate_with_lotter(self)
                n_validators_to_send -= 1
                if n_validators_to_send == 0:
                    break
                
    ### Validators ###
    def associate_with_lotter(self, lotter):
        self._associated_lotters.add(lotter)
        
    def receive_and_verify_lotter_tx_sig(self):
        for lotter in self._associated_lotters:
            if self.verify_tx_sig(lotter._lotter_tx):
                # TODO - check sig of model_sig
                if self.is_malicious:
                    continue
                self._verified_lotter_txs[lotter.idx] = lotter._lotter_tx
            else:
                print(f"Signature of tx from lotter {lotter['idx']} is invalid.")
    
    # TODO - this may not be necesssary, because it will not be included in the block, and if lotter uses low-weight to attack, model score will be low anyway            
    # def verify_model_sig_positions(self):    
    #     def verify_model_sig_positions_by_layer(verified_tx):
    #         lotter_idx = verified_tx['idx']
    #         model = verified_tx['model']
    #         m_sig = verified_tx['m_sig']
    #         for layer, module in model.named_children():
    #             for name, weight_params in module.named_parameters():
    #                 if 'weight' in name:
    #                     unpruned_positions = list(zip(*np.where(weight_params[layer] != 0)))
    #                     model_sig_positions = list(zip(*np.where(m_sig[layer] != 0)))
    #                     for pos in model_sig_positions:
    #                         if pos not in unpruned_positions:
    #                             print(f"Model signature from lotter {lotter_idx} is invalid.")
    #                             return False
    #         return True
    #     passed_model_sig_tx = []
    #     for verified_tx in self._verified_lotter_txs:
    #         if verify_model_sig_positions_by_layer(verified_tx):                    
    #             passed_model_sig_tx.append(verified_tx)
    #     self._verified_lotter_txs = passed_model_sig_tx

    def validate_models_and_init_validator_tx(self, idx_to_device):
        
        # def avg_individual_model(models_list, by_how_many):
        #     new_models_list = deepcopy(models_list)
        #     for idx, model in new_models_list.items():
        #         for name, param in model.named_parameters():
        #             param.data.copy_(torch.mul(param, 1/by_how_many))
        #     return new_models_list    
        
        def exclude_one_model(models_list):
            exclusion_models_list = {}
            for idx, model in models_list.items():
                tmp_models_list = copy(models_list)
                del tmp_models_list[idx]
                exclusion_models_list[idx] = fedavg_lotteryfl(list(tmp_models_list.values()), self.args.dev_device)
            return exclusion_models_list
        
        # validate model and model_sig sparsity
        lotter_idx_to_model = {}
        for lotter_idx, lotter_tx in self._verified_lotter_txs.items():
            lotter_model = lotter_tx['model']
            lotter_model_sig = lotter_tx['m_sig']
            # check lotter_tx['m_m_sig'] by lotter's rsa key, easy, skip
            # validate model spasity
            pruned_amount = round(get_pruned_amount_by_0_weights(lotter_model), 2)
            if round(pruned_amount, 2) < round(self.blockchain.get_cur_pruning_diff(), 1):
                # skip model below the current pruning difficulty
                continue
            if self.is_malicious:
                # drop legitimate model
                # even malicious validator cannot pass the model with invalid sparsity because it'll be detected by other validators
                continue
            # validate model_sig sparsity
            # if not round(get_model_sig_sparsity(self.model, lotter_model_sig), 2) >= (1 - self.blockchain.get_cur_pruning_diff()) * self.args.sig_portion:
            #     continue   # or vote -1?? no, hard requirement
            
            lotter_idx_to_model[lotter_idx] = lotter_model
            
        if not lotter_idx_to_model:
            # this validator either has not received any lotter tx, or did not pass the verification of any lotter tx
            return
        
        # validate model accuracy
        # base_aggr_model = fedavg(list(lotter_idx_to_model.values()), self.args.dev_device)
        base_aggr_model = fedavg_lotteryfl(list(lotter_idx_to_model.values()), self.args.dev_device)
        base_aggr_model_acc = test_by_data_set(base_aggr_model,
                               self._train_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['Accuracy'][0]
        
        # by_how_many = 1 if len(lotter_idx_to_model) == 1 else len(lotter_idx_to_model) - 1
        
        # lotter_idx_to_weighted_model = avg_individual_model(lotter_idx_to_model, by_how_many)
        
        exclusion_models_list = exclude_one_model(lotter_idx_to_model)
        
        # test each model
        validator_txes = []
        
        print(f"\nValidator {self.idx} with base model acc {round(base_aggr_model_acc, 2)} and labels {self._user_labels} starts validating models.")
        
        for lotter_idx, model in exclusion_models_list.items():
            this_model_acc = test_by_data_set(model,
                            self._train_loader,
                            self.args.dev_device,
                            self.args.test_verbose)['Accuracy'][0]
            
            acc_difference = base_aggr_model_acc - this_model_acc
            # added =0 because if n_class so small, in first few rounds the diff could = 0
            if acc_difference >= 0:
                model_vote = 1
                inc_or_dec = "decreased"
            else:
                model_vote = -1
                inc_or_dec = "increased"
        
            # disturb vote
            model_vote = model_vote * -1 if self.is_malicious else model_vote
            
            if model_vote == -1:
                print("debug")
                
            print(f"Excluding lotter {lotter_idx}'s ({idx_to_device[lotter_idx]._user_labels}) model, the accuracy {inc_or_dec} by {round(abs(acc_difference), 2)} - voted {model_vote}.")
            
            # form validator tx for this lotter tx (and model)
            validator_tx = {
                '1. validator_idx': self.idx,
                '2. lotter_idx': lotter_idx,
                '3. lotter_model': self._verified_lotter_txs[lotter_idx]['model'], # will be removed in final block
                '4. lotter_m_sig': self._verified_lotter_txs[lotter_idx]['m_sig'], # will be removed if unpicked for model averaging
                # Not any more. Preserve in final block, can be used to verify if a an unsed transaction has been used, or used is marked as unused
                '5. l_sign(lotter_m_sig)': self._verified_lotter_txs[lotter_idx]['m_sig_sig'], # the lotter makes sure that no other validators can temper with lotter_m_sig
                '6. lotter_rsa': self._verified_lotter_txs[lotter_idx]['rsa_pub_key'],
                '7. validator_rsa': self.return_rsa_pub_key(),
                '8. validator_vote': model_vote,
                '9. v_sign(validator_vote + lotter_sign(l_m_sig))': self.sign_msg(model_vote + self._verified_lotter_txs[lotter_idx]['m_sig_sig']), # the validator makes sure no other validators can temper with vote'
            } # TODO - should also have sign(1,2,5,6,7,8,9), sign (1,2,3,5,6,7,8,9) and sign(all_above) - this will be removed in final block
            
            validator_txes.append(validator_tx)
        self._validator_txs = validator_txes
        
    def exchange_and_verify_validator_tx(self, validators):
        self._received_validator_txs = {}
        # exchange among validators
        for validator in validators:
            # if validator == self:
            #     continue - it's okay to have it
            for tx in validator._validator_txs:
                # assume all signatures are verified
                if self.is_malicious:
                    continue
                if tx['2. lotter_idx'] in self._received_validator_txs:
                    self._received_validator_txs[tx['2. lotter_idx']].append(tx)
                else:
                    self._received_validator_txs[tx['2. lotter_idx']] = [tx]
            # if self.verify_tx_sig(validator._validator_tx):
            #     self._verified_validator_txs.add(validator._validator_tx)
            # else:
            #     print(f"Signature of tx from validator {validator['idx']} is invalid.")
            #     # TODO - record to black list
            
            
        
    
    def produce_global_model(self):
        # TODO - check again model sparsity from other validators' transactions
        # TODO - change _received_validator_txs to _verified_validator_txs
        final_models_to_fedavg = []
        pos_votes_txes = {} # lotter_idx to its validator tx, used txes in block
        duplicated_pos_votes_txes = {} # sign(1256789), recording participating validators
        neg_votes_txes = {} # unused txes in block
        participating_validators = set()
        for lotter_idx, corresponding_validators_txes in self._received_validator_txs.items():
            neg_votes_txes[lotter_idx] = []
            # validators do not check if lotters send different txes
            duplicated_pos_votes_txes[lotter_idx] = []
            # if lotter_idx in self._black_list: # gave up blacklist
            #     if self._black_list[lotter_idx] >= self.args.kick_out_rounds:
            #         continue
            model_votes = sum([validator_tx['8. validator_vote'] for validator_tx in corresponding_validators_txes])
            participating_validators = participating_validators.union(set([validator_tx['1. validator_idx'] for validator_tx in corresponding_validators_txes]))
            if model_votes >= 0:
                # random.choice is necessary because in this design one lotter can send different txs to different validators, or change to use stake_book to determine
                chosen_tx = random.choice(corresponding_validators_txes)
                corresponding_validators_txes.remove(chosen_tx)
                duplicated_pos_votes_txes[lotter_idx].extend(corresponding_validators_txes)
                final_models_to_fedavg.append(chosen_tx['3. lotter_model'])
                pos_votes_txes[lotter_idx] = chosen_tx
            else:
                neg_votes_txes[lotter_idx].extend(corresponding_validators_txes)
        self._final_ticket_model = fedavg_lotteryfl(final_models_to_fedavg, self.args.dev_device)
        self._pos_votes_txes = pos_votes_txes
        self._dup_pos_votes_txes = duplicated_pos_votes_txes
        self._neg_votes_txes = neg_votes_txes
        # no way to ensure that actually the validator chooses the model and model_sig within the same tx, but if a lotter doesn't send different models, there would be no issue
        self._participating_validators = participating_validators
    
    def produce_block(self):
        
        def sign_block(block_to_sign):
            block_to_sign.block_signature = self.sign_msg(block_to_sign.__dict__)
      
        last_block_hash = self.blockchain.get_last_block_hash()
        
        # TODO - delete all models from txs, only preserve model model_sigs
        block = Block(last_block_hash, self._final_ticket_model, self._pos_votes_txes, self._dup_pos_votes_txes, self._neg_votes_txes, self._participating_validators, self.idx, self.return_rsa_pub_key())
        
        sign_block(block)
        return block
        
    def broadcast_block(self, online_devices_list, block):
        for device in online_devices_list:
            device._received_blocks.append(block)
        
    ### General ###
    def return_rsa_pub_key(self):
        return {"modulus": self.modulus, "pub_key": self.public_key}
    
    def sign_msg(self, msg):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self.private_key, self.modulus)
        return signature
    
    def pick_wining_block(self, idx_to_device):
        
        def verify_block_sig(block):
            # TODO - complete
            return True
            block_content = copy(block.__dict__)
            block_content['block_signature'] = None
            return block.__dict__['block_signature'] == sha256(str(sorted(block_content.items())).encode('utf-8')).hexdigest()
        
        while self._received_blocks:
            # TODO - check while logic when blocks are not valid

            if not sum(self.stake_book.values()):
                # joined the 1st comm round, pick randomly
                picked_block = random.choice(self._received_blocks)
                self._received_blocks.remove(picked_block)
                if verify_block_sig(picked_block):
                    winning_validator = picked_block.produced_by
                    print(f"\n{self.role} {self.idx} {self._user_labels} picks {winning_validator}'s {idx_to_device[winning_validator]._user_labels} block.")
                    return picked_block
            else:
                self.stake_book = {validator: stake for validator, stake in sorted(self.stake_book.items(), key=lambda x: x[1], reverse=True)}
                received_validators_to_blocks = {block.produced_by: block for block in self._received_blocks}
                for validator, stake in self.stake_book.items():
                    if validator in received_validators_to_blocks:
                        picked_block = received_validators_to_blocks[validator]
                        winning_validator = validator.idx
                        print(f"\n{self.role} {self.idx} ({self._user_labels}) picks {winning_validator}'s ({idx_to_device[winning_validator]._user_labels}) block.")
                        return picked_block           
        
        print(f"\n{self.idx}'s received blocks are not valud. Resync chain next round.")
        return None # all validators are in black list, resync chain
        
        
    def append_block(self, winning_block):
        if self.blockchain.append_block(winning_block):
            return True
        # all blocks do not match previous_hash, resync chain
        return False
        
    def process_block(self, comm_round):
        
        def verify_m_sig(global_model, model_sig, n_lotters):
            return True
            # may also verify signature sparsity, but it should be verified by validator
            verified_positions = 0
            pass_threshold = int(get_num_total_model_params(global_model) * (1 - self.blockchain.get_cur_pruning_diff()) * self.args.sig_portion * self.args.sig_threshold)
            
            for layer_name, params in global_model.named_parameters():
                if 'weight' in layer_name:
                    layer_name = layer_name.split('.')[0]
                    model_sig_positions = list(zip(*np.where(model_sig[layer_name] != 0)))
                    for pos in model_sig_positions:
                            if params[pos] * n_lotters >= model_sig[layer_name][pos]:
                                verified_positions += 1
                                if verified_positions == pass_threshold:
                                    return True
                            else:
                                pass
            return False
            
            
            
            # for layer, module in global_model.named_children():
            #     for name, weight_params in module.named_parameters():
            #         if 'weight' in name:
            #             model_sig_positions = list(zip(*np.where(model_sig[layer] != 0)))
            #             for pos in model_sig_positions:
            #                 if weight_params[pos] * n_lotters >= model_sig[layer][pos]:
            #                     verified_positions += 1
            #                     if verified_positions == pass_threshold:
            #                         return True
            # return False
            
        block = self.blockchain.get_last_block()
        to_reward_lotters = [] # model signature checks. However, not necessarily send same model to every validator. It can do it anyway, but risk into validator voiding its rewards, so we also do not call "to_reward_lotters" here
        
        # validate model signature
        pass
        
        # no matter used or unused transactions, their model_sig sparsity have already passed hard requirments, so not check again here in code
        
        # for used positive transactions - votes >= 0
        n_lotters = len(block.pos_votes_txes)
        for lotter_idx, tx in block.pos_votes_txes.items():
            if verify_m_sig(block.global_ticket_model, tx['4. lotter_m_sig'], n_lotters):
                to_reward_lotters.append(lotter_idx)
                
        # for unused (duplicated) positive transactions - votes >= 0, record participating validators. Optionally, check if a lotter sends same tx to every validator, and deem it as dishonest if it sends different, but I think it's okay and it's its own risk to let validator pick model_sig that mismatches the original model
        # but at least should check (lotters in pos_votes_txes == lotters in dup_pos_votes_txes), if not, drop this block
        pass
        
                
        # validator could be dishonest about n_lotters - if over half of model signatures invalid, meaning high probablility that most of the lotters are dishonest, or the validator dishonest, so this block has to be dropped
        # if len(to_reward_lotters) < int(n_lotters * self.args.block_drop_threshold):
        #     self.blockchain.drop_block()
        #     return
                    
        # for unused transactions votes < 0
        # if the above passed, most likely validator is honest about n_lotters, so below we still use n_lotters, shouldn't be a big deal
        # if any unsed transaction found being used, reward the lotter and cut winning validator's reward (to normal)
        
        # TRICKY PART IS HOW DO WE CHOOSE n_lotters for unused??
        
        
        # get participating validators - TODO finish the code, iterate over all used and unused validator idxes
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
        
        # update global ticket model
        self.model = deepcopy(block.global_ticket_model)
        
        # global_model_accuracy = test_by_data_set(self.model,
        #                        self._test_loader,
        #                        self.args.dev_device,
        #                        self.args.test_verbose)['Accuracy'][0]
        # wandb.log({f"{self.idx}_global_acc": global_model_accuracy, "comm_round": comm_round})
        
    def test_accuracy(self, comm_round):
        global_acc_bm = test_by_data_set(self.model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        indi_acc_bm = test_by_data_set(self.model,
                self._test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        # apply mask to model
        apply_local_mask(self.model, self._mask)
        global_acc_am = test_by_data_set(self.model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        indi_acc_am = test_by_data_set(self.model,
                self._test_loader,
                self.args.dev_device,
                self.args.test_verbose)['Accuracy'][0]
        
        print("global_acc_bm", round(global_acc_bm, 2), "indi_acc_bm", round(indi_acc_bm, 2), "global_acc_am", round(global_acc_am, 2), "indi_acc_am", round(indi_acc_am, 2))
        
        # wandb.log({"id":self.idx, "global_acc_bm": global_acc_bm, "indi_acc_bm": indi_acc_bm, "global_acc_am": global_acc_am, "indi_acc_am": indi_acc_am})
    
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
        eval_score = test_by_data_set(model,
                               self.test_loader,
                               self.args.dev_device,
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
