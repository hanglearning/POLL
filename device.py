# TODO - record accuracy and pruned amount after training and pruning

import torch
import numpy as np
import wandb
from util import *
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
import math

from collections import defaultdict
from torch.utils.data import DataLoader


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
        self.args = args
        
        # blockchain variables
        self.idx = idx
        self.role = None
        self._is_malicious = is_malicious
        self.online = True
        self.has_appended_block = False
        self.peers = set()
        self.blockchain = Blockchain()
        self._received_blocks = {}
        self._resync_to = None # record the last round's picked winning validator to resync chain
        self.verified_winning_block = None
        # for workers
        self._worker_tx = None
        self.layer_to_model_sig_row = {}
        self.layer_to_model_sig_col = {}
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._user_labels = user_labels
        self.global_test_loader = global_test_loader
        self.init_global_model = copy_model(init_global_model, args.dev_device)
        self.model = copy_model(init_global_model, args.dev_device)
        self.model_path = init_global_model_path
        self.max_model_acc = 0
        self._worker_pruned_amount = 0
        # for validators
        self._validator_tx = None
        self._verified_worker_txs = {} # signature verified
        self._verified_validator_txs = {}
        self._final_global_model = None
        self.produced_block = None
        self._pouw_book = {}
        self.worker_to_model_sig = {}
        self.worker_to_acc = {}
        self._device_to_ungranted_uw = defaultdict(float)
        # init key pair
        self._modulus = None
        self._private_key = None
        self.public_key = None
        self.generate_rsa_key()
    
    ''' Generic Operations '''

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
    
    ''' Workers' Method '''        

    def model_learning_max(self, comm_round):

        produce_mask_from_model_in_place(self.model)

        wandb.log({f"{self.idx}_{self._user_labels}_global_test_acc": self.eval_model_by_global_test(self.model), "comm_round": comm_round})
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} {self._user_labels} Train to Max Acc Update ---------------------")

        if comm_round > 1 and self.args.rewind:
        # reinitialize model with initial params
            source_params = dict(self.init_global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)

        max_epoch = self.args.epochs

        # label flipping attack
        if self._is_malicious and self.args.attack_type == 2:
            self._train_loader.dataset.targets = 9 - self._train_loader.dataset.targets

        # lazy worker
        if self._is_malicious and self.args.attack_type == 3:
            max_epoch = int(max_epoch * 0.2)


        # init max_acc as the initial global model acc on local training set
        max_acc = self.eval_model_by_train(self.model)

        if self._is_malicious and self.args.attack_type == 1:
            # skip training and poison local model on trainable weights before submission
            self.poison_model(self.model)
            poinsoned_acc = self.eval_model_by_train(self.model)
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {max_acc - poinsoned_acc}.')
            self.max_model_acc = poinsoned_acc  
            # wandb.log({f"{self.idx}_after_poisoning_acc": poinsoned_acc, "comm_round": comm_round})  
        else:
            max_model = copy_model(self.model, self.args.dev_device)
            max_model_epoch = epoch = 0
            # train to max accuracy
            while epoch < max_epoch and max_acc != 1.0:
                if self.args.train_verbose:
                    print(f"Worker={self.idx}, epoch={epoch + 1}")

                util_train(self.model,
                            self._train_loader,
                            self.args.optimizer,
                            self.args.lr,
                            self.args.dev_device,
                            self.args.train_verbose)
                acc = self.eval_model_by_train(self.model)
                if acc > max_acc:
                    max_model = copy_model(self.model, self.args.dev_device)
                    max_acc = acc
                    max_model_epoch = epoch + 1

                epoch += 1


            print(f"Worker {self.idx} with max training acc {max_acc} arrived at epoch {max_model_epoch}.")
            wandb.log({f"{self.idx}_{self._user_labels}_training_max_epoch": max_model_epoch, "comm_round": comm_round})

            self.model = max_model
            self.max_model_acc = max_acc

        # self.save_model_weights_to_log(comm_round, max_model_epoch)
        wandb.log({f"{self.idx}_{self._user_labels}_trained_model_sparsity": 1 - get_pruned_amount(self.model), "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_max_local_training_acc": self.max_model_acc, "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_local_test_acc": self.eval_model_by_local_test(self.model), "comm_round": comm_round})

    def worker_prune(self, comm_round):

        if not self._is_malicious and self.max_model_acc < self.args.worker_prune_acc_trigger:
            print(f"Worker {self.idx}'s local model max accuracy is < the prune acc trigger {self.args.worker_prune_acc_trigger}. Skip pruning.")
            return

        # model prune percentage
        init_pruned_amount = get_pruned_amount(self.model) # pruned_amount = 0s/total_params = 1 - sparsity
        if not self._is_malicious and 1 - init_pruned_amount <= self.args.target_sparsity:
            print(f"Worker {self.idx}'s model at sparsity {1 - init_pruned_amount}, which is already <= the target sparsity {self.args.target_sparsity}. Skip pruning.")
            return
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} starts pruning ---------------------")

        init_model_acc = self.eval_model_by_train(self.model)
        accs = [init_model_acc]
        # models = deque([copy_model(self.model, self.args.dev_device)]) - used if want the model with the best accuracy arrived at an intermediate pruned amount

        to_prune_amount = init_pruned_amount
        last_pruned_model = copy_model(self.model, self.args.dev_device)

        while True:
            to_prune_amount += self.args.prune_step
            pruned_model = copy_model(self.model, self.args.dev_device)
            make_prune_permanent(pruned_model)
            l1_prune(model=pruned_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)
            
            model_acc = self.eval_model_by_train(pruned_model)

            # prune until the accuracy drop exceeds the threshold or below the target sparsity
            if init_model_acc - model_acc > self.args.prune_acc_drop_threshold or 1 - to_prune_amount <= self.args.target_sparsity:
                # revert to the last pruned model
                self.model = copy_model(last_pruned_model, self.args.dev_device)
                self.max_model_acc = accs[-1]
                break
            
            accs.append(model_acc)
            last_pruned_model = copy_model(pruned_model, self.args.dev_device)

        after_pruned_amount = get_pruned_amount(self.model) # pruned_amount = 0s/total_params = 1 - sparsity
        after_pruning_acc = self.eval_model_by_train(self.model)

        self._worker_pruned_amount = after_pruned_amount

        print(f"Model sparsity: {1 - after_pruned_amount:.2f}")
        print(f"Pruned model before and after accuracy: {init_model_acc:.2f}, {after_pruning_acc:.2f}")
        print(f"Pruned amount: {after_pruned_amount - init_pruned_amount:.2f}")

        wandb.log({f"{self.idx}_{self._user_labels}_after_pruning_sparsity": 1 - after_pruned_amount, "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_after_pruning_training_acc": after_pruning_acc, "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_after_pruning_local_test_acc": self.eval_model_by_local_test(self.model), "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_after_pruning_global_test_acc": self.eval_model_by_global_test(self.model), "comm_round": comm_round})
  

        # save lateste pruned model
        # self.save_model_weights_to_log(comm_round, self.args.epochs)

    def poison_model(self, model):
        layer_to_mask = calc_mask_from_model_with_mask_object(model) # introduce noise to unpruned weights
        for layer, module in model.named_children():
            for name, weight_params in module.named_parameters():
                if "weight" in name:
                    # noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * torch.from_numpy(layer_to_mask[layer]).to(self.args.dev_device)
                    noise = self.args.noise_variance * torch.randn(weight_params.size()).to(self.args.dev_device) * layer_to_mask[layer].to(self.args.dev_device)
                    weight_params = weight_params + noise.to(self.args.dev_device)
        print(f"Device {self.idx} poisoned the whole neural network with variance {self.args.noise_variance}.") # or should say, unpruned weights?

    def generate_model_sig(self):
        
        make_prune_permanent(self.model)
        # zero knowledge proof of model ownership
        self.layer_to_model_sig_row, self.layer_to_model_sig_col = sum_over_model_params(self.model)
        
    def make_worker_tx(self):
                
        worker_tx = {
            'worker_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'model' : self.model,
            # 'model_path' : self.last_local_model_path, # in reality could be IPFS
            'model_sig_row': self.layer_to_model_sig_row,
            'model_sig_col': self.layer_to_model_sig_col
        }
        worker_tx['model_sig_row_sig'] = self.sign_msg(str(worker_tx['model_sig_row']))
        worker_tx['model_sig_col_sig'] = self.sign_msg(str(worker_tx['model_sig_col']))
        worker_tx['tx_sig'] = self.sign_msg(str(worker_tx))
        self._worker_tx = worker_tx
    
    def broadcast_worker_tx(self):
        # worker broadcast tx, imagine the tx is broadcasted to all peers volunterring to become validators
        # see receive_and_verify_worker_tx_sig()
        return
   
    ''' Validators' Methods '''

    def receive_and_verify_worker_tx_sig(self, online_workers):
        for worker in online_workers:
            if worker.idx not in self.peers:
                continue
            self.update_peers(worker.peers)
            if self.verify_tx_sig(worker._worker_tx):
                if self.args.validation_verbose:
                    print(f"Validator {self.idx} has received and verified the signature of the tx from worker {worker.idx}.")
                self._verified_worker_txs[worker.idx] = worker._worker_tx
            else:
                print(f"Signature of tx from worker {worker['idx']} is invalid.")

    def receive_and_verify_validator_tx_sig(self, online_validators):
        for validator in online_validators:
            # if validator == self:
            #     continue
            if validator.idx not in self.peers:
                continue
            self.update_peers(validator.peers)
            if self.verify_tx_sig(validator._validator_tx):
                if self.args.validation_verbose:
                    print(f"Validator {self.idx} has received and verified the signature of the tx from validator {validator.idx}.")
                self._verified_validator_txs[validator.idx] = validator._validator_tx
            else:
                print(f"Signature of tx from worker {validator['idx']} is invalid.")

    def validate_models(self):

        # validate model siganture
        for widx, wtx in self._verified_worker_txs.items():
            worker_model = wtx['model']
            worker_model_sig_row_sig = wtx['model_sig_row_sig']
            worker_model_sig_col_sig = wtx['model_sig_col_sig']
            worker_rsa = wtx['rsa_pub_key']

            worker_layer_to_model_sig_row, worker_layer_to_model_sig_col = sum_over_model_params(worker_model)
            if self.compare_dicts_of_tensors(wtx['model_sig_row'], worker_layer_to_model_sig_row)\
                  and self.compare_dicts_of_tensors(wtx['model_sig_col'], worker_layer_to_model_sig_col)\
                  and self.verify_msg(wtx['model_sig_row'], worker_model_sig_row_sig, worker_rsa['pub_key'], worker_rsa['modulus'])\
                  and self.verify_msg(wtx['model_sig_col'], worker_model_sig_col_sig, worker_rsa['pub_key'], worker_rsa['modulus']):
                
                if self.args.validation_verbose:
                    print(f"Worker {widx} has valid model signature.")
            
                self.worker_to_model_sig[widx] = {'model_sig_row': wtx['model_sig_row'], 'model_sig_row_sig': wtx['model_sig_row_sig'], 'model_sig_col': wtx['model_sig_col'], 'model_sig_col_sig': wtx['model_sig_col_sig'], 'worker_rsa': worker_rsa}
            
        for widx, wtx in self._verified_worker_txs.items():
            worker_model = wtx['model']
            # calculate accuracy by validator's local dataset
            self.worker_to_acc[widx] = self.eval_model_by_train(worker_model)

        # add itself's accuracy
        self.worker_to_acc[self.idx] = self.max_model_acc

        # sometimes may inverse the accuracy weights to account for minority workers
        if self.args.inverse_acc_weights and random.random() <= 0.5:
            self.worker_to_acc = {worker_idx: 1 - acc for worker_idx, acc in self.worker_to_acc.items()}
            print(f"Validator {self.idx} has inversed its accuracy weights.")

    def make_validator_tx(self):
         
        validator_tx = {
            'validator_idx' : self.idx,
            'rsa_pub_key': self.return_rsa_pub_key(),
            'worker_to_acc' : self.worker_to_acc,
        }
        validator_tx['tx_sig'] = self.sign_msg(str(validator_tx))
        self._validator_tx = validator_tx

    def broadcast_validator_tx(self, online_validators):
        return

    def produce_global_model_and_reward(self, idx_to_device, comm_round):

        # NOTE - if change the aggregation rule, also need to change in verify_winning_block()

        # aggregate votes and accuracies - normalize by validator_power, defined by the historical useful work of the validator + 1 (to avoid float point number and 0 division)
        # for the useful work of the validator itself, it is directly adding its own max model accuracy, as an incentive to become a validator
        worker_to_acc_weight = defaultdict(float)
        for validator_idx, validator_tx in self._verified_validator_txs.items():
            validator_power = self._pouw_book[validator_idx] + 1
            for worker_idx, worker_acc in validator_tx['worker_to_acc'].items():
                worker_to_acc_weight[worker_idx] += worker_acc * validator_power
                self._device_to_ungranted_uw[worker_idx] += worker_acc
        
        # get models for aggregation
        worker_to_model = {worker_idx: self._verified_worker_txs[worker_idx]['model'] for worker_idx in worker_to_acc_weight}

        # only preserve model signature of selected models
        self.worker_to_model_sig = {worker_idx: self.worker_to_model_sig[worker_idx] for worker_idx in worker_to_acc_weight}

        # only assign useful work to selected models
        self._device_to_ungranted_uw = {worker_idx: self._device_to_ungranted_uw[worker_idx] for worker_idx in worker_to_acc_weight}

        # normalize weights to between 0 and 1
        worker_to_acc_weight = {worker_idx: acc/sum(worker_to_acc_weight.values()) for worker_idx, acc in worker_to_acc_weight.items()}
        
        # produce final global model
        self._final_global_model = weighted_fedavg(worker_to_acc_weight, worker_to_model, device=self.args.dev_device)

    def validator_post_prune(self): # prune by the weighted average of the pruned amount of the selected models

        init_pruned_amount = get_pruned_amount(self._final_global_model) # pruned_amount = 0s/total_params = 1 - sparsity
        
        if 1 - init_pruned_amount <= self.args.target_sparsity:
            print(f"\nValidator {self.idx}'s model at sparsity {1 - init_pruned_amount}, which is already <= the target sparsity. Skip post-pruning.")
            return
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Validator:{self.idx} post pruning ---------------------")

        selected_worker_to_pruned_amount = {}
        selected_worker_to_power = {}
        if self.idx in self._device_to_ungranted_uw:
            selected_worker_to_pruned_amount[self.idx] = self._worker_pruned_amount
            selected_worker_to_power[self.idx] = self._pouw_book[self.idx] + 1
            
        for worker_idx, _ in self._device_to_ungranted_uw.items():
            if worker_idx == self.idx:
                continue
            worker_model = self._verified_worker_txs[worker_idx]['model']
            selected_worker_to_pruned_amount[worker_idx] = get_pruned_amount(worker_model) 
            selected_worker_to_power[worker_idx] = self._pouw_book[worker_idx] + 1
        
        worker_to_prune_weight = {worker_idx: power/sum(selected_worker_to_power.values()) for worker_idx, power in selected_worker_to_power.items()}

        need_pruned_amount = sum([selected_worker_to_pruned_amount[worker_idx] * weight for worker_idx, weight in worker_to_prune_weight.items()])
        if self._is_malicious:
            need_pruned_amount *= 2

        if need_pruned_amount <= init_pruned_amount:
            print(f"The need_pruned_amount value ({need_pruned_amount}) <= init_pruned_amount ({init_pruned_amount}). Validator {self.idx} skips post-pruning.")
            return

        need_pruned_amount = min(need_pruned_amount, 1 - self.args.target_sparsity)
        to_prune_amount = need_pruned_amount
        if check_mask_object_from_model(self._final_global_model):
            to_prune_amount = (need_pruned_amount - init_pruned_amount) / (1 - init_pruned_amount)

        # post_prune the model
        l1_prune(model=self._final_global_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)

        print(f"{L_or_M} Validator {self.idx} has pruned {need_pruned_amount - init_pruned_amount:.2f} of the model. Final sparsity: {1 - need_pruned_amount:.2f}.")


    def produce_block(self):
        
        def sign_block(block_to_sign):
            block_to_sign.block_signature = self.sign_msg(str(block_to_sign.__dict__))

        # self-assign validator's useful work to itself
        last_block = self.blockchain.get_last_block() # before appending the winning block

        # assign useful work to itself by difference of pruned amount between last block and this block
        last_block_global_model_pruned_amount = get_pruned_amount(last_block.global_model) if last_block else 0
        new_global_model_pruned_amount = get_pruned_amount(self._final_global_model)
        self._device_to_ungranted_uw[self.idx] = self._device_to_ungranted_uw.get(self.idx, 0) + max(0, new_global_model_pruned_amount - last_block_global_model_pruned_amount)

        last_block_hash = self.blockchain.get_last_block_hash()

        block = Block(last_block_hash, self._final_global_model, self._device_to_ungranted_uw, self.idx, self.worker_to_model_sig, self._verified_validator_txs, self.return_rsa_pub_key())
        sign_block(block)

        self.produced_block = block
        
    def broadcast_block(self):
        # see receive_blocks()
        return
        
    ''' Blockchain Operations '''
        
    def generate_rsa_key(self):
        keyPair = RSA.generate(bits=1024)
        self._modulus = keyPair.n
        self._private_key = keyPair.d
        self.public_key = keyPair.e
        
    def assign_peers(self, idx_to_device):
        peer_size = math.ceil(len(idx_to_device) * self.args.peer_percent)
        self.peers = set(random.sample(list(idx_to_device.keys()), peer_size))
        self.peers.add(self.idx) # include itself
        self._pouw_book = {key: 0 for key in self.peers}
        
    def update_peers(self, peers_of_other_device):
        new_peers = peers_of_other_device.difference(self.peers)
        self._pouw_book.update({key: 0 for key in new_peers})
        self.peers.update(new_peers)

    def set_online(self):
        if self.args.malicious_always_online and self._is_malicious:
            self.online = True
            return True
        
        self.online = random.random() <= self.args.network_stability
        if not self.online:
            print(f"Device {self.idx} is offline in this communication round.")
        return self.online
    
    def is_online(self):
        return self.online
    
    def recalc_useful_work(self):
        self._pouw_book = {idx: 0 for idx in self._pouw_book}
        for block in self.blockchain.chain:
            for idx in block.device_to_uw:
                self._pouw_book[idx] += block.device_to_uw[idx]
    
    def resync_chain(self, comm_round, idx_to_device, skip_check_peers=False):
        # NOTE - if change logic of resync_chain(), also need to change logic in pick_winning_block()
        """ 
            Return:
            - True if the chain needs to be resynced, False otherwise
        """
        if comm_round == 1:
            # validator not applicable to resync chain
            return False

        if skip_check_peers:
            resync_to_device = idx_to_device[self._resync_to]
            # came from verify_winning_block() when hash is inconsistant rather than in the beginning, direct resync
            if self.validate_chain(resync_to_device.blockchain):
                # update chain
                self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                print(f"\n{self.role} {self.idx}'s chain is resynced from the picked winning validator {self._resync_to}.")                    
                return True
        
        longer_chain_peers = set()
        # check peer's longest chain length
        online_peers = [peer for peer in self.peers if idx_to_device[peer].is_online()]
        for peer in online_peers:
            if idx_to_device[peer].blockchain.get_chain_length() > self.blockchain.get_chain_length():
                # if any online peer's chain is longer, may need to resync. "may" because the longer chain may not be legitimate since this is not PoW, but PoUW similar to PoS
                longer_chain_peers.add(peer)        
        if not longer_chain_peers:
            return False
        
        # may need to resync
        if self._resync_to:
            if self._resync_to in longer_chain_peers:
                # _resync_to specified to the last time's picked winning validator
                resync_to_device = idx_to_device[self._resync_to]
                if self.blockchain.get_last_block_hash() == resync_to_device.blockchain.get_last_block_hash():
                    # if two chains are the same, no need to resync. assume the last block's hash is valid
                    return False
                if self.validate_chain(resync_to_device.blockchain):
                    # update chain
                    self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                    print(f"\n{self.role} {self.idx}'s chain is resynced from last time's picked winning validator {self._resync_to}.")                    
                    return True
                else:
                    print(f"\nDevice {self.idx}'s _resync_to device's ({self._resync_to}) chain is invalid, resync to another online peer based on '(uw + 1) * (pruned_amount + 1)'.")
            else:
                print(f"\nDevice {self.idx}'s _resync_to device ({self._resync_to})'s chain is not longer than its own chain. May need to resync to another online peer based on '(uw + 1) * (pruned_amount + 1)'.") # in the case both devices were offline
        else:
            print(f"\nDevice {self.idx}'s does not have a _resync_to device, resync to another online peer based on '(uw + 1) * (pruned_amount + 1)'.")


        # resync chain from online peers using the same logic in pick_winning_block()
        online_peers = [peer for peer in self.peers if idx_to_device[peer].is_online() and idx_to_device[peer].blockchain.get_last_block()]
        online_peer_to_uw_pruned = {peer: (self._pouw_book[peer] + 1) *(get_pruned_amount(idx_to_device[peer].blockchain.get_last_block().global_model) + 1) for peer in online_peers}
        top_uw_pruned = max(online_peer_to_uw_pruned.values())
        candidates = [peer for peer, uw_pruned in online_peer_to_uw_pruned.items() if uw_pruned == top_uw_pruned]
        self._resync_to = random.choice(candidates)
        resync_to_device = idx_to_device[self._resync_to]

        # compare chain difference, assume the last block's hash is valid
        if self.blockchain.get_last_block_hash() == resync_to_device.blockchain.get_last_block_hash():
            return False
        else:
            # validate chain
            if not self.validate_chain(resync_to_device.blockchain):
                print(f"resync_to device {resync_to_device.idx} chain validation failed. Chain not resynced.")
                return False
            # update chain
            self.blockchain.replace_chain(resync_to_device.blockchain.chain)
            print(f"\n{self.role} {self.idx}'s chain is resynced from {resync_to_device.idx}, who picked {resync_to_device.blockchain.get_last_block().produced_by}'s block.")
            return True

            
    def post_resync(self, idx_to_device):
        # update global model from the new block
        self.model = deepcopy(self.blockchain.get_last_block().global_model)
        # update peers
        self.update_peers(idx_to_device[self._resync_to].peers)
        # recalculate useful work
        self.recalc_useful_work()
    
    def validate_chain(self, chain_to_check):
        # TODO - should also verify the block signatures and the model signatures
        blockchain_to_check = chain_to_check.get_chain()
        for i in range(1, len(blockchain_to_check)):
            if not blockchain_to_check[i].previous_block_hash == blockchain_to_check[i-1].compute_hash():
                return False
        return True
        
    def verify_tx_sig(self, tx):
        tx_before_signed = copy(tx)
        del tx_before_signed["tx_sig"]
        modulus = tx['rsa_pub_key']["modulus"]
        pub_key = tx['rsa_pub_key']["pub_key"]
        signature = tx["tx_sig"]
        # verify
        hash = int.from_bytes(sha256(str(tx_before_signed).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, pub_key, modulus)
        return hash == hashFromSignature
    
    def return_rsa_pub_key(self):
        return {"modulus": self._modulus, "pub_key": self.public_key}
    
    def sign_msg(self, msg):
        # TODO - sorted migjt be a bug when signing. need to change in VBFL as well
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        # pow() is python built-in modular exponentiation function
        signature = pow(hash, self._private_key, self._modulus)
        return signature

    def verify_msg(self, msg, signature, public_key, modulus):
        hash = int.from_bytes(sha256(str(msg).encode('utf-8')).digest(), byteorder='big')
        hashFromSignature = pow(signature, public_key, modulus)
        return hash == hashFromSignature
    
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
    
    def receive_blocks(self, online_validators):
        for validator in online_validators:
            if validator.idx in self.peers:
                self.update_peers(validator.peers)
                self._received_blocks[validator.idx] = validator.produced_block

    def pick_winning_block(self, idx_to_device):

        # NOTE - if change logic of pick_winning_block(), also need to change logic in resync_chain()
        
        # pick the block with the highest (useful_work * pruned_amount) - mitigate monopoly
        picked_block = None

        if not self._received_blocks:
            print(f"\n{self.idx} has not received any block. Resync chain next round.")
            return picked_block
        
        received_validators_to_blocks = {block.produced_by: block for block in self._received_blocks.values()}
        received_validators_pouw_book = {block.produced_by: self._pouw_book[block.produced_by] for block in self._received_blocks.values()}
        received_validators_pruned_amount = {block.produced_by: get_pruned_amount(block.global_model) for block in self._received_blocks.values()} # a pruned amount is included in the block after validator post prune, use get_pruned_amount() 
        validator_to_uw_pruned = {validator: (uw + 1) * (pruned_amount + 1) for validator, uw in received_validators_pouw_book.items() for validator, pruned_amount in received_validators_pruned_amount.items()}
        top_uw_pruned = max(validator_to_uw_pruned.values())
        candidates = [validator for validator, uw_pruned in validator_to_uw_pruned.items() if uw_pruned == top_uw_pruned]
        # get the winning validator
        winning_validator = random.choice(candidates) # may cause forking in the 1st round and some middle rounds
        if self.idx in candidates:
            # oppourtunistic validator
            winning_validator = self.idx
        # winning_validator = max(validator_to_uw_pruned, key=validator_to_uw_pruned.get)        

        print(f"\n{self.role} {self.idx} ({self._user_labels}) picks {winning_validator}'s ({idx_to_device[winning_validator]._user_labels}) block.")

        return received_validators_to_blocks[winning_validator]

    def check_block_when_resyncing(self, block, last_block):
        # 1. check block signature
        if not self.verify_block_sig(block):
            return False
        # 2. check last block hash match
        if block.previous_block_hash != last_block.compute_hash():
            return False
        # block checked
        return True
    
    def check_last_block_hash_match(self, block):
        if not self.blockchain.get_last_block_hash():
            return True
        else:
            last_block_hash = self.blockchain.get_last_block_hash()
            if block.previous_block_hash == last_block_hash:
                return True
        return False

    def verify_winning_block(self, winning_block, comm_round, idx_to_device):

        # NOTE - change the aggregation rule if there's change in produce_global_model_and_reward()
        
        # verify block signature
        if not self.verify_block_sig(winning_block):
            print(f"{self.role} {self.idx}'s picked winning block has invalid signature. Block discarded.")
            return False
        
        # verify validator transactions signature
        for val_tx in winning_block.validator_txs.values():
            if not self.verify_tx_sig(val_tx):
                # TODO - may compare validator's signature with itself's received validator_txs, but a validator may send different transactions to sabortage this process
                print(f"{self.role} {self.idx}'s picked winning block has invalid validator transaction signature. Block discarded.")
                return False

        # check last block hash match
        if not self.check_last_block_hash_match(winning_block):
            print(f"{self.role} {self.idx}'s last block hash conflicts with {winning_block.produced_by}'s block. Resync to its chain.")
            self._resync_to = winning_block.produced_by
            self.resync_chain(comm_round, idx_to_device, skip_check_peers = True)
            self.post_resync(idx_to_device)
        

        ''' validate model signature to make sure the validator is performing model aggregation honestly '''
        layer_to_model_sig_row, layer_to_model_sig_col = sum_over_model_params(winning_block.global_model)
        
        # perform model signature aggregation by the same rule in produce_global_model_and_reward()
        worker_to_acc_weight = defaultdict(float)
        device_to_should_uw = defaultdict(float)
        for validator_idx, validator_tx in winning_block.validator_txs.items():
            validator_power = self._pouw_book[validator_idx] + 1
            for worker_idx, worker_acc in validator_tx['worker_to_acc'].items():
                worker_to_acc_weight[worker_idx] += worker_acc * validator_power
                device_to_should_uw[worker_idx] += worker_acc
        
        
        # (1) verify if validator honestly assigned useful work to devices
        # validator_self_assigned_uw = winning_block.device_to_uw[winning_block.produced_by] - self._pouw_book[winning_block.produced_by]
        last_block = self.blockchain.get_last_block() # before appending the winning block

        last_block_global_model_pruned_amount = get_pruned_amount(last_block.global_model) if last_block else 0
        new_global_model_pruned_amount = get_pruned_amount(winning_block.global_model)
        validator_should_self_assigned_uw = max(0, new_global_model_pruned_amount - last_block_global_model_pruned_amount)
        device_to_should_uw[winning_block.produced_by] += validator_should_self_assigned_uw

        if device_to_should_uw != winning_block.device_to_uw:
            print(f"{self.role} {self.idx}'s picked winning block has invalid useful work assignment. Block discarded.")
            return False

        # (2) verify if validator honestly aggregated the models
        # normalize weights to between 0 and 1
        worker_to_acc_weight = {worker_idx: acc/sum(worker_to_acc_weight.values()) for worker_idx, acc in worker_to_acc_weight.items()}
        
        # apply weights to worker's model signatures
        workers_layer_to_model_sig_row = {}
        workers_layer_to_model_sig_col = {}
        for worker_idx, acc_weight in worker_to_acc_weight.items():
            model_sig_row = winning_block.worker_to_model_sig[worker_idx]['model_sig_row']
            model_sig_col = winning_block.worker_to_model_sig[worker_idx]['model_sig_col']
            for layer in model_sig_row:
                if layer not in workers_layer_to_model_sig_row:
                    workers_layer_to_model_sig_row[layer] = model_sig_row[layer] * acc_weight
                    workers_layer_to_model_sig_col[layer] = model_sig_col[layer] * acc_weight
                else:
                    workers_layer_to_model_sig_row[layer] += model_sig_row[layer] * acc_weight
                    workers_layer_to_model_sig_col[layer] += model_sig_col[layer] * acc_weight
        
        if not self.compare_dicts_of_tensors(layer_to_model_sig_row, workers_layer_to_model_sig_row) or not self.compare_dicts_of_tensors(layer_to_model_sig_col, workers_layer_to_model_sig_col):
            print(f"{self.role} {self.idx}'s picked winning block has invalid workers' model signatures or useful work book is inconsistent with the block producer's.")
            return False

        self.verified_winning_block = winning_block
        return True
        
    def process_and_append_block(self, comm_round):

        if not self.verified_winning_block:
            print(f"\nNo verified winning block to append. Device {self.idx} will resync to last time's picked winning validator({self._resync_to})'s chain.")
            return

        self._resync_to = self.verified_winning_block.produced_by # in case of offline, resync to this validator's chain
        
        self.old_pouw_book = deepcopy(self._pouw_book) # helper used in check_validation_performance()

        # grant useful work to devices
        for device_idx, useful_work in self.verified_winning_block.device_to_uw.items():
            self._pouw_book[device_idx] += useful_work

        # used to record if a block is produced by a malicious device
        self.has_appended_block = True
        # update global model
        self.model = deepcopy(self.verified_winning_block.global_model)
        
        # save global model weights and update path
        # self.save_model_weights_to_log(comm_round, 0, global_model=True)
        
        self.blockchain.chain.append(deepcopy(self.verified_winning_block))

        print(f"\n{self.role} {self.idx} has appended the winning block produced by {self.verified_winning_block.produced_by}.")

    ''' Helper Functions '''

    def compare_dicts_of_tensors(self, dict1, dict2, atol=1e-3, rtol=1e-3):
            """
            Compares two dictionaries with torch.Tensor values.

            Parameters:
            - dict1, dict2: The dictionaries to compare.
            - atol: Absolute tolerance.
            - rtol: Relative tolerance.

            Returns:
            - True if the dictionaries are equivalent, False otherwise.
            """
            if dict1.keys() != dict2.keys():
                return False
            
            for key in dict1:
                if not torch.allclose(dict1[key], dict2[key], atol=atol, rtol=rtol):
                    return False
            
            return True
    
    def check_validation_performance(self, idx_to_device, comm_round):

        worker_to_acc_weight = defaultdict(float)
        for validator_idx, validator_tx in self.verified_winning_block.validator_txs.items():
            validator_power = self.old_pouw_book[validator_idx] + 1
            for worker_idx, worker_acc in validator_tx['worker_to_acc'].items():
                worker_to_acc_weight[worker_idx] += worker_acc * validator_power
        worker_to_acc_weight = {worker_idx: acc/sum(worker_to_acc_weight.values()) for worker_idx, acc in worker_to_acc_weight.items()}

        i = 1
        for widx, acc_weight in sorted(worker_to_acc_weight.items(), key=lambda x: x[1]):
            if idx_to_device[widx]._is_malicious:
                if i < len(worker_to_acc_weight) // 2:
                    print(f"Malicious worker {widx} has accuracy-model-weight {acc_weight:.3f}, ranked {i}, in the lower half (lower rank means smaller weight).")
                else:
                    print("\033[91m" + f"Malicious worker {widx} has accuracy-model-weight {acc_weight:.2f}, ranked {i}, in the higher half (higher rank means heavier weight)." + "\033[0m")
            i += 1


    def eval_model_by_local_test(self, model):
        """
            Eval self.model by local test dataset - containing all samples corresponding to the device's training labels
        """
        return test_by_data_set(model,
                               self._test_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['MulticlassAccuracy'][0]
    
    def eval_model_by_global_test(self, model):
        """
            Eval self.model by global test dataset - containing all samples of the original test dataset
        """
        return test_by_data_set(model,
                self.global_test_loader,
                self.args.dev_device,
                self.args.test_verbose)['MulticlassAccuracy'][0]


    def eval_model_by_train(self, model):
        """
            Eval self.model by local training dataset
        """
        return test_by_data_set(model,
                               self._train_loader,
                               self.args.dev_device,
                               self.args.test_verbose)['MulticlassAccuracy'][0]
