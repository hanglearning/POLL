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
        self._resync_to = idx # record the last round's picked winning validator to resync chain, default to itself
        
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
        self._device_to_ungranted_uw = {}
        self.worker_to_model_sig = {}
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

    def flip_labels(self):
        # Access the underlying dataset
        self._train_loader.dataset.targets = 9 - self._train_loader.dataset.targets

    def model_learning_max(self, comm_round):

        wandb.log({f"{self.idx}_{self._user_labels}_global_test_acc": self.eval_model_by_global_test(self.model), "comm_round": comm_round})
        
        # train to max accuracy
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} {self._user_labels} Train to Max Acc Update ---------------------")

        # generate mask object in-place to make reinit and copy_model() work, in case no block was appended from last round, then no pruned model was obtained
        produce_mask_from_model(self.model)

        if comm_round > 1 and self.args.rewind:
        # reinitialize model with init_params
            source_params = dict(self.init_global_model.named_parameters())
            for name, param in self.model.named_parameters():
                param.data.copy_(source_params[name].data)

        max_epoch = self.args.epochs # 500 is arbitrary, stronger hardware can have more

        # label flipping attack
        if self._is_malicious and self.args.attack_type == 2:
            self.flip_labels()

        # lazy worker
        if self._is_malicious and self.args.attack_type == 3:
            max_epoch = math.ceil(max_epoch * 0.2)

        
        epoch = 0
        max_model_epoch = epoch

        max_model = copy_model(self.model, self.args.dev_device)

        # init max_acc as the initial global model acc on local training set
        max_acc = self.eval_model_by_train(self.model)

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
            # print(epoch + 1, acc)
            if acc > max_acc:
                # print(self.idx, "epoch", epoch + 1, acc)
                max_model = copy_model(self.model, self.args.dev_device)
                max_acc = acc
                max_model_epoch = epoch + 1

            epoch += 1


        print(f"Worker {self.idx} trained for {epoch} epochs with max training acc {max_acc} arrived at epoch {max_model_epoch}.")
        wandb.log({f"{self.idx}_{self._user_labels}_training_max_epoch": max_model_epoch, "comm_round": comm_round})

        self.model = max_model
        self.max_model_acc = max_acc

        # model poisoning attack
        if self._is_malicious and self.args.attack_type == 1:
            # poison the last local model
            self.poison_model(self.model)
            poinsoned_acc = self.eval_model_by_train(self.model)
            print(f'Poisoned accuracy: {poinsoned_acc}, decreased {self.max_model_acc - poinsoned_acc}.')
            self.max_model_acc = poinsoned_acc  
            # wandb.log({f"{self.idx}_after_poisoning_acc": poinsoned_acc, "comm_round": comm_round})  

        # self.save_model_weights_to_log(comm_round, max_model_epoch)
        wandb.log({f"{self.idx}_{self._user_labels}_trained_model_sparsity": 1 - get_pruned_amount_by_mask(self.model), "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_max_local_training_acc": self.max_model_acc, "comm_round": comm_round})
        wandb.log({f"{self.idx}_{self._user_labels}_local_test_acc": self.eval_model_by_local_test(self.model), "comm_round": comm_round})

    def worker_prune(self, comm_round):

        # model prune percentage
        init_pruned_amount = get_prune_summary(model=self.model, name='weight')['global'] # pruned_amount = 0s/total_params = 1 - sparsity
        if 1 - init_pruned_amount <= self.args.target_sparsity:
            print(f"Worker {self.idx}'s model at sparsity {1 - init_pruned_amount}, which is already <= the target sparsity. Skip post-pruning.")
            return
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Worker:{self.idx} starts pruning ---------------------")

        init_model_acc = self.eval_model_by_train(self.model)
        accs = [init_model_acc]
        # models = deque([copy_model(self.model, self.args.dev_device)]) - used if want the model with the best accuracy arrived at an intermediate pruned amount
        # print("Initial pruned model accuracy", init_model_acc)

        to_prune_amount = init_pruned_amount
        last_pruned_model = copy_model(self.model, self.args.dev_device)

        while True:
            to_prune_amount += self.args.prune_step
            pruned_model = copy_model(self.model, self.args.dev_device)
            l1_prune(model=pruned_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)
            
            model_acc = self.eval_model_by_train(pruned_model)

            # prune until the accuracy drop exceeds the threshold or below the target sparsity
            if init_model_acc - model_acc > self.args.prune_acc_drop_threshold or 1 - to_prune_amount <= self.args.target_sparsity:
                # revert to the last pruned model
                # print("pruned amount", to_prune_amount, "target_sparsity", self.args.target_sparsity)
                # print(f"init_model_acc - model_acc: {init_model_acc- model_acc} > self.args.prune_acc_drop_threshold: {self.args.prune_acc_drop_threshold}")
                self.model = copy_model(last_pruned_model, self.args.dev_device)
                self.max_model_acc = accs[-1]
                break
            
            accs.append(model_acc)
            last_pruned_model = copy_model(pruned_model, self.args.dev_device)

        after_pruned_amount = get_pruned_amount_by_mask(self.model) # to_prune_amount = 0s/total_params = 1 - sparsity
        after_pruning_acc = self.eval_model_by_train(self.model)

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
                    weight_params.add_(noise.to(self.args.dev_device))
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
    
    def broadcast_worker_tx(self, online_devices_list):
        # worker broadcast tx, imagine the tx is broadcasted to all devices volunterring to become validators
        # see receive_and_verify_worker_tx_sig()
        return
   
    ''' Validators' Methods '''

    def receive_and_verify_worker_tx_sig(self, online_workers):
        for worker in online_workers:
            if worker == self:
                continue
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
            if validator == self:
                continue
            if validator.idx not in self.peers:
                continue
            self.update_peers(validator.peers)
            if self.verify_tx_sig(validator._validator_tx):
                if self.args.validation_verbose:
                    print(f"Validator {self.idx} has received and verified the signature of the tx from validator {validator.idx}.")
                self._verified_validator_txs[validator.idx] = validator._validator_tx
            else:
                print(f"Signature of tx from worker {validator['idx']} is invalid.")

    # validate model by comparing euclidean distance of the unpruned weights
    def validate_models(self):
        # at this moment, both models have no mask objects, and may or may not have been pruned

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
        
        # add its own model signature
        self.worker_to_model_sig[self.idx] = {'model_sig_row': self.layer_to_model_sig_row, 'model_sig_row_sig': self.sign_msg(str(self.layer_to_model_sig_row)), 'model_sig_col': self.layer_to_model_sig_col, 'model_sig_col_sig': self.sign_msg(str(self.layer_to_model_sig_col)), 'worker_rsa': self.return_rsa_pub_key()}
           
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
            worker_to_acc[widx] = self.eval_model_by_train(worker_model)

        # based on the euclidean distance, decide if the worker is benigh or not by kmeans = 2
        kmeans = KMeans(n_clusters=2, random_state=0)
        worker_eds = list(worker_to_ed.values())
        benigh_center_group = -1
        if len(worker_eds) > 1:
            kmeans.fit(np.array(worker_eds).reshape(-1,1))
            labels = list(kmeans.labels_)
            center0 = kmeans.cluster_centers_[0]
            center1 = kmeans.cluster_centers_[1]

            if center1 - center0 > self.args.validate_center_threshold:
                benigh_center_group = 0
            elif center0 - center1 > self.args.validate_center_threshold:
                benigh_center_group = 1
        
        workers_in_order = list(worker_to_ed.keys())
        for worker_iter in range(len(workers_in_order)):
            worker_idx = workers_in_order[worker_iter]
            if benigh_center_group == -1:
                # treat all workers as benigh
                self.benigh_worker_to_acc[worker_idx] = worker_to_acc[worker_idx]
            else:
                if labels[worker_iter] == benigh_center_group:
                    # malicious validator's behaviors - flip acc
                    if not self._is_malicious:
                        self.benigh_worker_to_acc[worker_idx] = worker_to_acc[worker_idx]
                    else:
                        self.malicious_worker_to_acc[worker_idx] = worker_to_acc[worker_idx]
                else:
                    if not self._is_malicious:
                        self.malicious_worker_to_acc[worker_idx] = worker_to_acc[worker_idx] 
                    else:
                        self.benigh_worker_to_acc[worker_idx] = worker_to_acc[worker_idx]

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
        return

    def produce_global_model_and_reward(self):

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

        pouw_ranks = assign_ranks(self._pouw_book)

        # aggregate votes - normalize by the rank of useful work of the validator
        worker_idx_to_votes = defaultdict(int)
        for validator_idx, validator_tx in self._verified_validator_txs.items():
            for worker_idx in validator_tx['benigh_worker_to_acc'].keys():
                worker_idx_to_votes[worker_idx] += 1 * (pouw_ranks[validator_idx] / len(pouw_ranks)) # malicious validator's voting power should be degraded
            for worker_idx in validator_tx['malicious_worker_to_acc'].keys():
                worker_idx_to_votes[worker_idx] += -1 * (pouw_ranks[validator_idx] / len(pouw_ranks))
        
        # for the vote of the validator itself, no normalization needed as an incentive
        for worker_idx in self.benigh_worker_to_acc.keys():
            worker_idx_to_votes[worker_idx] += 1
        for worker_idx in self.malicious_worker_to_acc.keys():
            worker_idx_to_votes[worker_idx] -= 1
        worker_idx_to_votes[self.idx] += 1
        
        # select local models if their votes > 0 for FedAvg, normalize acc by worker's historical pouw to avoid zero validated acc
        worker_to_acc_weight = {}
        for worker_idx, votes in worker_idx_to_votes.items():
            if worker_idx in self._verified_worker_txs and votes > 0:
                worker_acc = self.benigh_worker_to_acc[worker_idx] if worker_idx in self.benigh_worker_to_acc else self.malicious_worker_to_acc[worker_idx]
                # tanh normalize, a bit complicated
                # normalized_accuracy_weight = float(torch.tanh(torch.tensor(worker_acc + self._pouw_book[worker_idx], device=self.args.dev_device)))
                worker_to_acc_weight[worker_idx] = worker_acc + self._pouw_book[worker_idx]
                # reward the workers by self tested accuracy (not granted yet)
                self._device_to_ungranted_uw[worker_idx] = worker_acc
                # print(f"Worker {worker_idx}'s model is selected by validator {self.idx} for aggregation.")
            else:
                if worker_idx != self.idx:
                    if self.args.validation_verbose:
                        print(f"Worker {worker_idx}'s model is not selected by validator {self.idx}.")
        
        # get models for aggregation
        worker_to_model = {worker_idx: self._verified_worker_txs[worker_idx]['model'] for worker_idx in worker_to_acc_weight}

        # only preserve model signature of selected models
        self.worker_to_model_sig = {worker_idx: self.worker_to_model_sig[worker_idx] for worker_idx in worker_to_acc_weight}

        # add its own model
        worker_to_acc_weight[self.idx] = self.max_model_acc + self._pouw_book[self.idx]
        self._device_to_ungranted_uw[self.idx] = self.max_model_acc
        worker_to_model[self.idx] = self.model
        self.worker_to_model_sig[self.idx] = {'model_sig_row': self.layer_to_model_sig_row, 'model_sig_row_sig': self.sign_msg(str(self.layer_to_model_sig_row)), 'model_sig_col': self.layer_to_model_sig_col, 'model_sig_col_sig': self.sign_msg(str(self.layer_to_model_sig_col)), 'worker_rsa': self.return_rsa_pub_key()}

        # normalize weights to between 0 and 1
        worker_to_acc_weight = {worker_idx: acc/sum(worker_to_acc_weight.values()) for worker_idx, acc in worker_to_acc_weight.items()}

        # produce final global model
        self._final_global_model = weighted_fedavg(worker_to_acc_weight, worker_to_model, device=self.args.dev_device)
    
    def validator_post_prune(self): # essential to have to push model pruning, also seen as an incentive of becoming validators
        
        # create mask object in-place
        produce_mask_from_model(self._final_global_model)
        init_pruned_amount = get_prune_summary(model=self._final_global_model, name='weight')['global'] # pruned_amount = 0s/total_params = 1 - sparsity
        if 1 - init_pruned_amount <= self.args.target_sparsity:
            print(f"Validator {self.idx}'s model at sparsity {1 - init_pruned_amount}, which is already <= the target sparsity. Skip post-pruning.")
            return
        
        print()
        L_or_M = "M" if self._is_malicious else "L"
        print(f"\n---------- {L_or_M} Validator:{self.idx} post pruning ---------------------")

        init_model_acc = self.eval_model_by_train(self._final_global_model)
        accs = [init_model_acc]

        to_prune_amount = init_pruned_amount
        last_pruned_model = copy_model(self._final_global_model, self.args.dev_device)
        
        while True:
            to_prune_amount += self.args.prune_step
            pruned_model = copy_model(self._final_global_model, self.args.dev_device)
            l1_prune(model=pruned_model,
                        amount=to_prune_amount,
                        name='weight',
                        verbose=self.args.prune_verbose)

            model_acc = self.eval_model_by_train(pruned_model)

            # prune until the accuracy drop exceeds the threshold
            if init_model_acc - model_acc > self.args.prune_acc_drop_threshold or 1 - to_prune_amount <= self.args.target_sparsity:
                self._final_global_model = copy_model(last_pruned_model, self.args.dev_device)
                self.max_model_acc = accs[-1]
                break

            accs.append(model_acc)
            last_pruned_model = copy_model(pruned_model, self.args.dev_device)

        after_pruned_amount = get_pruned_amount_by_mask(self._final_global_model) # to_prune_amount = 0s/total_params = 1 - sparsity
        after_pruning_acc = self.eval_model_by_train(self._final_global_model)

        print(f"Model sparsity: {1 - after_pruned_amount:.2f}")
        print(f"Pruned model before and after accuracy: {init_model_acc:.2f}, {after_pruning_acc:.2f}")
        print(f"Pruned amount: {after_pruned_amount - init_pruned_amount:.2f}")

    def produce_block(self):
        
        def sign_block(block_to_sign):
            block_to_sign.block_signature = self.sign_msg(str(block_to_sign.__dict__))
      
        last_block_hash = self.blockchain.get_last_block_hash()

        block = Block(last_block_hash, self._final_global_model, self._device_to_ungranted_uw, self.idx, self.worker_to_model_sig, self.return_rsa_pub_key())
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
        self.peers.remove(self.idx)
        self._pouw_book = {key: 0 for key in idx_to_device.keys()}

    def update_peers(self, peers_of_other_device):
        self.peers = self.peers.union(peers_of_other_device)

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
        self._pouw_book = {idx: 0 for idx in self._pouw_book.keys()}
        for block in self.blockchain.chain:
            for idx in block.worker_to_uw:
                self._pouw_book[idx] += block.worker_to_uw[idx]
    
    def resync_chain(self, comm_round, idx_to_device, online_devices_list):
        """ 
            Return:
            - True if the chain needs to be resynced, False otherwise
        """
        if comm_round == 1 or self.role == "validator":
            # validator not applicable to resync chain
            return False
        if self._resync_to and idx_to_device[self._resync_to].is_online():
            # _resync_to specified to the last round's picked winning validator
            resync_to_device = idx_to_device[self._resync_to]
            if self.blockchain.get_last_block_hash() == resync_to_device.blockchain.get_last_block_hash():
                # if two chains are the same, no need to resync. assume the last block's hash is valid
                return False
            if self.validate_chain(resync_to_device.blockchain):
                # update chain
                self.blockchain.replace_chain(resync_to_device.blockchain.chain)
                print(f"\n{self.role} {self.idx}'s chain is resynced from last round's picked winning validator {self._resync_to}.")
                # update pouw_book
                self.recalc_useful_work()
                return True                
        online_devices_list = copy(online_devices_list)
        if self._pouw_book:
            # resync chain from online validators having the highest recorded uw
            self._pouw_book = {validator: uw for validator, uw in sorted(self._pouw_book.items(), key=lambda x: x[1], reverse=True)}
            for device_idx, uw in self._pouw_book.items():
                device = idx_to_device[device_idx]
                if device.role != "validator":
                    continue
                if device.is_online():
                    # compare chain difference, assume the last block's hash is valid
                    if self.blockchain.get_last_block_hash() == device.blockchain.get_last_block_hash():
                        return False
                    else:
                        # validate chain
                        if not self.validate_chain(device.blockchain):
                            continue
                        # update chain
                        self.blockchain.replace_chain(device.blockchain.chain)
                        print(f"\n{self.role} {self.idx}'s chain is resynced from {device.idx}, who picked {idx_to_device[device.idx].blockchain.get_last_block().produced_by}'s block.")
                        # update pouw_book
                        self.recalc_useful_work()
                        return True
        else:
            # sync chain by randomly picking a device when join in the middle
            while online_devices_list:
                picked_device = random.choice(online_devices_list)
                # validate chain
                if not self.validate_chain(picked_device.blockchain):
                    online_devices_list.remove(picked_device)
                    continue
                self.blockchain.replace_chain(picked_device.blockchain)
                self.recalc_useful_work()
                print(f"{self.idx}'s chain resynced chain from {device.idx}.")
                return True
        if self.args.resync_verbose:
            print(f"\nDevice {self.idx}'s chain not resynced.")
        return False
            
    def post_resync(self):
        # update global model from the new block
        self.model = deepcopy(self.blockchain.get_last_block().global_model)
    
    def validate_chain(self, chain_to_check):
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

    def pick_wining_block(self, idx_to_device):
        
        last_block = self.blockchain.get_last_block()
        if last_block:
            # will not choose the block from the last block's produced validator to prevent monopoly
            self._received_blocks.pop(last_block.produced_by, None)

        if not self._received_blocks:
            print(f"\n{self.idx} has not received any block. Resync chain next round.")
            return
            
        while self._received_blocks:
            # TODO - check while logic when blocks are not valid

            # when all validators have the same pouw, workers pick randomly, a validator favors its own block. This most likely happens only in the 1st comm round

            if len(set(self._pouw_book.values())) == 1:
                if self.role == 'worker':
                    picked_validator = random.choice(list(self._received_blocks.keys()))
                    picked_block = self._received_blocks[picked_validator]
                    self._received_blocks.pop(picked_validator, None)
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
                self._pouw_book = {validator: pouw for validator, pouw in sorted(self._pouw_book.items(), key=lambda x: x[1], reverse=True)}
                received_validators_to_blocks = {block.produced_by: block for block in self._received_blocks.values()}
                for validator, pouw in self._pouw_book.items():
                    if validator in received_validators_to_blocks:
                        picked_block = received_validators_to_blocks[validator]
                        winning_validator = validator
                        print(f"\n{self.role} {self.idx} ({self._user_labels}) picks {winning_validator}'s ({idx_to_device[winning_validator]._user_labels}) block.")
                        return picked_block           
        
        print(f"\n{self.idx}'s received blocks are not valid. Resync chain next round.")
        return None # all validators are in black list, resync chain

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

    def verify_winning_block(self, winning_block):
        # block signature check in pick_wining_block()
        # check last block hash match
        if not self.check_last_block_hash_match(winning_block):
            print(f"{self.role} {self.idx}'s last block hash conflicts with {winning_block.produced_by}'s block. Resync to its chain in next round.")
            self._resync_to = winning_block.produced_by
            return False
        
        ''' validate model signature to make sure the validator is performing model aggregation honestly '''
        layer_to_model_sig_row, layer_to_model_sig_col = sum_over_model_params(winning_block.global_model)
        worker_to_acc_weight = {}
        for worker_idx, _ in winning_block.worker_to_model_sig.items():
            worker_to_acc_weight[worker_idx] = winning_block.worker_to_uw[worker_idx] + self._pouw_book[worker_idx]

        # normalize weights (again) to between 0 and 1
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

        return True
        
    def append_and_process_block(self, winning_block, comm_round):

        self.blockchain.chain.append(copy(winning_block))
        
        block = self.blockchain.get_last_block()

        self._resync_to = block.produced_by # in case of offline, resync to this validator's chain

        # grant useful_work to workers
        for worker, useful_work in block.worker_to_uw.items():
            self._pouw_book[worker] += useful_work
        
        # used to record if a block is produced by a malicious device
        self.has_appended_block = True
        # update global model
        self.model = deepcopy(block.global_model)
        # save global model weights and update path
        # self.save_model_weights_to_log(comm_round, 0, global_model=True)

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
