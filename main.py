# TODO before the paper version
# TODO - write model signature function
# TODO - check pruning difficulty change and reinit correctly

''' wandb.log()
1. log latest model accuracy in test_accuracy()
2. log validation mechanism performance in check_validation_performance()
3. log forking event at the end of main.py
4. log stake book at the end of main.py
5. log when a malicious block has been added by any device in the network 
'''
import os
# from this import d
import torch
import argparse
import pickle
from datetime import datetime
from pytorch_lightning import seed_everything
from Blockchain import Blockchain
from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP
from device import Device
from util import *
import wandb
from dataset.datasource import DataLoaders
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import random
from copy import copy

import warnings
warnings.filterwarnings("ignore")

''' abbreviations:
    tx: transaction
'''

models = {
    'cifar10': {
        'cnn': CIFAR_CNN,
        'mlp': CIFAR_MLP
    },
    'mnist': {
        'cnn': MNIST_CNN,
        'mlp': MNIST_MLP
    }
}

parser = argparse.ArgumentParser(description='POLL - Proof Of Lottery Learning')

####################### system setting #######################
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--log_dir', type=str, default="./logs")
parser.add_argument('--train_verbose', type=bool, default=False)
parser.add_argument('--test_verbose', type=bool, default=False)
parser.add_argument('--prune_verbose', type=bool, default=False)
parser.add_argument('--resync_verbose', type=bool, default=True)
parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--wandb_username', type=str, default=None)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--run_note', type=str, default=None)
parser.add_argument('--debug_validation', type=int, default=1, help='show validation process detail')

####################### federated learning setting #######################
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="mnist")
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--comm_rounds', type=int, default=40)
parser.add_argument('--frac_devices_per_round', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default="SGD", help="SGD|Adam")
parser.add_argument('--n_samples', type=int, default=20)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--n_malicious', type=int, default=0, help="number of malicious nodes in the network")
parser.add_argument('--malicious_validators', type=int, default=0, help="malicious validators will disturb votes or randomly drop legitimate lotter transactions")
parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
# below for DataLoaders
parser.add_argument('--rate_unbalance', type=float, default=1.0)
parser.add_argument('--num_workers', type=int, default=0)
# above for DataLoaders
parser.add_argument('--pass_all_models', type=int, default=0, help='turn off validation and pass all models, typically used for debug')

####################### blockchained pruning setting #######################
parser.add_argument('--target_spar', type=float, default=0.8)
parser.add_argument('--diff_base', type=float, default=0.0, help='start pruning difficulty')
parser.add_argument('--diff_incre', type=float, default=0.2, help='increment of difficulty every diff_freq')
parser.add_argument('--diff_freq', type=int, default=2, help='difficulty increased by diff_incre every diff_freq rounds')
# parser.add_argument('--warm_mask', type=int, default=1, help='warm mask as a new comer')

####################### blockchain setting #######################
parser.add_argument('--n_devices', type=int, default=6)
parser.add_argument('--lotter_reward', type=int, default=10)
parser.add_argument('--validator_reward', type=int, default=8)
parser.add_argument('--win_val_reward', type=int, default=15) # generally, we encourage being a lotter, but being a validator you can hit the jackpot to be the winning validator and gain the most rewards.

# parser.add_argument('--validator_reward_punishment', type=int, default=5, help="if an unsed validator tx found being used in block, cut the winning validator's reward by this much, incrementally")
# parser.add_argument('--block_drop_threshold', type=float, default=0.5, help="if this portion of positively voted lotter txes have invalid model_sigs, the block will be dropped")
parser.add_argument('--n_lotters', type=str, default='3', 
                    help='The number of validators is determined by this number and --n_devices. If input * to this argument, num of lotters and validators are random from round to round')
parser.add_argument('--validator_portion', type=float, default=0.5,
                    help='this determins how many validators should one lotter send txs to. e.g., there are 6 validators in the network and validator_portion = 0.5, then one lotter will send tx to 6*0.5=3 validators')
parser.add_argument('--check_signature', type=int, default=0, 
                    help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('--network_stability', type=float, default=1.0, 
                    help='the odds a device can be reached')
# parser.add_argument('--kick_out_rounds', type=int, default=6, 
#                     help='if a lotter reaches this many of rounds of negative votes, its model will not be considered for averaging') # TODO- change to consecutive. Actually, gave up this idea as this is public chain


args = parser.parse_args()

def main(): 

    seed_everything(seed=args.seed, workers=True)
    
    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ######## setup wandb ########
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_username)
    wandb.run.name = datetime.now().strftime(f"los_{args.n_lotters}_vas_{args.n_devices - int(args.n_lotters)}_mali_{args.n_malicious}_inc_{args.diff_incre}_freq_{args.diff_freq}_{args.run_note}_%m%d%Y_%H%M%S")
    wandb.config.update(args)
    
    ######## initiate devices ########
    init_global_model = create_model_no_prune(cls=models[args.dataset]
                         [args.arch], device=args.dev_device)
    
    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(n_devices=args.n_devices,
    dataset_name=args.dataset,
    n_class=args.n_class,
    nsamples=args.n_samples,
    mode=args.dataset_mode,
    batch_size=args.batch_size,
    rate_unbalance=args.rate_unbalance,
    num_workers=args.num_workers)
    
    idx_to_device = {}
    n_malicious = args.n_malicious
    for i in range(args.n_devices):
        is_malicious = True if n_malicious > 0 else False
        device = Device(i + 1, is_malicious, args, train_loaders[i], test_loaders[i], user_labels[i], global_test_loader, init_global_model)
        idx_to_device[i + 1] = device
        n_malicious -= 1
    
    devices_list = list(idx_to_device.values())
    for device in devices_list:
        device.assign_peers(idx_to_device)
    
    ######## Fed-POLL ########
    for comm_round in range(1, args.comm_rounds + 1):
        
        pruning_diff = round(min(args.target_spar, args.diff_base + (comm_round - 1) // args.diff_freq * args.diff_incre), 2)
        text = f'Comm Round {comm_round}, Pruning Diff {pruning_diff}'
        print(f"{len(text) * '='}\n{text}\n{len(text) * '='}")
        
        wandb.log({"comm_round": comm_round, "pruning_diff": pruning_diff})
        
        ''' device assign roles '''
        if args.n_lotters == '*':
            n_lotters = random.randint(0, args.n_devices - 1)
        else:
            n_lotters = int(args.n_lotters)
                    
        random.shuffle(devices_list)
        online_lotters = []
        online_validators = []
        for device_iter in range(len(devices_list)):
            if device_iter < n_lotters:
                devices_list[device_iter].role = 'lotter'
                if devices_list[device_iter].is_online():
                    online_lotters.append(devices_list[device_iter])
            else:
                devices_list[device_iter].role = 'validator'
                if devices_list[device_iter].is_online():
                    online_validators.append(devices_list[device_iter])
        
        online_devices_list = online_lotters + online_validators
        
        ''' reinit params '''
        for device in online_devices_list:
            device._received_blocks = []
            device.has_appended_block = False
            # lotters
            device._associated_validators = set()
            # validators
            device._validator_txs = []
            device._associated_lotters = set()
            device._verified_lotter_txs = {}
            device._neg_voted_txes = {}
            device._received_validator_txs = {}
            device._verified_validator_txs = set()
            device._final_ticket_model = None
            device.produced_block = None
            #device._final_models_signatures = set()
            device._dup_pos_voted_txes = {}
            
        ''' device starts Fed-POLL '''
        ### lotter starts learning and pruning ###
        for lotter_iter in range(len(online_lotters)):
            lotter = online_lotters[lotter_iter]
            # resync chain
            if lotter.resync_chain(comm_round, idx_to_device, online_devices_list, online_validators):
                lotter.post_resync()
            # perform regular ticket learning
            lotter.ticket_learning(comm_round)
            # create model signature
            # lotter.create_model_sig()
            # make tx
            lotter.make_lotter_tx()
            # associate with validators
            lotter.asso_validators(online_validators)
            
        ### validators validate models and broadcast transations ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # resync chain
            if validator.resync_chain(comm_round, idx_to_device, online_devices_list, online_validators):
                validator.post_resync()
            # verify tx signature
            validator.receive_and_verify_lotter_tx_sig()
            # validate model accuracy and form voting tx
            validator.validate_models_and_init_validator_tx(idx_to_device)
        
        ### validators perform FedAvg and produce blocks ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # validate exchange tx and validation results
            validator.exchange_and_verify_validator_tx(online_validators)
            # validator produces global ticket model
            validator.produce_global_model()
            # validator produce block
            block = validator.produce_block()
            # validator broadcasts block
            validator.broadcast_block(online_devices_list, block)
            
        ### all ONLINE devices process received blocks ###
        for device in online_devices_list:
            # pick winning block based on PoS
            winning_block = device.pick_wining_block(idx_to_device)
            if not winning_block:
                # no winning_block found, perform chain_resync next round
                continue
            # check block
            if not device.check_block_when_appending(winning_block):
                # block check failed, perform chain_resync next round
                continue
            # append and process block
            device.append_and_process_block(winning_block)
            # check performance of the validation mechanism
            device.check_validation_performance(winning_block, idx_to_device, comm_round)
        
        ### all devices test latest models ###

        for device in devices_list:
            device.test_accuracy(comm_round)

        # import pdb
        # pdb.set_trace()


        ### record forking events ###
        forking = 0
        if len(set([d.blockchain.get_last_block().produced_by for d in online_devices_list])) != 1:
            forking = 1
        wandb.log({"comm_round": comm_round, "forking_event": forking})

        ### record stake book ###
        for device in devices_list:
            to_log = {}
            to_log["comm_round"] = comm_round
            to_log[f"{device.idx}_stake_book"] = device.stake_book
            wandb.log(to_log)

        ### record when malicious validator produced a block in network ###
        malicious_block = 0
        for device in online_devices_list:
            if device.has_appended_block:
                block_produced_by = device.blockchain.get_last_block().produced_by
                if idx_to_device[block_produced_by]._is_malicious:
                   malicious_block = 1
                   break
        wandb.log({"comm_round": comm_round, "malicious_block": malicious_block})
            
        #     print(device.idx, "pruned_amount", round(get_pruned_amount_by_weights(device.model), 2))
        #     print(f"Length: {device.blockchain.get_chain_length()}")
        
        

if __name__ == "__main__":
    main()