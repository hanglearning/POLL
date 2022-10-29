# TODO before the paper version
# TODO - write model signature function
# TODO - check pruning difficulty change and reinit correctly

''' wandb.log()
1. log latest model accuracy in test_indi_accuracy()
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
parser.add_argument('--log_model_acc_freq', type=int, default=1, help='frequency of logging global model individual and global test accuracy')
parser.add_argument('-lb', '--logs_base_folder', type=str, default="/content/drive/MyDrive/POLL", help='base folder dir to store running logs')


####################### federated learning setting #######################
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="mnist")
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--comm_rounds', type=int, default=40)
parser.add_argument('--frac_devices_per_round', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")
parser.add_argument('--n_samples', type=int, default=20)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--n_malicious', type=int, default=3, help="number of malicious nodes in the network")
parser.add_argument('--mal_vs', type=int, default=0, help="malicious validators will disturb votes (or later add randomly drop legitimate worker transactions)")
parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
# below for DataLoaders
parser.add_argument('--rate_unbalance', type=float, default=1.0)
parser.add_argument('--num_workers', type=int, default=0)
# above for DataLoaders


####################### validation and rewards setting #######################

parser.add_argument('--pass_all_models', type=int, default=0, help='turn off validation and pass all models, typically used for debug or create baseline with all legitimate models')

parser.add_argument('--validation_method', type=int, default=2, help='1 - pure shapley value based, 2 - filter valuation, 3 - attack level based, 4 - greedy soup inspired')
parser.add_argument('--reward_method', type=int, default='2', help='1 - reward based on shapley acc diff, 2 - reward by individual test acc') # V1 - has to choose R1, V3 - has to choose R2, others can choose either. Used when reward, choose winining val, and resync chain 
parser.add_argument('--voting_style', type=int, default='1', help='1 - vote by 1 and 0, select top n determined by assumed_attack_level and agg_models_portion, 2 - vote by 1 and -1, select only positive votes')

parser.add_argument('--assumed_attack_level', type=float, default=0.15, help='Used in validation method 2 and 3, to determine how many models to vote 1')
parser.add_argument('--agg_models_portion', type=float, default=0.85, help='Determine how many models to use for final aggregation based on votes, usually (1 - assumed_attack_level). Used in voting_style==1')
parser.add_argument('--z_counts', type=int, default=1, help='Counts of zscores, used in standard deviation based validation (method 2)')
parser.add_argument('--vote_than_fork', type=int, default=1, help='If set to 1, validators will exchange and aggregate voting methods. If not, validator will just choose its filtered out models for aggregation and broadcast a block - the block_fork method')


parser.add_argument('--oppo_v', type=int, default=1, help="opportunistic validator - simulate that when a device sees its stake top 1, choose its role as validator")

# parser.add_argument('--v_reward_punishment', type=int, default=5, help="if an unsed validator tx found being used in block, cut the winning validator's reward by this much, incrementally")
# parser.add_argument('--block_drop_threshold', type=float, default=0.5, help="if this portion of positively voted worker txes have invalid model_sigs, the block will be dropped")



####################### blockchained pruning setting #######################
parser.add_argument('--target_spar', type=float, default=0.8)
parser.add_argument('--diff_base', type=float, default=0.0, help='start pruning difficulty')
parser.add_argument('--diff_incre', type=float, default=0.2, help='increment of difficulty every diff_freq')
parser.add_argument('--diff_freq', type=int, default=2, help='difficulty increased by diff_incre every diff_freq rounds')
# parser.add_argument('--warm_mask', type=int, default=1, help='warm mask as a new comer')

####################### blockchain setting #######################
parser.add_argument('--n_devices', type=int, default=20)

parser.add_argument('--n_workers', type=str, default='12', 
                    help='The number of validators is determined by this number and --n_devices. If input * to this argument, num of workers and validators are random from round to round')
parser.add_argument('--v_portion', type=float, default=1,
                    help='this determins how many validators should one worker send txs to. e.g., there are 6 validators in the network and v_portion = 0.5, then one worker will send tx to 6*0.5=3 validators')
parser.add_argument('--check_signature', type=int, default=0, 
                    help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('--network_stability', type=float, default=1.0, 
                    help='the odds a device can be reached')
# parser.add_argument('--kick_out_rounds', type=int, default=6, 
#                     help='if a worker reaches this many of rounds of negative votes, its model will not be considered for averaging') # TODO- change to consecutive. Actually, gave up this idea as this is public chain


args = parser.parse_args()

if args.validation_method == 1:
    args.reward_method = 1
elif args.validation_method == 3:
    args.reward_method = 2
else:
    pass

vars(args)["POS_VOTE"] = 1
if args.voting_style == 1:
    vars(args)["NEG_VOTE"] = 0
elif args.voting_style == 2:
    vars(args)["NEG_VOTE"] = 0

def main(): 

    seed_everything(seed=args.seed, workers=True)
    
    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.dev_device}")

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_dirpath = f"{args.logs_base_folder}/POLL_BLKC/{exe_date_time}"
    os.makedirs(log_dirpath)


    ######## setup wandb ########
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_username)
    wandb.run.name = f"val_{args.validation_method}_reward_{args.reward_method}_voting_{args.voting_style}_malvs_{args.mal_vs}_seed_{args.seed}_{exe_date_time}"
    wandb.config.update(args)
    
    ######## initiate devices ########
    init_global_model = create_model_no_prune(cls=models[args.dataset]
                         [args.arch], device=args.dev_device)
    
    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(n_devices=args.n_devices,
    dataset_name=args.dataset,
    n_class=args.n_class,
    nsamples=args.n_samples,
    log_dirpath=log_dirpath,
    mode=args.dataset_mode,
    batch_size=args.batch_size,
    rate_unbalance=args.rate_unbalance,
    num_workers=args.num_workers)
    
    idx_to_device = {}
    n_malicious = args.n_malicious
    for i in range(args.n_devices):
        is_malicious = True if args.n_devices - i <= n_malicious else False
        device = Device(i + 1, is_malicious, args, train_loaders[i], test_loaders[i], user_labels[i], global_test_loader, init_global_model)
        idx_to_device[i + 1] = device
    
    devices_list = list(idx_to_device.values())
    for device in devices_list:
        device.assign_peers(idx_to_device)
    
    malicious_block_record = []
    ######## Fed-POLL ########
    for comm_round in range(1, args.comm_rounds + 1):
        
        pruning_diff = round(min(args.target_spar, args.diff_base + (comm_round - 1) // args.diff_freq * args.diff_incre), 2)
        text = f'Comm Round {comm_round}, Pruning Diff {pruning_diff}'
        print(f"{len(text) * '='}\n{text}\n{len(text) * '='}")
        
        wandb.log({"comm_round": comm_round, "pruning_diff": pruning_diff})
        
        ''' device assign roles '''
        if args.n_workers == '*':
            n_workers = random.randint(0, args.n_devices - 1)
        else:
            n_workers = int(args.n_workers)
                    
        random.shuffle(devices_list)
        # winning validator cannot be a validator in the next round
        # 1. May result in stake monopoly
        # 2. since chain resyncing chooses the highest stakeholding validator, if the winning validator is compromised, the attacker controls the whole chain
        online_workers = []
        online_validators = []
        role_assign_list = copy(devices_list)
        
        # assign devices with top stake to validators
        if args.oppo_v and comm_round != 1:
            for device in role_assign_list:
                top_stake_device_idx = [idx for idx, stake in sorted(device.stake_book.items(), key=lambda item: item[1])][0]
                if device.idx == top_stake_device_idx and device.is_online():
                    online_validators.append(device)
            for oppo_validator in online_validators:
                role_assign_list.remove(oppo_validator)

        for device_iter in range(len(role_assign_list)):
            if device_iter < n_workers:
                role_assign_list[device_iter].role = 'worker'
                if role_assign_list[device_iter].is_online():
                    online_workers.append(role_assign_list[device_iter])
            else:
                role_assign_list[device_iter].role = 'validator'
                if role_assign_list[device_iter].is_online():
                    online_validators.append(role_assign_list[device_iter])
        
        online_devices_list = online_workers + online_validators
        
        ''' reinit params '''
        for device in online_devices_list:
            device._received_blocks = []
            device.has_appended_block = False
            # workers
            device._associated_validators = set()
            # validators
            device._validator_txs = []
            device._associated_workers = set()
            device._verified_worker_txs = {}
            device._unused_worker_txes = {}
            device._received_validator_txs = {}
            device._verified_validator_txs = set()
            device._final_ticket_model = None
            device.produced_block = None
            #device._final_models_signatures = set()
            device._dup_used_worker_txes = {}
            
        ''' device starts Fed-POLL '''
        ### worker starts learning and pruning ###
        for worker_iter in range(len(online_workers)):
            worker = online_workers[worker_iter]
            # resync chain
            if worker.resync_chain(comm_round, idx_to_device, online_devices_list, online_validators):
                worker.post_resync()
            # perform regular ticket learning
            worker.ticket_learning(comm_round)
            # create model signature
            # worker.create_model_sig()
            # make tx
            worker.make_worker_tx()
            # associate with validators
            worker.asso_validators(online_validators)
            
        ### validators validate models ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # resync chain
            if validator.resync_chain(comm_round, idx_to_device, online_devices_list, online_validators):
                validator.post_resync()
            # verify tx signature
            validator.receive_and_verify_worker_tx_sig()
            # validate model accuracy and form validator tx
            validator.validate_model(idx_to_device)
            #validator.shapley_value_validation_VBFL(comm_round, idx_to_device)
        
        ### validators broadcast(exchange) transations, perform FedAvg and produce blocks ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # validate exchange tx and validation results
            if args.vote_than_fork:
                validator.exchange_and_verify_validator_tx(online_validators)
            else:
                # in this mode, validators directly use its received models for FedAvg and produce block
                validator.validator_no_exchange_tx()
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
            # device.check_validation_performance(winning_block, idx_to_device, comm_round)
        
        ### all devices test latest models ###
        if comm_round == 1 or comm_round % args.log_model_acc_freq == 0:
            # this process is slow, so added frequency control
            for device in devices_list:
                device.test_indi_accuracy(comm_round)
            device.test_global_accuracy(comm_round)

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
        malicious_block_record.append([comm_round, malicious_block])
            
        #     print(device.idx, "pruned_amount", round(get_pruned_amount_by_weights(device.model), 2))
        #     print(f"Length: {device.blockchain.get_chain_length()}")

    wandb.log({"malicious_block" : wandb.plot.scatter(malicious_block_record, "comm_round", "malicious_block", title="Rounds that Malicious Device Wins")})

        

if __name__ == "__main__":
    main()