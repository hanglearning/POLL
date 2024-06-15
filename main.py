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

from pathlib import Path


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
parser.add_argument('--log_dir', type=str, default="./logs")


####################### federated learning setting #######################
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="mnist")
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--comm_rounds', type=int, default=25)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")
parser.add_argument('--n_samples', type=int, default=20)
parser.add_argument('--n_class', type=int, default=3)
parser.add_argument('--n_malicious', type=int, default=8, help="number of malicious nodes in the network")

# pruning type
parser.add_argument('--CELL', type=int, default=1)
parser.add_argument('--overlapping_prune', type=int, default=0)

# for CELL
parser.add_argument('--eita', type=float, default=0.5,
                    help="accuracy threshold")
parser.add_argument('--alpha', type=float, default=0.5,
                    help="accuracy reduction factor")

parser.add_argument('--mal_vs', type=int, default=0, help="malicious validators will disturb votes (or later add randomly drop legitimate worker transactions)")
parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
parser.add_argument('--rate_unbalance', type=float, default=1.0, help='unbalance between labels')
parser.add_argument('--dataloader_workers', type=int, default=0, help='num of pytorch dataloader workers')
parser.add_argument('--prune_threshold', type=float, default=0.8)

####################### validation and rewards setting #######################

parser.add_argument('--pass_all_models', type=int, default=0, help='turn off validation and pass all models, typically used for debug or create baseline with all legitimate models')

parser.add_argument('--oppo_v', type=int, default=1, help="opportunistic validator - simulate that when a device sees its stake top 1, choose its role as validator")

parser.add_argument('--overlapping_threshold', type=float, default=0.2, help='check this percent of top overlapping ragion')
parser.add_argument('--check_whole', type=int, default=1, help='check the whole network for overlapping_threshold or unpruned region. checking the whole network makes pruning faster')
parser.add_argument('--reward', type=int, default=10, help='basic reward for identified benigh workers')

####################### blockchained pruning setting #######################
parser.add_argument('--target_pruned_perc', type=float, default=0.8)
parser.add_argument('--prune_step', type=float, default=0.2, help='increment of difficulty every diff_freq')

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

# for debug
parser.add_argument('--save_data_loaders', type=int, default=0)
parser.add_argument('--save_intermediate_models', type=int, default=0)
parser.add_argument('--save_full_local_models', type=int, default=0)


args = parser.parse_args()

def set_seed(seed):
    seed_everything(seed, workers=True)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main(): 

    set_seed(args.seed)
    
    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.dev_device}")

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_root_name = f"malvs_{args.mal_vs}_seed_{args.seed}_{exe_date_time}"

    try:
        # on Google Drive     
        import google.colab
        args.log_dir = f"/content/drive/MyDrive/POLL_BLKC/{log_root_name}"
    except:
        # local
        args.log_dir = f"{args.log_dir}/{log_root_name}"
    os.makedirs(args.log_dir)
    print(f"Model weights saved at {args.log_dir}.")

    ######## setup wandb ########
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_username)
    wandb.run.name = log_root_name
    wandb.config.update(args)
    
    ######## initiate devices ########
    init_global_model = create_model(cls=models[args.dataset]
                         [args.arch], device=args.dev_device)
    
    # save init_global_model weights
    model_save_path = f"{args.log_dir}/models_weights"
    Path(model_save_path).mkdir(parents=True, exist_ok=True)
    trainable_model_weights = get_trainable_model_weights(init_global_model)
    init_global_model_path = f"{model_save_path}/R0.pkl"
    with open(init_global_model_path, 'wb') as f:
        pickle.dump(trainable_model_weights, f)
    
    train_loaders, test_loaders, user_labels, global_test_loader = DataLoaders(n_devices=args.n_devices,
    dataset_name=args.dataset,
    n_class=args.n_class,
    nsamples=args.n_samples,
    log_dirpath=args.log_dir,
    mode=args.dataset_mode,
    batch_size=args.batch_size,
    rate_unbalance=args.rate_unbalance,
    dataloader_workers=args.dataloader_workers)
    
    idx_to_device = {}
    n_malicious = args.n_malicious
    for i in range(args.n_devices):
        is_malicious = True if args.n_devices - i <= n_malicious else False
        device = Device(i + 1, is_malicious, args, train_loaders[i], test_loaders[i], user_labels[i], global_test_loader, init_global_model, init_global_model_path)
        if is_malicious:
            print(f"Assigned device {i + 1} malicious.")
        idx_to_device[i + 1] = device
    
    devices_list = list(idx_to_device.values())
    for device in devices_list:
        device.assign_peers(idx_to_device)
    
    malicious_block_record = []
    malicious_winning_count = 0
    ######## Fed-POLL ########

    # TODO - test with local epoch increasing, the relationship between local acc and overlapping ratio, then determine how validator should be rewarded
    for comm_round in range(1, args.comm_rounds + 1):
        
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
        
        # assign devices with top stake to opportunistic validators
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
            device._verified_worker_txs = {}
            device._final_ticket_model = None
            device.produced_block = None
            device._iden_benigh_workers = None
            device._worker_to_reward = {}
            
        ''' device starts Fed-POLL '''
        ### worker starts learning and pruning ###
        for worker_iter in range(len(online_workers)):
            worker = online_workers[worker_iter]
            # resync chain
            if worker.resync_chain(comm_round, idx_to_device, online_devices_list, online_validators):
                worker.post_resync()
            # perform CELL ticket learning
            worker.ticket_learning(comm_round)
            # make tx
            worker.make_worker_tx()
            # broadcast tx to the network (of validators)
            worker.broadcast_tx(online_devices_list)
            
        ### validators validate models ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # resync chain
            if validator.resync_chain(comm_round, idx_to_device, online_devices_list, online_validators):
                validator.post_resync()
            # verify tx signature
            validator.receive_and_verify_worker_tx_sig()
            # validate model based on top-overlapping ratio
            validator.validate_models(comm_round, idx_to_device)
        
        ### validators perform FedAvg and produce blocks ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # validator produces global ticket model
            validator.produce_global_model_and_reward()
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
            device.append_and_process_block(winning_block, comm_round)
            # check performance of the validation mechanism
            # device.check_validation_performance(winning_block, idx_to_device, comm_round)
        
        ### all devices test latest models ###
        benigh_device_accs = []
        if comm_round == 1 or comm_round % args.log_model_acc_freq == 0:
            # this process is slow, so added frequency control
            for device in devices_list:
                acc = device.test_indi_accuracy(comm_round)
                if not device._is_malicious:
                    benigh_device_accs.append(acc)
        wandb.log({"comm_round": comm_round, "avg_acc_benigh_devices": np.mean(benigh_device_accs)})
            # device.test_global_accuracy(comm_round)

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
                   malicious_winning_count += 1
                   with open(f'{args.log_dir}/malicious_winning_record.txt', 'a') as f:
                    f.write(f'{comm_round}\n')
                   break
        malicious_block_record.append([comm_round, malicious_block])
            
        #     print(device.idx, "pruned_amount", round(get_pruned_amount_by_weights(device.model), 2))
        #     print(f"Length: {device.blockchain.get_chain_length()}")
    
    print(f"{malicious_winning_count}/{comm_round} times malicious device won a block.")
    with open(f'{args.log_dir}/malicious_winning_record.txt', 'a') as f:
        f.write(f'Total times: malicious_winning_count/{comm_round}\n')
    malicious_block_record = wandb.Table(data=malicious_block_record, columns = ["comm_round", "malicious_block"])
    wandb.log({log_root_name : wandb.plot.scatter(malicious_block_record, "comm_round", "malicious_block", title="Rounds that Malicious Device Wins")})

        

if __name__ == "__main__":
    main()