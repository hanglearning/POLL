# TODO before the paper version
# TODO - write model signature function
# TODO - check pruning difficulty change and reinit correctly

# TODO - EARLY stop if target pruning rate and accuracy reached
# TODO - prevent free or lazy rider - 

''' wandb.log()
1. log latest model accuracy in test_indi_accuracy()
2. log validation mechanism performance in check_validation_performance()
3. log forking event at the end of main.py
4. log pouw book at the end of main.py
5. log when a malicious block has been added by any device in the network 
'''
import os
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
from collections import defaultdict
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
parser.add_argument('--log_model_acc_freq', type=int, default=1, help="frequency of logging global model's individual and global test accuracy")
parser.add_argument('--log_dir', type=str, default="./logs")


####################### federated learning setting #######################
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="mnist")
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--comm_rounds', type=int, default=25)
parser.add_argument('--epochs', type=int, default=500, help="local max training epochs to get the max accuracy")
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

parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
parser.add_argument('--rate_unbalance', type=float, default=1.0, help='unbalance between labels')
parser.add_argument('--dataloader_workers', type=int, default=0, help='num of pytorch dataloader workers')
parser.add_argument('--prune_threshold', type=float, default=0.8)

####################### validation and rewards setting #######################

parser.add_argument('--pass_all_models', type=int, default=0, help='turn off validation and pass all models, typically used for debug or create baseline with all legitimate models')
parser.add_argument('--validate_center_threshold', type=float, default=0.1, help='only recognize malicious devices if the difference of two centers of KMeans exceed this threshold')

# parser.add_argument('--oppo_v', type=int, default=1, help="opportunistic validator - simulate that when a device sees its pouw top 1, choose its role as validator")

# parser.add_argument('--overlapping_threshold', type=float, default=0.2, help='check this percent of top overlapping ragion')
# parser.add_argument('--check_whole', type=int, default=1, help='check the whole network for overlapping_threshold or unpruned region. checking the whole network makes pruning faster')
# parser.add_argument('--reward', type=int, default=10, help='basic reward for identified benigh workers')

####################### pruning setting #######################
parser.add_argument('--target_pruned_sparsity', type=float, default=0.2)
parser.add_argument('--prune_step', type=float, default=0.05, help='increment of pruning step')
# parser.add_argument('--max_prune_step', type=float, default=0.2) # otherwise, some devices may prune too aggressively
parser.add_argument('--rewind', type=int, default=0, help="reinit ticket model parameters before training")
parser.add_argument('--prune_acc_drop_threshold', type=float, default=0.05, help='if the accuracy drop is larger than this threshold, stop prunning')


####################### blockchain setting #######################
parser.add_argument('--n_devices', type=int, default=10)
parser.add_argument('--n_validators', type=str, default='*', 
                    help='if input * to this argument, the number of validators is random from round to round')
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
    log_root_name = f"seed_{args.seed}_{exe_date_time}"

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
    # wandb.init(project=args.wandb_project, entity=args.wandb_username)
    wandb.init(mode="disabled")
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
                    
        random.shuffle(devices_list)
        # winning validator cannot be a validator in the next round
        # 1. May result in pouw monopoly
        # 2. since chain resyncing chooses the highest pouw holding validator, if the winning validator is compromised, the attacker controls the whole chain

        ''' find online devices '''
        # also, devices exit the network if reaching target_pruned_sparsity
        init_online_devices = [device for device in devices_list if device.is_online()]
        
        ''' reset params '''
        for device in init_online_devices:
            device._received_blocks = {}
            device.has_appended_block = False
            # workers
            device._associated_validators = set()
            device.layer_to_model_sig_row = {}
            device.layer_to_model_sig_col = {}
            # validators
            device._verified_worker_txs = {}
            device._final_global_model = None
            device.final_ticket_model = None
            device.produced_block = None            
            device.benigh_worker_to_acc = {}
            device.malicious_worker_to_acc = {}
            device._device_to_ungranted_pouw = {}
            device.worker_to_model_sig = {}
            
        ''' device starts Fed-POLL '''
        # all online devices become workers in this phase
        online_workers = []
        for device in init_online_devices:
            device.role = 'worker'
            online_workers.append(device)

        ### worker starts learning and pruning ###
        for worker_iter in range(len(online_workers)):
            worker = online_workers[worker_iter]
            # resync chain
            if worker.resync_chain(comm_round, idx_to_device, init_online_devices):
                worker.post_resync()
            # perform training
            worker.model_learning_max(comm_round)
            # perform pruning
            worker.prune_model(comm_round)
            # generate model signature
            worker.generate_model_sig()
            # make tx
            worker.make_worker_tx()
            # broadcast tx to the network
            worker.broadcast_worker_tx(init_online_devices)

        ### validators validate models ###

        # workers volunteer to become validators
        if args.n_validators == '*':
            n_validators = random.randint(1, len(online_workers))
        else:
            n_validators = int(args.n_validators)

        online_validators = []
        random.shuffle(online_workers)
        # for worker in online_workers:
        #     if worker.is_online():
        #         if comm_round == 1 and n_validators > 0:
        #             worker.role = 'validator'
        #             online_validators.append(worker)
        #             n_validators -= 1
        #         else:
        #             if n_validators > 0 and worker.blockchain.get_last_block().produced_by != worker.idx: 
        #                 online_validators.append(worker)
        #                 n_validators -= 1

        for worker in online_workers:
            if worker.is_online() and n_validators > 0:
                if comm_round == 1 or worker.blockchain.get_last_block().produced_by != worker.idx: # by rule, a winning validator cannot be a validator in the next round
                    worker.role = 'validator'
                    online_validators.append(worker)
                    n_validators -= 1
            
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # resync chain
            if validator.resync_chain(comm_round, idx_to_device, init_online_devices):
                validator.post_resync()
            # verify worker tx signature
            validator.receive_and_verify_worker_tx_sig(online_workers)
            # validate model based on euclidean distance and accuracy
            validator.validate_models(comm_round, idx_to_device)
            # make validator transaction
            validator.make_validator_tx()
            # broadcast tx to all the validators
            validator.broadcast_validator_tx(online_validators)

        
        ### validators perform FedAvg and produce blocks ###
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # verify validator tx signature
            validator.receive_and_verify_validator_tx_sig(online_validators)
            # validator produces global model
            validator.produce_global_model_and_reward(comm_round)
            # validator produce block
            block = validator.produce_block()
            # validator broadcasts block
            validator.broadcast_block(validator.idx, online_workers, block)
        
        ### all ONLINE devices process received blocks ###
        for device in online_workers:
            # pick winning block based on PoUW
            winning_block = device.pick_wining_block(idx_to_device)
            if not winning_block:
                # no winning_block found, perform chain_resync next round
                continue
            # check block
            if not device.verify_block(winning_block):
                # block check failed, perform chain_resync next round
                continue
            # append and process block
            device.append_and_process_block(winning_block, comm_round)
            # check performance of the validation mechanism
            # device.check_validation_performance(winning_block, idx_to_device, comm_round)

        ### record forking events ###
        forking = 0
        if len(set([d.blockchain.get_last_block().produced_by for d in online_workers])) != 1:
            forking = 1
        wandb.log({"comm_round": comm_round, "forking_event": forking})

        ### record pouw book ###
        for device in devices_list:
            to_log = {}
            to_log["comm_round"] = comm_round
            to_log[f"{device.idx}_pouw_book"] = device._pouw_book
            wandb.log(to_log)

        ### record when malicious validator produced a winning block in network ###
        malicious_block = 0
        for device in online_workers:
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