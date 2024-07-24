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
import wandb
import random
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from datetime import datetime
from pytorch_lightning import seed_everything

from device import Device
from util import *
from dataset.datasource import DataLoaders

from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP


''' abbreviations:
    tx: transaction
    uw: useful work
    pouw: proof of useful work
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

parser = argparse.ArgumentParser(description='LBFL')

####################### wandb setting #######################

parser.add_argument('--wandb_enable', type=int, default=0, help= '0 to disable logging, 1 to enable wandb logging')
parser.add_argument('--wandb_username', type=str, default=None)
parser.add_argument('--wandb_project', type=str, default=None)
parser.add_argument('--run_note', type=str, default=None)

####################### system setting #######################
parser.add_argument('--train_verbose', type=bool, default=False)
parser.add_argument('--test_verbose', type=bool, default=False)
parser.add_argument('--prune_verbose', type=bool, default=False)
parser.add_argument('--resync_verbose', type=bool, default=True)
parser.add_argument('--validation_verbose', type=int, default=0, help='show validation process detail')
parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--log_dir', type=str, default="./logs")
parser.add_argument('--peer_percent', type=float, default=1, help='this indicates the percentage of peers to assign. See assign_peers() in device.py. As the communication goes on, a device should be able to know all other devices in the network.')

####################### federated learning setting #######################
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="mnist")
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--rate_unbalance', type=float, default=1.0, help='unbalance between labels')
parser.add_argument('--dataloader_workers', type=int, default=0, help='num of pytorch dataloader workers')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--rounds', type=int, default=25)
parser.add_argument('--epochs', type=int, default=500, help="local max training epochs to get the max accuracy")
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--optimizer', type=str, default="Adam", help="SGD|Adam")
parser.add_argument('--n_samples', type=int, default=20)
parser.add_argument('--n_classes', type=int, default=3)
parser.add_argument('--n_malicious', type=int, default=8, help="number of malicious nodes in the network")

parser.add_argument('--noise_variance', type=int, default=1, help="noise variance level of the injected Gaussian Noise")
parser.add_argument('--target_acc', type=float, default=0.9, help='target accuracy for training and/or pruning, gone offline if achieved')

####################### validation and rewards setting #######################
parser.add_argument('--pass_all_models', type=int, default=0, help='turn off validation and pass all models, typically used for debug or create baseline with all legitimate models')
parser.add_argument('--validate_center_threshold', type=float, default=0.1, help='only recognize malicious devices if the difference of two centers of KMeans exceed this threshold')
parser.add_argument('--inverse_acc_weights', type=int, default=1, help='sometimes may inverse the accuracy weights to give more weights to minority workers. ideally, malicious workers should have been filtered out and not be considered here')

####################### attack setting #######################
parser.add_argument('--attack_type', type=int, default=0, help='0 - no attack, 1 - model poisoning attack, 2 - label flipping attack, 3 - lazy attack')

####################### pruning setting #######################
parser.add_argument('--rewind', type=int, default=1, help="reinit ticket model parameters before training")
parser.add_argument('--target_sparsity', type=float, default=0.1, help='target sparsity for pruning, stop pruning if below this threshold')
parser.add_argument('--prune_step', type=float, default=0.05, help='increment of pruning step')
parser.add_argument('--prune_acc_drop_threshold', type=float, default=0.05, help='if the accuracy drop is larger than this threshold, stop prunning')
parser.add_argument('--worker_prune_acc_trigger', type=float, default=0.8, help='must achieve this accuracy to trigger worker to post prune its local model')
parser.add_argument('--validator_prune_acc_trigger', type=float, default=0.8, help='must achieve this accuracy to trigger validator to post prune the global model')


####################### blockchain setting #######################
parser.add_argument('--n_devices', type=int, default=10)
parser.add_argument('--n_validators', type=str, default='*', 
                    help='if input * to this argument, the number of validators is random from round to round')
parser.add_argument('--check_signature', type=int, default=0, 
                    help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('--network_stability', type=float, default=1.0, 
                    help='the odds a device can be reached')
parser.add_argument('--malicious_always_online', type=int, default=1, 
                    help='1 - malicious devices are always online; 0 - malicious devices can be online or offline depending on network_stability')
parser.add_argument('--top_percent_winning', type=int, default=0.3, 
                    help='when picking the winning block, considering the validators having the useful work within this top percent. see pick_winning_block()')

####################### debug setting #######################
parser.add_argument('--model_save_freq', type=int, default=0, help='0 - never save, 1 - save every round, n - save every n rounds')

args = parser.parse_args()

def set_seed(seed):
    seed_everything(seed, workers=True)
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def main(): 

    set_seed(args.seed)

    if not args.attack_type:
        args.n_malicious = 0
    
    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device {args.dev_device}")

    exe_date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    log_root_name = f"LBFL_seed_{args.seed}_{exe_date_time}_rounds_{args.rounds}_epochs_{args.epochs}_val_{args.n_validators}_mal_{args.n_malicious}_attack_{args.attack_type}_noise_{args.noise_variance}_rewind_{args.rewind}_nsamples_{args.n_samples}_nclasses_{args.n_classes}"

    try:
        # on Google Colab with Google Drive mounted
        import google.colab
        args.log_dir = f"/content/drive/MyDrive/LBFL/{log_root_name}"
    except:
        # local
        args.log_dir = f"{args.log_dir}/{log_root_name}"
    os.makedirs(args.log_dir)
    print(f"Model weights saved at {args.log_dir}.")

    ######## setup wandb ########
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_username)
    if not args.wandb_enable:
        wandb.init(mode="disabled")
    wandb.run.name = f"{log_root_name}_run_note_{args.run_note}"
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
    n_classes=args.n_classes,
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
    
    ######## LBFL ########

    for comm_round in range(1, args.rounds + 1):
        
        print_text = f"Comm Round: {comm_round}"
        print()
        print("=" * len(print_text))
        print(print_text)
        print("=" * len(print_text))
        
        random.shuffle(devices_list)
        
        ''' find online devices '''
        # devices exit the network if reaching target_sparsity and target_acc
        init_online_devices = [device for device in devices_list if device.set_online()]
        if len(init_online_devices) < 2:
            print(f"Total {len(init_online_devices)} device online, skip this round.")
            continue

        wandb.log({"comm_round": comm_round, "n_online_devices": len(init_online_devices)})

        ''' reset params '''
        for device in init_online_devices:
            device._received_blocks = {}
            device.has_appended_block = False
            # workers
            device.layer_to_model_sig_row = {}
            device.layer_to_model_sig_col = {}
            device._worker_pruned_amount = 0
            # validators
            device._verified_worker_txs = {}
            device._final_global_model = None
            device.produced_block = None            
            device.benigh_worker_to_acc = {}
            device.malicious_worker_to_acc = {}
            device._device_to_ungranted_uw = {}
            device.worker_to_model_sig = {}
            device.produced_block = None
            
        ''' Device Starts LBFL '''

        ''' Phase 1 - Worker Learning and Pruning '''
        # all online devices become workers in this phase
        online_workers = []
        for device in init_online_devices:
            device.role = 'worker'
            online_workers.append(device)

        ### worker starts learning and pruning ###
        for worker_iter in range(len(online_workers)):
            worker = online_workers[worker_iter]
            # resync chain
            if worker.resync_chain(comm_round, idx_to_device):
                worker.post_resync()
            # perform training
            worker.model_learning_max(comm_round)
            # perform pruning
            worker.worker_prune(comm_round)
            # generate model signature
            worker.generate_model_sig()
            # make tx
            worker.make_worker_tx()
            # broadcast tx to the network
            worker.broadcast_worker_tx()

        print(f"\nWorkers {[worker.idx for worker in online_workers]} have broadcasted worker transactions to validators.")

        ''' Phase 2 - Validators Model Validation and Exchange Votes '''
        # workers volunteer to become validators
        if args.n_validators == '*':
            n_validators = random.randint(1, len(online_workers))
        else:
            n_validators = int(args.n_validators)
        
        print(f"\nRound {comm_round}, {n_validators} validators selected.")
        wandb.log({"comm_round": comm_round, "n_validators": n_validators})

        online_validators = []
        random.shuffle(online_workers)
        
        for worker in online_workers:
            if worker.is_online() and n_validators > 0:
                worker.role = 'validator'
                online_validators.append(worker)
                n_validators -= 1
            
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # verify worker tx signature
            validator.receive_and_verify_worker_tx_sig(online_workers)
            # validate model based on euclidean distance and accuracy
            validator.validate_models()
            # make validator transaction
            validator.make_validator_tx()
            # broadcast tx to all the validators
            validator.broadcast_validator_tx(online_validators)
        
        print(f"\nValidators {[validator.idx for validator in online_validators]} have broadcasted validator transactions to other validators.")


        ''' Phase 3 - Validators Perform FedAvg and Produce Blocks '''
        for validator_iter in range(len(online_validators)):
            validator = online_validators[validator_iter]
            # verify validator tx signature
            validator.receive_and_verify_validator_tx_sig(online_validators)
            # validator produces global model
            validator.produce_global_model_and_reward(idx_to_device, comm_round)
            # validator post prune the global model
            # validator.validator_post_prune()
            validator.validator_post_prune2()
            # validator produce block
            validator.produce_block()
            # validator broadcasts block
            validator.broadcast_block()
        print(f"\nValidators {[validator.idx for validator in online_validators]} have broadcasted their blocks to the network.")

        ''' Phase 4 - All Online Devices Process Received Blocks '''
        for device in online_workers:
            # receive blocks from validators
            device.receive_blocks(online_validators)
            # pick winning block based on PoUW
            winning_block = device.pick_winning_block(idx_to_device)
            if not winning_block:
                # no winning_block found, perform chain_resync next round
                continue
            # check block
            if not device.verify_winning_block(winning_block):
                # block check failed, perform chain_resync next round
                continue
            # append and process block
            device.append_and_process_block(winning_block, comm_round)
            # check performance of the validation mechanism
            # device.check_validation_performance(winning_block, idx_to_device, comm_round)

        ''' End of LBFL '''

        ''' Evaluation '''
        ### record forking events ###
        forking = 0
        blocks_produced_by = set()
        for device in online_workers:
            if device.has_appended_block:
                blocks_produced_by.add(device.blockchain.get_last_block().produced_by)
                if len(blocks_produced_by) > 1:
                    forking = 1
                    break
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
    
    print(f"{malicious_winning_count}/{comm_round} times malicious device won a block.")
    with open(f'{args.log_dir}/malicious_winning_record.txt', 'a') as f:
        f.write(f'Total times: malicious_winning_count/{comm_round}\n')
    malicious_block_record = wandb.Table(data=malicious_block_record, columns = ["comm_round", "malicious_block"])
    wandb.log({log_root_name : wandb.plot.scatter(malicious_block_record, "comm_round", "malicious_block", title="Rounds that Malicious Devices' Blocks Won")})

        

if __name__ == "__main__":
    main()