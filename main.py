import os
import torch
import argparse
import pickle
from datetime import datetime
from pytorch_lightning import seed_everything
from model.cifar10.cnn import CNN as CIFAR_CNN
from model.cifar10.mlp import MLP as CIFAR_MLP
from model.mnist.cnn import CNN as MNIST_CNN
from model.mnist.mlp import MLP as MNIST_MLP
from device import Device
from util import create_model
import wandb
from dataset.datasource import DataLoaders
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
import random

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
parser.add_argument('--resync_verbose', type=bool, default=False)
parser.add_argument('--seed', type=int, default=40)
parser.add_argument('--wandb_username', type=str, default=None)

####################### federated learning setting #######################
parser.add_argument('--dataset', help="mnist|cifar10",type=str, default="cifar10")
parser.add_argument('--arch', type=str, default='cnn', help='cnn|mlp')
parser.add_argument('--dataset_mode', type=str,default='non-iid', help='non-iid|iid')
parser.add_argument('--comm_rounds', type=int, default=50)
parser.add_argument('--frac_devices_per_round', type=float, default=1.0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_samples', type=int, default=20)
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--num_malicious', type=int, default=0, help="number of malicious nodes in the network")


####################### pruning setting #######################
parser.add_argument('--target_spar', type=float, default=0.8)
parser.add_argument('--sig_portion', type=float, default=0.25, help="portion of the subsampled weights used to construct the signature")
parser.add_argument('--sig_threshold', type=float, default=0.8, help="portion of the signature that satisfied hard requirement")


####################### blockchain setting #######################
parser.add_argument('--prune_diff', type=float, default=0.2, help='base pruning difficulty')
parser.add_argument('--diff_freq', type=int, default=1, help='difficulty increases every this-number-of rounds')
parser.add_argument('--num_devices', type=int, default=16)
parser.add_argument('--num_lotters', type=str, default='10', 
                    help='The number of validators is determined by this number and --num_devices. If input * to this argument, num of lotters and validators are random from round to round')
parser.add_argument('--validator_portion', type=int, default=0.5,
                    help='this determins how many validators should one lotter send transactions to. e.g., there are 6 validators in the network and validator_portion = 0.5, then one lotter will send transaction to 6*0.5=3 validators')
parser.add_argument('--check_signature', type=int, default=1, 
                    help='if set to 0, all signatures are assumed to be verified to save execution time')
parser.add_argument('--network_stability', type=float, default=1.0, 
                    help='the odds a device can be reached')


# unknown settings
parser.add_argument('--rate_unbalance', type=float, default=1.0)
parser.add_argument('--fast_dev_run', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()

def main(): 

    seed_everything(seed=args.seed, workers=True)
    
    args.dev_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    ######## setup wandb ########
    wandb.login()
    wandb.init(project="POLL", entity=args.wandb_username)
    wandb.run.name = datetime.now().strftime("%m%d%Y_%H%M%S")
    wandb.run.save()
    wandb.config.update(args)
    
    ######## initiate devices ########
    init_global_model = create_model(cls=models[args.dataset]
                         [args.arch], device=args.dev_device)

    train_loaders, test_loaders = DataLoaders(num_devices=args.num_devices,
                                              dataset_name=args.dataset,
                                              n_class=args.n_class,
                                              nsamples=args.n_samples,
                                              mode=args.dataset_mode,
                                              batch_size=args.batch_size,
                                              rate_unbalance=args.rate_unbalance,
                                              num_workers=args.num_workers)
    
    idx_to_device = {}
    for i in range(args.num_devices):
        device = Device(i, args, train_loaders[i], test_loaders[i], init_global_model)
        idx_to_device[i] = device
    
    devices_list = idx_to_device.values()
    for device in devices_list:
        device.assign_peers(idx_to_device)
    
    ######## Fed-POLL ########
    for comm_round in range(1, args.comm_rounds + 1):
        
        ''' device assign roles '''
        if args.num_lotters == '*':
            num_lotters = random.randint(0, args.num_devices - 1)
        else:
            num_lotters = int(args.num_lotters)
                    
        devices_list = random.shuffle(devices_list)
        lotters = []
        validators = []
        for device_iter in range(len(devices_list)):
            if device_iter < num_lotters:
                devices_list[device_iter].role = 'lotter'
                lotters.append(devices_list[device_iter])
            else:
                devices_list[device_iter].role = 'validator'
                validators.append(devices_list[device_iter])
        
        ''' reinit params '''
        # device set is_online
        for device in devices_list:
            device.set_is_online()
            device._received_blocks = []
        
        for validator in validators:
            validator._associated_lotters = set()
            validator._received_lotter_txs = set()
            validator._verified_lotter_txs = set()
            validator._lotter_idx_to_model_score = {}
            validator._received_validator_txs = set()
            validator._verified_validator_txs = set()
            validator._final_ticket_model = None
            validator._final_models_signatures = set()
            
        ''' device starts Fed-POLL '''
        ### lotter starts learning and pruning ###
        for lotter_iter in range(len(lotters)):
            lotter = lotters[lotter_iter]
            # resync chain
            lotter.resync_chain() #TODO - update global model
            if lotter._mask:
                # fresh joining, warm the mask
                print(f"Lotter ({lotter_iter+1}/{num_lotters}) is warming its mask...")
                lotter.warm_initial_mask()
            else:
                # perform regular ticket learning
                lotter.regular_ticket_learning()
            # create model signature
            lotter.create_model_sig()
            # make transaction
            lotter.make_lotter_transaction()
            # associate with validators
            lotter.asso_validators(validators)
            
        ### validator validates and perform FedAvg ###
        for validator_iter in range(len(validators)):
            validator = validators[validator_iter]
            # resync chain
            validator.resync_chain() #TODO - update global model
            # verify transaction signature
            validator.verify_lotter_tx_sig()
            # verify model_signature is within mask
            validator.verify_model_sig_positions()
            # validate model accuracy
            validator.validate_model_accuracy()
            # validator make tx
            validator.make_validator_transaction()
            # validate exchange transaction and validation results
            validator.exchange_and_verify_validator_tx(validators)
            # validator aggregate model scores and produce global ticket model
            validator.produce_global_model()
            # validator produce block
            block = validator.produce_block()
            # validator broadcasts block
            validator.broadcast_block(devices_list, block)
            
        ### all devices process received blocks ###
        for device in devices_list:
            # pick winning block based on PoS
            winning_block = device.pick_wining_block()
            if not winning_block:
                device.need_chain_resync = True
                continue
            # append block
            if not device.append_block(winning_block):
                device.need_chain_resync = True
                continue
            # process block
            device.process_block()
        

if __name__ == "__main__":
    main()