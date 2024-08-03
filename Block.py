from copy import deepcopy
import json
from hashlib import sha256
import torch

class Block:
    def __init__(self, previous_block_hash, global_model, device_to_uw, produced_by, worker_to_model_sig, validator_txs, validator_rsa_pub_key):
        self.previous_block_hash = previous_block_hash
        self.global_model = global_model
        self.device_to_uw = device_to_uw
        # validator specific
        self.produced_by = produced_by
        self.worker_to_model_sig = worker_to_model_sig
        self.validator_txs = validator_txs
        self.validator_rsa_pub_key = validator_rsa_pub_key
        self.block_signature = None

    # compute_hash() also used to return value for block verification
    # if False by default, used for pow and verification, in which pow_proof has to be None, because at this moment -
    # pow - block hash is None, so does not affect much
    # verification - the block already has its hash
    # if hash_entire_block == True -> used in set_previous_block_hash, where we need to hash the whole previous block
    def compute_hash(self):
        # need sort keys to preserve order of key value pairs
        return sha256(str(sorted(self.__dict__.items())).encode('utf-8')).hexdigest()

    def remove_signature_for_verification(self):
        self.signature = None