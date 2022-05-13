import copy
import json
from hashlib import sha256

class Block:
    def __init__(self, previous_block_hash, all_transactions,global_ticket_model, model_signatures, model_scores, produced_by, validator_rsa_pub_key):
        self.previous_block_hash = previous_block_hash
        self.all_transactions = all_transactions
        self.global_ticket_model = global_ticket_model
        self.model_signatures = model_signatures
        self.model_scores = model_scores
        # validator specific
        self.produced_by = produced_by
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

    # returners of the private attributes
    
    def get_previous_block_hash(self):
        return self.previous_block_hash
    
    def get_validator_rsa_pub_key(self):
        return self.validator_rsa_pub_key

    ''' Miner Specific '''
    def set_previous_block_hash(self, hash_to_set):
        self._previous_block_hash = hash_to_set

    def set_mined_by(self, mined_by):
        self._mined_by = mined_by
    
    def get_mined_by(self):
        return self._mined_by

    def set_signature(self, signature):
        # signed by mined_by node
        self._signature = signature

    def get_signature(self):
        return self._signature

    def set_mining_rewards(self, mining_rewards):
        self._mining_rewards = mining_rewards

    def get_mining_rewards(self):
        return self._mining_rewards
    
    def get_transactions(self):
        return self._transactions

    # a temporary workaround to free GPU mem by delete txs stored in the blocks. Not good when need to resync chain
    def free_tx(self):
        try:
            del self._transactions
        except:
            pass
