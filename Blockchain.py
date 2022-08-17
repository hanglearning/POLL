from Block import Block
import copy
from hashlib import sha256

class Blockchain:

    def __init__(self, base_prunt_difficulty, diff_increase_amount, diff_increase_frequncy, target_pruning_rate):
        self.chain = []
        self.base_prunt_difficulty = base_prunt_difficulty
        self.diff_increase_amount = diff_increase_amount
        self.difficulty_increase_frequency = diff_increase_frequncy
        self.target_pruning_rate = target_pruning_rate
        
  
    def get_chain(self):
        return self.chain

    def get_chain_length(self):
        return len(self.chain)

    def get_last_block(self):
        if len(self.chain) > 0:
            return self.chain[-1]
        else:
            # blockchain doesn't have its genesis block
            return None

    def get_cur_pruning_diff(self):
        return round(min(self.target_pruning_rate, self.base_prunt_difficulty + (len(self.chain) // self.difficulty_increase_frequency) * self.diff_increase_amount), 2)

    def get_last_block_hash(self):
        if len(self.chain) > 0:
            return self.get_last_block().compute_hash()
        else:
            return None

    def replace_chain(self, chain):
        self.chain = copy.copy(chain)
    
    def drop_block(self):
        self.chain.pop()