from Block import Block
import copy

class Blockchain:
    
    base_pruning_difficulty = 0.2

    def __init__(self):
        self.chain = []
        self.difficulty_increase_frequency = 2
  
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
        return (len(self.chain // self.difficulty_increase_frequency) + 1) * self.base_pruning_difficulty

    def get_last_block_hash(self):
        if len(self.chain) > 0:
            return self.get_last_block().compute_hash(hash_entire_block=True)
        else:
            return None

    def replace_chain(self, chain):
        self.chain = copy.copy(chain)

    def append_block(self, block):
        self.chain.append(copy.copy(block))