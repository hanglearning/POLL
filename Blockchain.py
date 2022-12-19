from Block import Block
import copy
from hashlib import sha256

class Blockchain:

    def __init__(self):
        self.chain = []
  
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

    def get_last_block_hash(self):
        if len(self.chain) > 0:
            return self.get_last_block().compute_hash()
        else:
            return None

    def replace_chain(self, chain):
        self.chain = copy.copy(chain)
    
    def drop_block(self):
        self.chain.pop()