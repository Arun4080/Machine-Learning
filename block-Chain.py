import datetime
import hashlib

class Block:
    blockNo = 0
    date = None
    next = None
    hash = None
    nonce = 0
    previous_hash = 0*0
    timestamp = datetime.datetime.now()

    def __init__(self, deta):
        self.data = data