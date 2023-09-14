from  ..common.custom_type import *
from typing import Sequence

def hash(data: bytes) -> Bytes32:
    pass

# a function for hashing objects into a single root by utilizing a hash tree structure, as defined in the SSZ spec.
def hash_tree_root(object: SSZSerializable) -> Root:
    pass


class BLS:
    def Sign(privkey: int, message: bytes) -> BLSSignature:
        pass

    def Verify(pubkey: BLSPubkey, message: bytes, signature: BLSSignature) -> bool:
        pass

    def Aggregate(signatures: Sequence[BLSSignature]) -> BLSSignature:
        pass

    def FastAggregateVerify(pubkeys: Sequence[BLSPubkey], message: bytes, signature: BLSSignature) -> bool:
        pass

    def AggregateVerify(pubkeys: Sequence[BLSPubkey], messages: Sequence[bytes], signature: BLSSignature) -> bool:
        pass

    def KeyValidate(pubkey: BLSPubkey) -> bool:
        pass
bls = BLS()