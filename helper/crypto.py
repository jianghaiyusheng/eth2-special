from copy import copy
from  ..common.custom_type import *
from typing import Sequence

G2_POINT_AT_INFINITY = BLSSignature(b'\xc0' + b'\x00' * 95)

def hash(data: bytes) -> Bytes32:
    pass

# a function for hashing objects into a single root by utilizing a hash tree structure, as defined in the SSZ spec.
def hash_tree_root(object: SSZSerializable) -> Root:
    pass

def eth_aggregate_pubkeys(pubkeys: Sequence[BLSPubkey]) -> BLSPubkey:
    """
    Return the aggregate public key for the public keys in ``pubkeys``.

    NOTE: the ``+`` operation should be interpreted as elliptic curve point addition, which takes as input
    elliptic curve points that must be decoded from the input ``BLSPubkey``s.
    This implementation is for demonstrative purposes only and ignores encoding/decoding concerns.
    Refer to the BLS signature draft standard for more information.
    """
    assert len(pubkeys) > 0
    # Ensure that the given inputs are valid pubkeys
    assert all(bls.KeyValidate(pubkey) for pubkey in pubkeys)

    result = copy(pubkeys[0])
    for pubkey in pubkeys[1:]:
        result += pubkey
    return result

def eth_fast_aggregate_verify(pubkeys: Sequence[BLSPubkey], message: Bytes32, signature: BLSSignature) -> bool:
    """
    Wrapper to ``bls.FastAggregateVerify`` accepting the ``G2_POINT_AT_INFINITY`` signature when ``pubkeys`` is empty.
    """
    if len(pubkeys) == 0 and signature == G2_POINT_AT_INFINITY:
        return True
    return bls.FastAggregateVerify(pubkeys, message, signature)


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