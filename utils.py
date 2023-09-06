from .common.custom_type import *
from typing import Sequence
from .common.const import *

def integer_squareroot(n: uint64) -> uint64:
    """
    Return the largest integer ``x`` such that ``x**2 <= n``.
    """
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

def xor(bytes_1: Bytes32, bytes_2: Bytes32) -> Bytes32:
    """
    Return the exclusive-or of two 32-byte strings.
    """
    return Bytes32(a ^ b for a, b in zip(bytes_1, bytes_2))

def uint_to_bytes(n: int) -> bytes:
    return int.to_bytes(n, 8, ENDIANNESS)

def bytes_to_uint64(data: bytes) -> uint64:
    """
    Return the integer deserialization of ``data`` interpreted as ``ENDIANNESS``-endian.
    """
    return uint64(int.from_bytes(data, ENDIANNESS))

def saturating_sub(a: int, b: int) -> int:
    """
    Computes a - b, saturating at numeric bounds.
    """
    return a - b if a > b else 0


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

