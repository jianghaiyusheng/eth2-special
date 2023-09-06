class uint8(int):
    pass

class uint16(int):
    pass

class uint32(int):
    pass

class uint64(int):
    pass

class Bytes4(bytes):
    pass

class Bytes32(bytes):
    pass

class Bytes48(bytes):
    pass

class Bytes96(bytes):
    pass


class Slot(uint64):
    pass

class Epoch(uint64):
    pass

class CommitteeIndex(uint64):
    pass

class ValidatorIndex(uint64):
    pass

class Gwei(uint64):
    pass

class Root(Bytes32):
    pass

class Hash32(Bytes32):
    pass

class Version(Bytes4):
    pass

class DomainType(Bytes4):
    pass

class ForkDigest(Bytes4):
    pass

class Domain(Bytes32):
    pass

class BLSPubkey(Bytes48):
    pass

class BLSSignature(Bytes96):
    pass


class SSZSerializable:
    pass
class SSZObject:
    pass