"""Mock linkup module for testing"""
from collections import UserDict

class OperableMapping(UserDict):
    """Simple mock of linkup.base.OperableMapping"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def __truediv__(self, other):
        """Allow division operator"""
        if isinstance(other, dict):
            return OperableMapping({k: self.get(k, 0) / other.get(k, 1) for k in set(self.keys()) | set(other.keys())})
        return OperableMapping({k: v / other for k, v in self.items()})

def map_op_val(*args, **kwargs):
    """Mock function"""
    pass

def key_aligned_val_op_with_forced_defaults(*args, **kwargs):
    """Mock function"""
    pass

def key_aligned_val_op(*args, **kwargs):
    """Mock function"""
    pass
