from torch.utils.data.sampler import BatchSampler, SequentialSampler, RandomSampler
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union, Sized, Tuple
import hashlib
import torch
import random

class TensorSocketSampler(BatchSampler):
    def __init__(self, data_source: Sized, batch_size: int, drop_last: bool = False, shuffle: bool = True, seed: Optional[int] = None):
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # Choose the appropriate sampler based on shuffle argument
        if shuffle:
            sampler = RandomSampler(data_source)
        else:
            sampler = SequentialSampler(data_source)
        
        # Initialize the base BatchSampler with the chosen sampler
        super().__init__(sampler, batch_size, drop_last)
        
    def __iter__(self) -> Iterator[Tuple[str, List[int]]]:
             # Generate batches from the underlying sampler
        for batch_indices in super().__iter__():
            # Generate a unique ID for the batch using a hash of the indices
            batch_id = hashlib.md5(str(batch_indices).encode()).hexdigest()
            yield (batch_id, batch_indices)

    def __len__(self) -> int:
        return super().__len__()

# Example usage
if __name__ == "__main__":
    from torch.utils.data import RandomSampler

    # Example dataset size
    dataset_size = 100

    # Example usage of BatchSamplerWithID with shuffling
    batch_sampler_with_id = TensorSocketSampler(data_source=range(dataset_size), batch_size=10, drop_last=False, shuffle=True, seed=42)

    # Iterate over batches and print batch IDs and indices
    for batch_id, batch_indices in batch_sampler_with_id:
        print(f"Batch ID: {batch_id}, Batch Indices: {batch_indices}")
