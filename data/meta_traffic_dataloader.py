

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import sys
import os

# Add the path for importing local modules.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from meta_traffic_dataset import QueryTrafficDataset, SupportTrafficDataset

import torch.distributed as dist
def is_main_process():
    """Check whether this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

class MetaTrafficDataLoader:
    """
    Data loader for meta-learning.
    Combines query and support sets with an output format fully compatible with MultiMetaFingerNet.forward().
    """
    
    def __init__(self,
                 query_json_path: str,
                 query_files_dir: str,
                 support_root_dir: str,
                 activated_classes: List[int] = None,
                 support_target_length: int = 10000,
                 query_target_length: int = 20000,
                 shots_per_class: int = 1,
                 batch_size: int = 4,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 random_sampling: bool = False):
        """
        Args:
            query_json_path: Path to the query set index JSON file.
            query_files_dir: Directory containing query data files.
            support_root_dir: Root directory of the support set.
            activated_classes: List of active classes; defaults to 0-59.
            target_length: Target sequence length.
            shots_per_class: Number of support samples per class.
            batch_size: Batch size.
            shuffle: Whether to shuffle the query dataset.
            num_workers: Number of data loading workers.
            random_sampling: Whether to use random sampling mode for training.
        """
        self.activated_classes = activated_classes if activated_classes else list(range(60))  # 0-59
        self.support_target_length = support_target_length
        self.query_target_length = query_target_length
        self.shots_per_class = shots_per_class
        self.batch_size = batch_size
        self.random_sampling = random_sampling
        if is_main_process():
            print(f"Initializing MetaTrafficDataLoader...")
            print(f"  - Active classes: {len(self.activated_classes)} (0-{max(self.activated_classes)})")
            print(f"  - Samples per class: {shots_per_class}")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Random sampling: {random_sampling}")
        
        # Initialize the query dataset.
        self.query_dataset = QueryTrafficDataset(
            json_index_path=query_json_path,
            query_files_dir=query_files_dir,
            target_length=query_target_length,
            activated_classes=self.activated_classes
        )
        
        # Initialize the support dataset.
        self.support_dataset = SupportTrafficDataset(
            support_root_dir=support_root_dir,
            activated_classes=self.activated_classes,
            target_length=support_target_length,
            shots_per_class=shots_per_class,
            random_sampling=random_sampling
        )
        
        # Create the query DataLoader.
        self.query_loader = DataLoader(
            self.query_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._query_collate_fn
        )
        
        if not self.random_sampling:
            # Fixed sampling mode: preload all support set data.
            self.support_data, self.support_masks, self.class_order = self.support_dataset.get_all_support_data()
            if is_main_process():
                print(f"  - Support set shape: {self.support_data.shape}")
        else:
            # Random sampling mode: generate the support set dynamically on each iteration.
            self.class_order = sorted(self.activated_classes)
            if is_main_process():
                print(f"  - Support set: dynamic random sampling mode")
        if is_main_process():
            print(f"  - Query samples: {len(self.query_dataset)}")
            print(f"  - Data loader initialization complete!")
    
    def _query_collate_fn(self, batch):
        """Collate function for the query set."""
        query_data_list = []
        query_labels_list = []
        metadata_list = []
        
        for query_data, query_labels, metadata in batch:
            query_data_list.append(query_data)
            query_labels_list.append(query_labels)
            metadata_list.append(metadata)
        
        # Stack into a batch.
        batch_query_data = torch.stack(query_data_list)  # (batch_size, target_length)
        batch_query_labels = torch.stack(query_labels_list)  # (batch_size, num_classes)
        
        return batch_query_data, batch_query_labels, metadata_list
    
    def get_support_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get support set data.
        
        Returns:
            support_data: (num_classes, shots_per_class, target_length)
            support_masks: (num_classes, shots_per_class, target_length)
        """
        if self.random_sampling:
            # Random sampling mode: generate new random samples on each call.
            support_data, support_masks, _ = self.support_dataset.get_all_support_data()
            return support_data, support_masks
        else:
            # Fixed sampling mode: return preloaded data.
            return self.support_data, self.support_masks
    
    def __iter__(self):
        """Return the data iterator."""
        return MetaTrafficIterator(self)
    
    def __len__(self):
        """Return the number of batches."""
        return len(self.query_loader)


class MetaTrafficIterator:
    """
    Meta traffic data iterator.
    The output format is fully compatible with MultiMetaFingerNet.forward().
    """
    
    def __init__(self, dataloader: MetaTrafficDataLoader):
        self.dataloader = dataloader
        self.query_iter = iter(dataloader.query_loader)
        self.support_data, self.support_masks = dataloader.get_support_data()
        
    def __iter__(self):
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Return the next batch in a format compatible with MultiMetaFingerNet.
        
        Returns:
            query_data: (batch_size, target_length) Query set data.
            support_data: (num_classes, shots_per_class, target_length) Support set data.
            support_masks: (num_classes, shots_per_class, target_length) Support set masks.
            batch_info: Dict containing query labels and metadata.
        """
        try:
            # Get a query batch.
            query_data, query_labels, metadata = next(self.query_iter)
            #self.support_data, self.support_masks = self.dataloader.get_support_data()
            # Organize batch information.
            batch_info = {
                'query_labels': query_labels,  # (batch_size, num_classes)
                'metadata': metadata,
                'class_order': self.dataloader.class_order,
                'num_classes': len(self.dataloader.activated_classes)
            }
            
            return query_data, self.support_data, self.support_masks, batch_info
            
        except StopIteration:
            raise StopIteration


