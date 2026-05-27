
import os
import random
import torch
import numpy as np
import pickle
import json
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
import torch.distributed as dist
def is_main_process():
    """Check whether this is the main process."""
    return not dist.is_initialized() or dist.get_rank() == 0

class QueryTrafficDataset(Dataset):
    """
    Query set data loader.
    Follows the listDataset design from Few-shot Detection.
    """
    
    def __init__(self, 
                 json_index_path: str,
                 query_files_dir: str,
                 target_length: int = 30000,
                 activated_classes: List[int] = None):
        """
        Args:
            json_index_path: Path to the query set index JSON file.
            query_files_dir: Directory containing query data files.
            target_length: Target sequence length.
            activated_classes: List of active classes; defaults to 0-59.
        """
        self.json_index_path = json_index_path
        self.query_files_dir = query_files_dir
        self.target_length = target_length
        self.activated_classes = activated_classes if activated_classes else list(range(60))  # 0-59
        
        # Load the query set index.
        self._load_query_index()
        if is_main_process():
            print(f"QueryTrafficDataset initialization complete:")
            print(f"  - Query samples: {len(self.query_index)}")
            print(f"  - Active classes: {len(self.activated_classes)}")
            print(f"  - Target sequence length: {self.target_length}")
    
    def _load_query_index(self):
        """Load the query set index."""
        if os.path.exists(self.json_index_path):
            with open(self.json_index_path, 'r') as f:
                query_file_names = json.load(f)
        else:
            # Get all filenames ending with .pkl in the directory.
            query_file_names = [f for f in os.listdir(self.query_files_dir) if f.endswith('.pkl')]
            # Sort to keep the loading order consistent across machines and runs.
            query_file_names.sort()
        self.query_index = []
        for filename in query_file_names:
            # Parse labels from the filename.
            labels = self._parse_labels_from_filename(filename)
            
            if labels:  # Keep only files with valid labels.
                file_path = os.path.join(self.query_files_dir, filename)
                self.query_index.append({
                    'filename': filename,
                    'labels': labels,
                    'file_path': file_path
                })
                
        if is_main_process():
            print(f"Valid query samples: {len(self.query_index)}")
    
    def _parse_labels_from_filename(self, filename: str) -> List[int]:
        """
        Parse labels from the filename.
        Filename format: "class1_class2_class3_random_filename.pkl".
        """
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        
        labels = []
        for part in parts:
            if 'novel' in part:
                continue
            else:
                try:
                    label = int(part)
                    if label in self.activated_classes:
                        labels.append(label)
                except ValueError:
                    # Treat the first non-numeric part as the random filename and stop parsing.
                    break
        
        return labels
    
    def _process_sequence(self, raw_data: List) -> torch.Tensor:
        """
        Process sequence data by truncating or padding to the target length.
        """
        if len(raw_data) >= self.target_length:
            # Truncate.
            processed = raw_data[:self.target_length]
        else:
            # Pad with zeros.
            processed = raw_data + [0] * (self.target_length - len(raw_data))
        
        return torch.tensor(processed, dtype=torch.float32)
    
    def _labels_to_multihot(self, labels: List[int]) -> torch.Tensor:
        """
        Convert a list of labels to multi-hot encoding.
        """
        num_classes = len(self.activated_classes)
        multihot = torch.zeros(num_classes, dtype=torch.float32)
        
        for label in labels:
            if label in self.activated_classes:
                idx = self.activated_classes.index(label)
                multihot[idx] = 1.0
                
        return multihot
    
    def __len__(self) -> int:
        return len(self.query_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Get a query sample.
        
        Returns:
            query_data: (target_length,) Query sequence.
            query_labels: (num_classes,) Multi-hot encoded labels.
            metadata: Metadata dictionary.
        """
        sample_info = self.query_index[idx]
        
        # Load data.
        with open(sample_info['file_path'], 'rb') as f:
            sample_data = pickle.load(f)
        
        # Handle different data formats.
        if isinstance(sample_data, dict) :
            if 'data' in sample_data:
                raw_data = sample_data['data']
            else: 
                raw_data = sample_data['direction']
        elif isinstance(sample_data, (list, np.ndarray)):
            raw_data = sample_data
        else:
            # Use other formats directly.
            raw_data = sample_data
        
        # Ensure raw_data is a list.
        if isinstance(raw_data, np.ndarray):
            raw_data = raw_data.tolist()
        elif not isinstance(raw_data, list):
            raw_data = [raw_data]
        
        # Process the sequence.
        query_data = self._process_sequence(raw_data)
        
        # Process labels.
        query_labels = self._labels_to_multihot(sample_info['labels']) ##Note Do not use Multi-hot if order information is needed.
        
        # Metadata.
        metadata = {
            'filename': sample_info['filename'],
            'original_labels': sample_info['labels'],
            'file_path': sample_info['file_path']
        }
        
        return query_data, query_labels, metadata


class SupportTrafficDataset(Dataset):
    """
    Support set data loader.
    Follows the MetaDataset design from Few-shot Detection.
    Generates support sets for all classes without episode sampling.
    Supports fixed sampling for few-shot adaptation and random sampling for training.
    """
    
    def __init__(self,
                 support_root_dir: str,
                 activated_classes: List[int] = None,
                 target_length: int = 30000,
                 shots_per_class: int = 1,
                 random_sampling: bool = False):
        """
        Args:
            support_root_dir: Root directory of the support set.
            activated_classes: List of active classes; defaults to 0-59.
            target_length: Target sequence length, corrected to 30000.
            shots_per_class: Number of samples per class.
            random_sampling: Whether to use random sampling mode. True selects randomly each time; False uses fixed selection.
        """
        self.support_root_dir = support_root_dir
        self.activated_classes = activated_classes if activated_classes else list(range(60))  # 0-59
        self.target_length = target_length
        self.shots_per_class = shots_per_class
        self.random_sampling = random_sampling
        
        # Build the support set index.
        self._build_support_index()
        
        if not self.random_sampling:
            # Fixed sampling mode: pre-generate support sets for all classes.
            self._prepare_all_support_data()
        else:
            if is_main_process():
                # Random sampling mode: only record file indices and load dynamically each time.
                print(f"SupportTrafficDataset initialization complete (random sampling mode):")
                print(f"  - Active classes: {len(self.activated_classes)}")
                print(f"  - Samples per class: {self.shots_per_class}")
                print(f"  - Target sequence length: {self.target_length}")
                print(f"  - Random sampling: {self.random_sampling}")
    
    def _build_support_index(self):
        """Build the support set index."""
        self.support_files_by_class = {}
        
        for class_id in self.activated_classes:
            class_dir = os.path.join(self.support_root_dir, str(class_id))
            if not os.path.exists(class_dir):
                print(f"Warning: Directory for class {class_id} does not exist: {class_dir}")
                continue
            
            # Collect all pkl files for this class.
            class_files = [
                os.path.join(class_dir, f) 
                for f in os.listdir(class_dir) 
                if f.endswith('.pkl')
            ]
            
            if len(class_files) < self.shots_per_class:
                print(f"Warning: Class {class_id} has insufficient samples; need {self.shots_per_class}, found {len(class_files)}")
            
            self.support_files_by_class[class_id] = class_files
            if is_main_process():
                print(f"Class {class_id}: found {len(class_files)} support samples")
    
    def _process_support_sequence(self, raw_data: List) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process support set sequences by padding to the target length and generating masks.
        
        Returns:
            data: (target_length,) Padded sequence.
            mask: (target_length,) Valid data mask; 1 means valid and 0 means padded.
        """
        original_length = len(raw_data)
        
        if original_length >= self.target_length:
            # Truncate.
            data = raw_data[:self.target_length]
            mask = torch.ones(self.target_length, dtype=torch.bool)
        else:
            # Pad with zeros.
            data = raw_data + [0] * (self.target_length - original_length)
            mask = torch.zeros(self.target_length, dtype=torch.bool)
            mask[:original_length] = True
        
        return torch.tensor(data, dtype=torch.float32), mask
    
    def _prepare_all_support_data(self):
        """Pre-generate support set data for all classes in fixed sampling mode."""
        self.all_support_data = []
        self.all_support_masks = []
        self.class_order = []  # Record class order to keep indices aligned.
        
        for class_id in sorted(self.activated_classes):  # Sort for consistency.
            if class_id not in self.support_files_by_class:
                # Create a zero vector if a class has no data.
                print(f"Warning: Class {class_id} has no support samples")
                exit(0)
            
            class_files = self.support_files_by_class[class_id]
            
            for shot_idx in range(self.shots_per_class):
                # Select files deterministically by cycling through them.
                if len(class_files) > 0:
                    file_idx = shot_idx % len(class_files)
                    file_path = class_files[file_idx]
                    
                    # Load data.
                    data, mask = self._load_and_process_sample(file_path)
                else:
                    data = torch.zeros(self.target_length, dtype=torch.float32)
                    mask = torch.zeros(self.target_length, dtype=torch.bool)
                
                self.all_support_data.append(data)
                self.all_support_masks.append(mask)
                self.class_order.append(class_id)
        
        # Convert to tensors.
        # shape: (num_classes * shots_per_class, target_length)
        self.support_data_tensor = torch.stack(self.all_support_data)
        self.support_masks_tensor = torch.stack(self.all_support_masks)
        if is_main_process():
            print(f"Support set data preparation complete (fixed sampling mode):")
            print(f"  - Support set shape: {self.support_data_tensor.shape}")
            print(f"  - Mask shape: {self.support_masks_tensor.shape}")
            print(f"  - Class order: {self.class_order}")
    
    def _load_and_process_sample(self, file_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and process a single sample file.
        
        Args:
            file_path: Sample file path.
            
        Returns:
            data: (target_length,) Processed sequence.
            mask: (target_length,) Valid data mask.
        """
        try:
            with open(file_path, 'rb') as f:
                sample_data = pickle.load(f)
            
            # Handle different data formats.
            if isinstance(sample_data, dict) :
                if 'data' in sample_data:
                    raw_data = sample_data['data']
                else: 
                    raw_data = sample_data['direction']
            elif isinstance(sample_data, (list, np.ndarray)):
                raw_data = sample_data
            else:
                # Use other formats directly.
                raw_data = sample_data
            
            # Ensure raw_data is a list.
            if isinstance(raw_data, np.ndarray):
                raw_data = raw_data.tolist()
            elif not isinstance(raw_data, list):
                raw_data = [raw_data]
            
            # Process the sequence and mask.
            data, mask = self._process_support_sequence(raw_data)
            return data, mask
            
        except Exception as e:
            print(f"Warning: Failed to load file {file_path}: {e}; using a zero vector")
            data = torch.zeros(self.target_length, dtype=torch.float32)
            mask = torch.zeros(self.target_length, dtype=torch.bool)
            return data, mask
    
    def _generate_random_support_batch(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Randomly generate one support set batch in random sampling mode.
        
        Returns:
            support_data: (num_classes, shots_per_class, target_length)
            support_masks: (num_classes, shots_per_class, target_length)
            class_order: List[int] Class order, preserving the 0-59 order.
        """
        import random
        
        batch_support_data = []
        batch_support_masks = []
        class_order = sorted(self.activated_classes)  # Keep ordering consistent.
        
        for class_id in class_order:
            if class_id not in self.support_files_by_class:
                print(f"Warning: Class {class_id} has no support samples")
                exit(0)
            
            class_files = self.support_files_by_class[class_id]
            
            for shot_idx in range(self.shots_per_class):
                if len(class_files) > 0:
                    # Randomly select a file.
                    file_path = random.choice(class_files)
                    data, mask = self._load_and_process_sample(file_path)
                
                batch_support_data.append(data)
                batch_support_masks.append(mask)
        
        # Convert to tensors and reshape.
        num_classes = len(class_order)
        support_data_tensor = torch.stack(batch_support_data)
        support_masks_tensor = torch.stack(batch_support_masks)
        
        # Reshape to (num_classes, shots_per_class, target_length).
        support_data = support_data_tensor.view(
            num_classes, self.shots_per_class, self.target_length
        )
        support_masks = support_masks_tensor.view(
            num_classes, self.shots_per_class, self.target_length
        )
        
        return support_data, support_masks, class_order

    def get_all_support_data(self) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """
        Get support set data for all classes.
        
        Returns:
            support_data: (num_classes, shots_per_class, target_length)
            support_masks: (num_classes, shots_per_class, target_length)  
            class_order: List[int] Class order.
        """
        if self.random_sampling:
            # Random sampling mode: generate new random samples on each call.
            return self._generate_random_support_batch()
        else:
            # Fixed sampling mode: return pre-generated data.
            num_classes = len(self.activated_classes)
            
            # Reshape to (num_classes, shots_per_class, target_length).
            support_data = self.support_data_tensor.view(
                num_classes, self.shots_per_class, self.target_length
            )
            support_masks = self.support_masks_tensor.view(
                num_classes, self.shots_per_class, self.target_length
            )
            
            return support_data, support_masks, self.activated_classes
    
    def __len__(self) -> int:
        return len(self.all_support_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a single support sample. Usually not used directly; use get_all_support_data instead.
        """
        return (
            self.all_support_data[idx],
            self.all_support_masks[idx], 
            self.class_order[idx]
        )

