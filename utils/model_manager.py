"""
Model manager.
Handles model saving, loading, and checkpoint management.
"""

import torch
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import glob
import numpy as np


class ModelManager:
    """Model manager."""
    
    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: Directory for saving checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # File paths.
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        self.final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
        self.latest_model_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        self.metrics_history_path = os.path.join(checkpoint_dir, 'metrics_history.json')
        
        # Metric history.
        self.metrics_history = []
        self.load_metrics_history()
        
        print(f"Model manager initialized: {checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """
        Save checkpoint.
        
        Args:
            model: Model.
            optimizer: Optimizer.
            scheduler: Learning-rate scheduler.
            epoch: Current epoch.
            metrics: Evaluation metrics.
            is_best: Whether this is the best model.
        """
        # Prepare state to save.
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save latest model.
        #torch.save(checkpoint, self.latest_model_path)
        
        # Save to best_model if this is the best model.
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"Saved best model: epoch {epoch+1}, mAP={metrics.get('sig_mAP', 0):.4f}")
            # Update metric history.
            self.metrics_history.append({
                'epoch': epoch,
                'metrics': metrics
            })
            self.save_metrics_history()
        # Save periodic checkpoint.
        #if (epoch + 1) % 50 == 0:
         #   epoch_checkpoint_path = os.path.join(
          #      self.checkpoint_dir, 
           #     f'checkpoint_epoch_{epoch+1}.pth'
            #)
            #torch.save(checkpoint, epoch_checkpoint_path)
        
    def save_final_checkpoint(self, model, optimizer, scheduler, epoch, metrics):
        """Save the model from the final training epoch."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.final_model_path)
        print(f"Saved final model: epoch {epoch+1}, mAP={metrics.get('sig_mAP', 0):.4f}")


    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, 
                       checkpoint_path=None, load_best=True):
        """
        Load checkpoint.
        
        Args:
            model: Model.
            optimizer: Optional optimizer.
            scheduler: Optional learning-rate scheduler.
            checkpoint_path: Specific checkpoint path.
            load_best: Whether to load the best model.
            
        Returns:
            loaded_info: Dictionary with load information.
        """
        # Determine which checkpoint path to load.
        if checkpoint_path is None:
            if load_best and os.path.exists(self.best_model_path):
                checkpoint_path = self.best_model_path
            elif os.path.exists(self.latest_model_path):
                checkpoint_path = self.latest_model_path
            else:
                print("No available checkpoint found")
                return None
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file does not exist: {checkpoint_path}")
            return None
        
        # Load checkpoint.
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state.
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state.
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state.
            if (scheduler is not None and 
                'scheduler_state_dict' in checkpoint and 
                checkpoint['scheduler_state_dict'] is not None):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            loaded_info = {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'timestamp': checkpoint.get('timestamp', ''),
                'checkpoint_path': checkpoint_path
            }
            
            print(f"Checkpoint loaded successfully: {checkpoint_path}")
            print(f"   - Epoch: {loaded_info['epoch']}")
            print(f"   - mAP: {loaded_info['metrics'].get('mAP', 0):.4f}")
            
            return loaded_info
            
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None
    
    def load_model_only(self, model, checkpoint_path=None, load_best=True):
        """
        Load only model weights, without optimizer or other states.
        
        Args:
            model: Model.
            checkpoint_path: Specific checkpoint path.
            load_best: Whether to load the best model.
        """
        # Determine which checkpoint path to load.
        if checkpoint_path is None:
            if load_best and os.path.exists(self.best_model_path):
                checkpoint_path = self.best_model_path
            elif os.path.exists(self.latest_model_path):
                checkpoint_path = self.latest_model_path
            else:
                print("No available checkpoint found")
                return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"Model weights loaded successfully: {checkpoint_path}")
            return {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {})
            }
            
        except Exception as e:
            print(f"Failed to load model weights: {e}")
            return None
    
    def save_metrics_history(self):
        """Save metric history."""
        def _json_safe(obj):
            if isinstance(obj, dict):
                return {str(k): _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_json_safe(v) for v in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return obj

        try:
            with open(self.metrics_history_path, 'w') as f:
                json.dump(_json_safe(self.metrics_history), f, indent=2)
        except Exception as e:
            print(f"Warning: failed to save metric history: {e}")
    
    def load_metrics_history(self):
        """Load metric history."""
        if os.path.exists(self.metrics_history_path):
            try:
                with open(self.metrics_history_path, 'r') as f:
                    self.metrics_history = json.load(f)
                print(f"Loaded metric history: {len(self.metrics_history)} records")
            except Exception as e:
                print(f"Warning: failed to load metric history: {e}")
                self.metrics_history = []
    
    def get_best_metrics(self):
        """Get the best metrics."""
        if not self.metrics_history:
            return None
        
        best_entry = max(self.metrics_history, 
                        key=lambda x: x['metrics'].get('mAP', 0))
        return best_entry
    
    def get_training_summary(self):
        """Get the training summary."""
        if not self.metrics_history:
            return {}
        
        # Extract all mAP values.
        map_values = [entry['metrics'].get('mAP', 0) for entry in self.metrics_history]
        
        summary = {
            'total_epochs': len(self.metrics_history),
            'best_mAP': max(map_values) if map_values else 0,
            'final_mAP': map_values[-1] if map_values else 0,
            'mAP_improvement': map_values[-1] - map_values[0] if len(map_values) > 1 else 0,
            'best_epoch': max(self.metrics_history, 
                            key=lambda x: x['metrics'].get('mAP', 0))['epoch'] if self.metrics_history else 0
        }
        
        return summary
    
    def clean_old_checkpoints(self, keep_latest=5):
        """Clean old epoch checkpoints and keep the latest few."""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= keep_latest:
            return
        
        # Sort by modification time.
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete old files.
        for old_file in checkpoint_files[keep_latest:]:
            try:
                os.remove(old_file)
                print(f"Deleted old checkpoint: {os.path.basename(old_file)}")
            except Exception as e:
                print(f"Warning: failed to delete {old_file}: {e}")
    
    def export_model(self, model, export_path=None, include_config=True):
        """
        Export model for deployment.
        
        Args:
            model: Model.
            export_path: Export path.
            include_config: Whether to include configuration information.
        """
        if export_path is None:
            export_path = os.path.join(self.checkpoint_dir, 'exported_model.pth')
        
        # Ensure the model is in eval mode.
        model.eval()
        
        # Prepare export data.
        export_data = {
            'model_state_dict': model.state_dict(),
            'export_timestamp': datetime.now().isoformat(),
        }
        
        # Add best-model metrics if available.
        best_metrics = self.get_best_metrics()
        if best_metrics:
            export_data['best_metrics'] = best_metrics['metrics']
            export_data['best_epoch'] = best_metrics['epoch']
        
        # Save.
        torch.save(export_data, export_path)
        print(f"Model exported: {export_path}")
        
        return export_path
    
    def print_training_summary(self):
        """Print the training summary."""
        summary = self.get_training_summary()
        
        if not summary:
            print("No training records yet")
            return
        
        print("Training summary:")
        print(f"  - Training epochs: {summary['total_epochs']}")
        print(f"  - Best mAP: {summary['best_mAP']:.4f} (Epoch {summary['best_epoch']+1})")
        print(f"  - Final mAP: {summary['final_mAP']:.4f}")
        print(f"  - mAP improvement: {summary['mAP_improvement']:+.4f}")
        
        # Show available checkpoints.
        available_checkpoints = []
        if os.path.exists(self.best_model_path):
            available_checkpoints.append("best_model.pth")
        if os.path.exists(self.latest_model_path):
            available_checkpoints.append("latest_model.pth")
        
        epoch_checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        )
        available_checkpoints.extend([os.path.basename(f) for f in epoch_checkpoints])
        
        print(f"  - Available checkpoints: {len(available_checkpoints)}")
        for checkpoint in available_checkpoints[:5]:  # Show the first five.
            print(f"    - {checkpoint}")
        if len(available_checkpoints) > 5:
            print(f"    - ... and {len(available_checkpoints)-5} more")


def test_model_manager():
    """Test the model manager."""
    print("Testing model manager...")
    
    # Create test directory.
    test_dir = "./test_checkpoints"
    manager = ModelManager(test_dir)
    
    # Simulate model and optimizer.
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    # Simulate several training epochs.
    for epoch in range(5):
        # Simulate metrics.
        metrics = {
            'mAP': 0.5 + epoch * 0.1,
            'precision_macro': 0.4 + epoch * 0.1,
            'recall_macro': 0.3 + epoch * 0.1
        }
        
        is_best = epoch == 3  # Assume the fourth epoch is best.
        
        # Save checkpoint.
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch, metrics, is_best
        )
    
    # Print training summary.
    manager.print_training_summary()
    
    # Test loading.
    loaded_info = manager.load_checkpoint(model, optimizer, scheduler)
    print(f"\nLoad test: {loaded_info}")
    
    # Clean up test files.
    shutil.rmtree(test_dir)
    print("\nTest complete, cleaned up test files")


if __name__ == '__main__':
    test_model_manager() 
