"""
模型管理器
负责模型保存、加载和checkpoint管理
"""

import torch
import os
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import glob


class ModelManager:
    """模型管理器"""
    
    def __init__(self, checkpoint_dir: str):
        """
        Args:
            checkpoint_dir: checkpoint保存目录
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 文件路径
        self.best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        self.latest_model_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        self.metrics_history_path = os.path.join(checkpoint_dir, 'metrics_history.json')
        
        # 指标历史
        self.metrics_history = []
        self.load_metrics_history()
        
        print(f"📁 模型管理器初始化: {checkpoint_dir}")
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """
        保存checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            epoch: 当前epoch
            metrics: 评估指标
            is_best: 是否是最佳模型
        """
        # 准备保存的状态
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存最新模型
        #torch.save(checkpoint, self.latest_model_path)
        
        # 如果是最佳模型，保存到best_model
        if is_best:
            torch.save(checkpoint, self.best_model_path)
            print(f"💾 保存最佳模型: epoch {epoch+1}, mAP={metrics.get('sig_mAP', 0):.4f}")
            # 更新指标历史
            self.metrics_history.append({
                'epoch': epoch,
                'metrics': metrics
            })
            self.save_metrics_history()
        # # 保存定期checkpoint
        # if (epoch + 1) % 50 == 0:
        #     epoch_checkpoint_path = os.path.join(
        #         self.checkpoint_dir, 
        #         f'checkpoint_epoch_{epoch+1}.pth'
        #     )
        #     torch.save(checkpoint, epoch_checkpoint_path)
        

    
    def load_checkpoint(self, model, optimizer=None, scheduler=None, 
                       checkpoint_path=None, load_best=True):
        """
        加载checkpoint
        
        Args:
            model: 模型
            optimizer: 优化器（可选）
            scheduler: 学习率调度器（可选）
            checkpoint_path: 指定的checkpoint路径
            load_best: 是否加载最佳模型
            
        Returns:
            loaded_info: 加载信息字典
        """
        # 确定要加载的checkpoint路径
        if checkpoint_path is None:
            if load_best and os.path.exists(self.best_model_path):
                checkpoint_path = self.best_model_path
            elif os.path.exists(self.latest_model_path):
                checkpoint_path = self.latest_model_path
            else:
                print("❌ 没有找到可用的checkpoint")
                return None
        
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
            return None
        
        # 加载checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # 加载模型状态
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加载优化器状态
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # 加载调度器状态
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
            
            print(f"✅ 成功加载checkpoint: {checkpoint_path}")
            print(f"   - Epoch: {loaded_info['epoch']}")
            print(f"   - mAP: {loaded_info['metrics'].get('mAP', 0):.4f}")
            
            return loaded_info
            
        except Exception as e:
            print(f"❌ 加载checkpoint失败: {e}")
            return None
    
    def load_model_only(self, model, checkpoint_path=None, load_best=True):
        """
        只加载模型权重，不加载优化器等
        
        Args:
            model: 模型
            checkpoint_path: 指定的checkpoint路径
            load_best: 是否加载最佳模型
        """
        # 确定要加载的checkpoint路径
        if checkpoint_path is None:
            if load_best and os.path.exists(self.best_model_path):
                checkpoint_path = self.best_model_path
            elif os.path.exists(self.latest_model_path):
                checkpoint_path = self.latest_model_path
            else:
                print("❌ 没有找到可用的checkpoint")
                return None
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print(f"✅ 成功加载模型权重: {checkpoint_path}")
            return {
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {})
            }
            
        except Exception as e:
            print(f"❌ 加载模型权重失败: {e}")
            return None
    
    def save_metrics_history(self):
        """保存指标历史"""
        try:
            with open(self.metrics_history_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ 保存指标历史失败: {e}")
    
    def load_metrics_history(self):
        """加载指标历史"""
        if os.path.exists(self.metrics_history_path):
            try:
                with open(self.metrics_history_path, 'r') as f:
                    self.metrics_history = json.load(f)
                print(f"📊 加载指标历史: {len(self.metrics_history)}条记录")
            except Exception as e:
                print(f"⚠️ 加载指标历史失败: {e}")
                self.metrics_history = []
    
    def get_best_metrics(self):
        """获取最佳指标"""
        if not self.metrics_history:
            return None
        
        best_entry = max(self.metrics_history, 
                        key=lambda x: x['metrics'].get('mAP', 0))
        return best_entry
    
    def get_training_summary(self):
        """获取训练摘要"""
        if not self.metrics_history:
            return {}
        
        # 提取所有mAP值
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
        """清理旧的epoch checkpoints，保留最新的几个"""
        pattern = os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        checkpoint_files = glob.glob(pattern)
        
        if len(checkpoint_files) <= keep_latest:
            return
        
        # 按修改时间排序
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)
        
        # 删除旧文件
        for old_file in checkpoint_files[keep_latest:]:
            try:
                os.remove(old_file)
                print(f"🗑️ 删除旧checkpoint: {os.path.basename(old_file)}")
            except Exception as e:
                print(f"⚠️ 删除失败 {old_file}: {e}")
    
    def export_model(self, model, export_path=None, include_config=True):
        """
        导出模型用于部署
        
        Args:
            model: 模型
            export_path: 导出路径
            include_config: 是否包含配置信息
        """
        if export_path is None:
            export_path = os.path.join(self.checkpoint_dir, 'exported_model.pth')
        
        # 确保模型在eval模式
        model.eval()
        
        # 准备导出数据
        export_data = {
            'model_state_dict': model.state_dict(),
            'export_timestamp': datetime.now().isoformat(),
        }
        
        # 如果有最佳模型的指标，添加进去
        best_metrics = self.get_best_metrics()
        if best_metrics:
            export_data['best_metrics'] = best_metrics['metrics']
            export_data['best_epoch'] = best_metrics['epoch']
        
        # 保存
        torch.save(export_data, export_path)
        print(f"📦 模型已导出: {export_path}")
        
        return export_path
    
    def print_training_summary(self):
        """打印训练摘要"""
        summary = self.get_training_summary()
        
        if not summary:
            print("📊 暂无训练记录")
            return
        
        print("📊 训练摘要:")
        print(f"  - 训练轮数: {summary['total_epochs']}")
        print(f"  - 最佳mAP: {summary['best_mAP']:.4f} (Epoch {summary['best_epoch']+1})")
        print(f"  - 最终mAP: {summary['final_mAP']:.4f}")
        print(f"  - mAP提升: {summary['mAP_improvement']:+.4f}")
        
        # 显示可用的checkpoint
        available_checkpoints = []
        if os.path.exists(self.best_model_path):
            available_checkpoints.append("best_model.pth")
        if os.path.exists(self.latest_model_path):
            available_checkpoints.append("latest_model.pth")
        
        epoch_checkpoints = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_epoch_*.pth')
        )
        available_checkpoints.extend([os.path.basename(f) for f in epoch_checkpoints])
        
        print(f"  - 可用checkpoint: {len(available_checkpoints)}个")
        for checkpoint in available_checkpoints[:5]:  # 显示前5个
            print(f"    • {checkpoint}")
        if len(available_checkpoints) > 5:
            print(f"    • ... 和其他{len(available_checkpoints)-5}个")


def test_model_manager():
    """测试模型管理器"""
    print("测试模型管理器...")
    
    # 创建测试目录
    test_dir = "./test_checkpoints"
    manager = ModelManager(test_dir)
    
    # 模拟模型和优化器
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    
    # 模拟训练几个epoch
    for epoch in range(5):
        # 模拟指标
        metrics = {
            'mAP': 0.5 + epoch * 0.1,
            'precision_macro': 0.4 + epoch * 0.1,
            'recall_macro': 0.3 + epoch * 0.1
        }
        
        is_best = epoch == 3  # 假设第4个epoch是最佳
        
        # 保存checkpoint
        manager.save_checkpoint(
            model, optimizer, scheduler, epoch, metrics, is_best
        )
    
    # 打印训练摘要
    manager.print_training_summary()
    
    # 测试加载
    loaded_info = manager.load_checkpoint(model, optimizer, scheduler)
    print(f"\n加载测试: {loaded_info}")
    
    # 清理测试文件
    shutil.rmtree(test_dir)
    print("\n✅ 测试完成，清理测试文件")


if __name__ == '__main__':
    test_model_manager() 