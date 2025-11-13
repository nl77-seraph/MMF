"""
修复JSON索引格式，从现有字典格式中提取filename，随机打乱后保存为文件名列表格式
"""

import json
import random

def fix_json_indices():
    """从现有字典格式JSON中提取filename并随机打乱"""
    
    print("🔧 修复JSON索引格式...")
    
    # 设置随机种子保证可重现
    random.seed(42)
    
    # 读取现有的字典格式JSON文件
    train_dict_path = 'datasets/3tab_exp/base_train/3tab_train.json'
    val_dict_path = 'datasets/3tab_exp/base_train/3tab_val.json'
    
    # 处理训练集
    print(f"📖 读取训练集JSON: {train_dict_path}")
    with open(train_dict_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 提取filename
    train_filenames = [entry['filename'] for entry in train_data]
    print(f"  - 提取到{len(train_filenames)}个训练集文件名")
    
    # 随机打乱顺序
    random.shuffle(train_filenames)
    print(f"  - 随机打乱完成")
    
    # 处理验证集
    print(f"📖 读取验证集JSON: {val_dict_path}")
    with open(val_dict_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    
    # 提取filename
    val_filenames = [entry['filename'] for entry in val_data]
    print(f"  - 提取到{len(val_filenames)}个验证集文件名")
    
    # 随机打乱顺序
    random.shuffle(val_filenames)
    print(f"  - 随机打乱完成")
    
    # 保存原有文件作为备份
    backup_train_path = 'datasets/3tab_exp/base_train/3tab_train_detailed.json'
    backup_val_path = 'datasets/3tab_exp/base_train/3tab_val_detailed.json'
    
    with open(backup_train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(backup_val_path, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 备份详细信息:")
    print(f"  - 训练集详细信息: {backup_train_path}")
    print(f"  - 验证集详细信息: {backup_val_path}")
    
    # 保存新的文件名列表格式
    with open(train_dict_path, 'w', encoding='utf-8') as f:
        json.dump(train_filenames, f, ensure_ascii=False, indent=2)
    
    with open(val_dict_path, 'w', encoding='utf-8') as f:
        json.dump(val_filenames, f, ensure_ascii=False, indent=2)
    
    print(f'✅ JSON索引修复完成:')
    print(f'  - 训练集: {len(train_filenames)}个文件 (已随机打乱)')
    print(f'  - 验证集: {len(val_filenames)}个文件 (已随机打乱)')
    
    # 验证几个文件名的格式
    print(f'\n📋 训练集示例文件名:')
    for i, f in enumerate(train_filenames[:5]):
        print(f'  {i+1}. {f}')
    
    print(f'\n📋 验证集示例文件名:')
    for i, f in enumerate(val_filenames[:3]):
        print(f'  {i+1}. {f}')

if __name__ == "__main__":
    fix_json_indices() 