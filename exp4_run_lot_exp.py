import subprocess
import sys
from datetime import datetime

# 指定输出文件路径
output_file = "02_4tab+04_5tab_disjoint.txt"  # 可以修改为你想要的文件名

def run_and_log(cmd, description, log_file):
    """运行命令并同时输出到控制台和文件"""
    print(f"Running {description}...")
    with open(log_file, 'a', encoding='utf-8') as f:
        # 写入分隔符和时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = f"\n{'='*60}\n"
        header = f"{separator}Running {description}\nTime: {timestamp}{separator}\n"
        
        print(header)
        f.write(header)
        
        # 运行命令
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # 输出到控制台
        print(res.stdout, end="")
        
        # 输出到文件
        f.write(res.stdout)
        f.write("\n")
    
    return res

# 运行第一个配置
cmd2 = [sys.executable, "train_enhanced.py",
        "--config", "configs/unseen_combinations/02over_4tab.json"]
res2 = run_and_log(cmd2, "4tab", output_file)

cmd4 = [sys.executable, "train_enhanced.py",
        "--config", "configs/unseen_combinations/04over_5tab.json"]
res4 = run_and_log(cmd4, "5tab", output_file)

print(f"\n所有日志已保存到: {output_file}")