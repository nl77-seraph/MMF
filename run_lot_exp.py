import subprocess
import sys
from datetime import datetime

# 指定输出文件路径
output_file = "wtf_pad.txt"  # 可以修改为你想要的文件名

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

# # 运行第一个配置
# cmd2 = [sys.executable, "train_enhanced.py",
#         "--config", "configs/Front_weight_bce_short_support_mixed.json"]
# res2 = run_and_log(cmd2, "front", output_file)

# # 运行第二个配置
# cmd3 = [sys.executable, "train_enhanced.py",
#         "--config", "configs/multi_walkie_weight_bce_short_support_mixed.json"]
# res3 = run_and_log(cmd3, "walkie", output_file)
# # 运行第二个配置
# cmd4 = [sys.executable, "train_enhanced.py",
#         "--config", "configs/silver_weight_bce_short_support_mixed.json"]
# res4 = run_and_log(cmd4, "silver", output_file)
# 运行第二个配置
cmd5 = [sys.executable, "train_enhanced.py",
        "--config", "configs/wtf_pad_weight_bce_short_support_mixed.json"]
res5 = run_and_log(cmd5, "wtf_pad", output_file)

print(f"\n所有日志已保存到: {output_file}")