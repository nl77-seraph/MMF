import subprocess
import sys
from datetime import datetime

# Specify the output file path.
output_file = "fewshot_dataset_generator_log_front.txt"  # Log filename; adjust as needed.

def run_and_log(cmd, description, log_file):
    """Run a command and write output to both the console and a file."""
    print(f"Running {description}...")
    with open(log_file, 'a', encoding='utf-8') as f:
        # Write the separator and timestamp.
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = f"\n{'='*60}\n"
        header = f"{separator}Running {description}\nTime: {timestamp}{separator}\n"
        
        print(header)
        f.write(header)
        
        # Run the command.
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Output to the console.
        print(res.stdout, end="")
        
        # Output to the log file.
        f.write(res.stdout)
        f.write("\n")
    
    return res

# All k-shot settings to run.
k_shots = [20]

for k in k_shots:
    for num_base in range(2, 5):  # 1, 2, 3
        cmd = [
            sys.executable, "fewshot_dataset_generator.py",
            "--base-training-dir", "/data/datasets/benchdata/MMF_datasets/datasets/front/multi/3tab/",
            "--novel-source-dir", "/data/datasets/benchdata/MMF_datasets/datasets/front/single/",
            "--output-dir", "/data/datasets/benchdata/MMF_datasets/datasets/front/fewshot/",
            "--novel-classes", "60-89",
            "--k-shot", str(k),
            "--num-base-per-query", str(num_base),
        ]

        desc = f"front_fewshot_dataset_generator_k{k}_base{num_base}"
        run_and_log(cmd, desc, output_file)
    # cmd = [
    #     sys.executable, "fewshot_dataset_generator.py",
    #     "--base-training-dir", "/store1/chenyi/WF/datasets/6000ada/CW_base_training/mixed_tab/",
    #     "--novel-source-dir", "/store1/chenyi/WF/datasets/6000ada/CW_split_folder/",
    #     "--output-dir", "/store1/chenyi/WF/datasets/6000ada/CW_fewshot/",
    #     "--novel-classes", "60-89",
    #     "--k-shot", str(k),
    #     "--mixed"
    # ]

    desc = f"front_fewshot_dataset_generator_k{k}_base{num_base}"
    run_and_log(cmd, desc, output_file)
print(f"\nAll logs have been saved to: {output_file}")
