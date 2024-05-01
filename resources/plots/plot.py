import matplotlib.pyplot as plt
import seaborn as sns
import json
import glob
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot PSNR vs Steps for a given scene")
    parser.add_argument("--scene", type=str, help="Name of the scene")
    parser.add_argument("--directory", type=str, help="Directory path containing the PSNR JSON files")
    return parser.parse_args()

# parse commandline arguments
args = parse_arguments()

scene = args.scene
directory = args.directory#"/home/hice1/apeng39/scratch/SENeLF/MobileR2L/logs/Experiments/pretrained_mobiler2l"

lines = []

test_psnrs = []
train_psnrs = []
steps = []

for psnr_file in glob.iglob(os.path.join(directory, f'{scene}*', '**', 'psnr.json'), recursive=True):
    # print(psnr_file)
    with open(psnr_file, 'r') as f:
        lines.extend(f.readlines())

for line in lines:
    line = line.strip()

    if "{" in line or "}" in line:
        continue
    
    value = float(line.split(":")[1][1:-1])
    if "best_psnr" in line and "step" not in line:
        test_psnrs.append(value)
    elif "best_psnr_step" in line:
        steps.append(value)
    elif "train_psnr" in line:
        train_psnrs.append(value)

# print(train_psnrs)
# print(test_psnrs)
# print(steps)

# sort data by steps
steps = list(map(int, steps))
sorted_lists = sorted(zip(steps, train_psnrs, test_psnrs))
steps, train_psnrs, test_psnrs = zip(*sorted_lists)

# print("\n")
# print(train_psnrs)
# print(test_psnrs)
# print(steps)

# plot
sns.set_style("whitegrid")
plt.plot(steps, train_psnrs, color='blue', label='train PSNR')
plt.plot(steps, test_psnrs, color='orange', label='test PSNR')

plt.xticks(range(int(min(steps)), int(max(steps)) + 1, 10000)[::3])
plt.yticks(range(int(min(train_psnrs + test_psnrs)), int(max(train_psnrs + test_psnrs)) + 1))

plt.xlabel('Steps')
plt.ylabel('PSNR')
plt.title('Train and Test PSNR vs Steps')

plt.legend()
plt.grid(True)

plt.savefig(f'{scene}_psnr_plot.png')
