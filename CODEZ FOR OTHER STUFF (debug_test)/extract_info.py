import os
import json
import glob
from collections import Counter

def print_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}\n")

# 1. runs/segment/surgical/args.yaml
print_header("runs/segment/surgical/args.yaml")
try:
    with open('runs/segment/surgical/args.yaml', 'r') as f:
        print(f.read().strip())
except Exception as e:
    print(f"Error: {e}")

# 2. Last 60 lines of runs/segment/surgical/results.csv, plus header
print_header("runs/segment/surgical/results.csv (Header + Last 60 lines)")
try:
    with open('runs/segment/surgical/results.csv', 'r') as f:
        lines = f.readlines()
        if lines:
            print(lines[0].strip())
            for line in lines[-60:]:
                print(line.strip())
except Exception as e:
    print(f"Error: {e}")

# 3. data/yolo_final/.build_complete.json
print_header("data/yolo_final/.build_complete.json")
try:
    with open('data/yolo_final/.build_complete.json', 'r') as f:
        print(f.read().strip())
except Exception as e:
    print(f"Error: {e}")

# 4. Directory file counts
print_header("Directory File Counts")
dirs = [
    'data/yolo_final/images/train', 'data/yolo_final/images/val', 'data/yolo_final/images/test',
    'data/yolo_final/labels/train', 'data/yolo_final/labels/val', 'data/yolo_final/labels/test',
    'data/synthetic/images', 'data/synthetic/labels_seg', 'Merged_Dataset/images'
]
for d in dirs:
    try:
        count = len([name for name in os.listdir(d) if os.path.isfile(os.path.join(d, name))])
        print(f"{d}: {count} files")
    except Exception as e:
        print(f"{d}: Error - {e}")

# 5. train_yolo_run.log
print_header("train_yolo_run.log Extractions")
try:
    with open('train_yolo_run.log', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        in_per_class = False
        in_leakage = False
        for line in lines:
            if "PER-CLASS INSTANCES" in line:
                in_per_class = True
                in_leakage = False
            elif "2. LEAKAGE AUDIT" in line:
                in_leakage = True
                in_per_class = False
            elif line.strip() == "" or "---" in line or "===" in line: # simplistic end condition, let's just print
                pass

            # A better way to extract sections:
            if in_per_class:
                # We need a stop condition for PER-CLASS INSTANCES. Usually an empty line or a specific next section.
                pass
except Exception as e:
    print(f"Error: {e}")

