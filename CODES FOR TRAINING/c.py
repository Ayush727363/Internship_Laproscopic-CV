import json
from pathlib import Path

p = Path(r"D:\Study\CDC Project 1\Project\Merged_Dataset\annotations\instances_train.json")
coco = json.loads(p.read_text())
imgs = coco["images"]

print(f"total images: {len(imgs)}")
print("\nfirst entry (ALL keys):")
print(json.dumps(imgs[0], indent=2))

common = {"id", "file_name", "width", "height"}
extra_keys = set()
for im in imgs:
    extra_keys |= (set(im.keys()) - common)
print(f"\nextra keys found: {extra_keys}")

from collections import Counter
for key in extra_keys:
    vals = [str(im.get(key)) for im in imgs]
    c = Counter(vals)
    print(f"\n'{key}': {len(c)} distinct values, top 5: {c.most_common(5)}")