import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, default='trashnet\\data\\dataset-resized')
parser.add_argument('--val_split', type=float, default=0.2)
args = parser.parse_args()

train_path = os.path.abspath(args.train_path)
val_split = args.val_split

try:
    import pandas as pd
    from sklearn.model_selection import train_test_split
except Exception as e:
    print('pandas or sklearn not available:', e)
    print('Install them or run the script in an environment with pandas and scikit-learn to compute exact hold-out percentages.')
    raise SystemExit(1)

rows = []
for cls in os.listdir(train_path):
    cls_dir = os.path.join(train_path, cls)
    if not os.path.isdir(cls_dir):
        continue
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG'):
        for p in glob.glob(os.path.join(cls_dir, ext)):
            rows.append({'filename': p, 'class': cls})

if not rows:
    print('No images found in', train_path)
    raise SystemExit(1)

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.DataFrame(rows)
train_df, val_df = train_test_split(df, test_size=val_split, stratify=df['class'], random_state=42)

total_val = len(val_df)
print(f'Validation set: {total_val} images ({val_split*100:.1f}% hold-out)')
val_counts = val_df['class'].value_counts()
for cls, cnt in val_counts.items():
    pct = cnt / total_val * 100
    print(f'{cls}: {cnt} images ({pct:.2f}%)')
