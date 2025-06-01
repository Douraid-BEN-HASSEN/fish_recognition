import h5py
import json
import numpy as np

DATASET_FILE_PATH = './datasets/sardine_classifier_dataset/profile_sardine_classifier_dataset.h5'

dataset = {
    'A': None,
    'B': None,
    'C': None
}

with open('./datasets/sardine_classifier_dataset/profile_sardine_A.json', 'r') as f:
    dataset['A'] = json.load(f)['vectors']
with open('./datasets/sardine_classifier_dataset/profile_sardine_B.json', 'r') as f:
    dataset['B'] = json.load(f)['vectors']
with open('./datasets/sardine_classifier_dataset/profile_sardine_C.json', 'r') as f:
    dataset['C'] = json.load(f)['vectors']

X = []
y = []
class_mapping = {name: i for i, name in enumerate(dataset.keys())}

for fish_class, vectors in dataset.items():
    X.extend(vectors)
    y.extend([class_mapping[fish_class]] * len(vectors))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Save to HDF5
with h5py.File(DATASET_FILE_PATH, 'w') as f:
    f.create_dataset('X', data=X)  # Features
    f.create_dataset('y', data=y)  # Labels
    
    # Store class mapping as attribute
    f.attrs['class_mapping'] = str(class_mapping)
    f.attrs['description'] = "1 hour, profile normalization"