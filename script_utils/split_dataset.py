import json
import os
from collections import defaultdict
import random

# Path to the original file
input_file = 'data/mnist-inrs/mnist_splits_full.json'
output_file = 'data/mnist-inrs/mnist_splits.json'

# Load the original JSON file
with open(input_file, 'r') as f:
    data = json.load(f)


# Structure of the JSON file
# data = {
#     "test": {
#        "path": []
#        "label": []
#     }
#
#     "train": {
#        "path": []
#        "label": []
#     }
#
#     "val": {
#        "path": []
#        "label": []
#     }
# }

samples_per_class = {
    "train": 1,
    "val": 1,
    "test": 1
}
new_config = {}
for split in ["test", "train", "val"]:
    # Filter paths and labels for the current split
    paths = data[split]['path']
    labels = data[split]['label']

    # Create a dictionary to store indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Sample a percentage of the data while maintaining class balance
    sampled_indices = []
    for label, indices in class_indices.items():
        sampled_indices.extend(random.sample(indices, samples_per_class[split]))

    # Filter paths and labels using the sampled indices
    paths = [paths[i] for i in sampled_indices]
    labels = [labels[i] for i in sampled_indices]

    # Create a new dictionary with the filtered data
    new_config[split] = {
        'path': paths,
        'label': labels
    }

# Save the reduced version to a JSON file
with open(output_file, 'w') as f:
    json.dump(new_config, f, indent=4)

print(f"Reduced file saved to {output_file}")
