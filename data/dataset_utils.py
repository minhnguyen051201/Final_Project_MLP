import os
from glob import glob
from random import sample


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def get_dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = get_dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


def build_dataset_from_dir(root_dir, domain_name, exts=("*.jpg", "*.jpeg", "*.png")):
    """Build file names and integer labels by scanning a root directory.

    Expected structure: root_dir/domain_name/<class_name>/*.{jpg,png,...}
    Returns names relative to root_dir, i.e., f"{domain_name}/{class_name}/{filename}"
    Labels are assigned by sorted class_name order for determinism.
    """
    domain_root = os.path.join(root_dir, domain_name)
    if not os.path.isdir(domain_root):
        raise FileNotFoundError(f"Domain path not found: {domain_root}")

    classes = [d for d in os.listdir(domain_root) if os.path.isdir(os.path.join(domain_root, d))]
    classes.sort()
    class_to_idx = {c: i for i, c in enumerate(classes)}

    names = []
    labels = []
    for cls in classes:
        cls_dir = os.path.join(domain_root, cls)
        files = []
        for pat in exts:
            files.extend(glob(os.path.join(cls_dir, pat)))
        files.sort()
        rels = [os.path.join(domain_name, cls, os.path.basename(f)) for f in files]
        names.extend(rels)
        labels.extend([class_to_idx[cls]] * len(rels))

    return names, labels
