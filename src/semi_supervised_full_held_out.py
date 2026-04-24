"""

Selects additional training data from held out subset.

Includes samples predicted with high probability.
A threshold that corresponds to x% precision on validation data is used to select
the samples (95%, for example).
A second threshold that corresponds to y% precision on validation data is also used 
(75%, for example) in case the first threshold didn't guarantee a minimum quantity
of samples for the class.


"""

import argparse
import json
import os
from collections import Counter

import numpy as np

def _count_class_occurrences(id2class, args):
    # Counts the occurrence of each class in the original training data
    with open(os.path.join(args.dataset_dir, "data/model_data/train_initial_model.json")) as f:
        train_data = json.load(f)
    occurrences = Counter()
    for sample in train_data:
        for label_id in sample['labels']:
            label = id2class[label_id]
            occurrences[label] += 1
    return occurrences


def _get_precision_threshold(label, classes_thresholds, precision):
    min_threshold = 1.1
    for class_threshold in classes_thresholds:
        if class_threshold['class_name'] != label:
            continue
        if class_threshold['precision'] >= precision:
            if class_threshold['threshold'] < min_threshold:
                min_threshold = class_threshold['threshold']
    return min_threshold


def _get_best_thresholds(classes_thresholds):
    """
        Gets the best threshold (greater f1) for each class
    """
    best_thresholds = {}
    best_f1s = {}
    for class_threshold in classes_thresholds:
        class_name = class_threshold['class_name']
        f1 = class_threshold['f1']
        threshold = class_threshold['threshold']
        try:
            current_best_t = best_thresholds[class_name]
        except KeyError:
            best_thresholds[class_name] = 1.1
            best_f1s[class_name] = -1
            current_best_f1 = best_f1s[class_name]
        else:
            current_best_f1 = best_f1s[class_name]
        finally:
            # Updates if f1 is better or if f1 is the same but threshold is lower
            if f1 > current_best_f1 or \
               f1 == current_best_f1 and threshold < current_best_t:
               best_thresholds[class_name] = threshold
               best_f1s[class_name] = f1

    print("Best thresholds (according to f1):")
    for class_name in best_thresholds.keys():
        print("\t{}: {} ({:.2f})".format(class_name, 
                                        best_thresholds[class_name], 
                                        best_f1s[class_name]*100))

    return best_thresholds


def _label_sample(pred_probs, id2class, best_thresholds, target_class):
    labels = []
    for label_id, prob in enumerate(pred_probs):
        class_name = id2class[label_id]
        if class_name == target_class:
            # Target class is always included
            labels.append(label_id)
            #print("target", class_name, prob)
        elif prob > best_thresholds[class_name]:
            labels.append(label_id)
            #print("other", class_name, prob)
    return labels


def _create_sample(sample_idx, sample_ids, held_out_data, new_labels):
    sample_id = sample_ids[sample_idx]
    original_sample = held_out_data[sample_id]
    original_sample['labels'] = new_labels
    return original_sample


def main(args):
    # Before anything, validate we're working from the correct dataset version
    with open(os.path.join(args.dataset_dir, "metadata.json")) as f:
        dataset_metadata = json.loads(f.read())
    assert dataset_metadata['dataset_version'] == args.original_dataset_version, "Dataset version is not {}".format(args.original_dataset_version)

    original_training_data_size = None
    original_held_out_data_size = None
    modified_training_data_size = None
    modified_held_out_data_size = None

    # Load id2class mapping
    with open(os.path.join(args.held_out_classification_dir, "id2class.json")) as f:
        id2class = json.load(f)
    id2class = {int(class_id): class_name for class_id, class_name in id2class.items()}
    class2id = {v: k for k, v in id2class.items()}

    # Loading all thresholds
    with open(os.path.join(args.held_out_classification_dir, "classes_thresholds.json")) as f:
        classes_thresholds = json.load(f)

    # Loading samples ids and prediction probabilities
    with open(os.path.join(args.held_out_classification_dir, "sample_ids.json")) as f:
        samples_ids = json.load(f)
    preds_probs = np.load(os.path.join(args.held_out_classification_dir, "probs.npy"))

    # Loading held out data
    with open(os.path.join(args.dataset_dir, "data/model_data/train_later_usage.json")) as f:
        held_out_data = json.load(f)

    original_held_out_data_size = len(held_out_data)

    # Indexing held out data by ID for fast retrieval
    held_out_data = {data['id']: data for data in held_out_data}

    # Get best thresholds for each class (according to f1 on validation)
    best_thresholds = _get_best_thresholds(classes_thresholds)

    """
    For each class (from least frequent to most frequent):
        - Sorts all samples according to threshold (descending order)
        - Includes all samples with probability that corresponds to minimum-precision
        - Includes all samples with probability that corresponds to backup-precision
          until min_samples are added for the class (including the samples added in
          the last step)

    
    Automatic labelling is done according to the following:
        - For the loop class, it's always included as label
        - All other classes are included if above the best validation threshold 
          according to f1

    """

    classes_occurrences = _count_class_occurrences(id2class, args)
    added_classes = Counter()
    new_samples = []
    later_usage = []
    already_added = set()
    # For each class
    for label, _ in sorted(classes_occurrences.items(), key=lambda x: x[1]):
        label_id = class2id[label]
        # Get threshold that corresponds to minimum precision
        minimum_threshold = _get_precision_threshold(label,
                                                     classes_thresholds,
                                                     args.minimum_precision)
        # Get threshold that corresponds to backup precision
        backup_threshold = _get_precision_threshold(label,
                                                    classes_thresholds,
                                                    args.backup_precision)
        #print("Thresholds {}: {}, {}".format(label, minimum_threshold, backup_threshold))

        # Sort samples by deacresing probability for this class
        for sample_idx in reversed(np.argsort(preds_probs[:, label_id])):
            sample_id = samples_ids[sample_idx]
            if sample_id in already_added:
                continue
            target_class_prob = preds_probs[sample_idx][label_id]
            # if above or equal to minimum threshold, adds the sample
            if target_class_prob >= minimum_threshold:
                # We are simulating a semi-supervised scenario, so labels
                # are automatically generated, we can't use the dataset's labels
                new_labels = _label_sample(pred_probs=preds_probs[sample_idx], 
                                           id2class=id2class, 
                                           best_thresholds=best_thresholds, 
                                           target_class=label)
                added_classes.update(new_labels)
                new_sample = _create_sample(sample_idx=sample_idx, 
                                            sample_ids=samples_ids, 
                                            held_out_data=held_out_data, 
                                            new_labels=new_labels)
                new_samples.append(new_sample)
                already_added.add(sample_id)
                continue
            # if minimum per class was not reached yet, and target class probability
            # is equal to or greater than the backup threshold, also adds it
            if added_classes[label_id] < args.min_samples and \
                target_class_prob >= backup_threshold:
                new_labels = _label_sample(pred_probs=preds_probs[sample_idx], 
                                           id2class=id2class, 
                                           best_thresholds=best_thresholds, 
                                           target_class=label)
                added_classes.update(new_labels)
                new_sample = _create_sample(sample_idx=sample_idx, 
                                            sample_ids=samples_ids, 
                                            held_out_data=held_out_data, 
                                            new_labels=new_labels)
                new_samples.append(new_sample)
                already_added.add(sample_id)
                continue     
    print("Total new samples: {} ({:.2f}% of total)".format(len(new_samples),
                                                            len(new_samples)/len(held_out_data)*100))
    # Samples that were not added, continue on the train later usage list
    for sample_id in samples_ids:
        if not sample_id in already_added:
            sample = held_out_data[sample_id]
            later_usage.append(sample)       
    print("Total later usage samples: ", len(later_usage))
    print("Added per class:")
    for class_name, class_count in added_classes.most_common():
        print("\t{} ({})".format(id2class[class_name], class_count))

    # Saving data
    with open(os.path.join(args.dataset_dir, "data/model_data/train_initial_model.json")) as f:
        train_data = json.load(f)
    original_training_data_size = len(train_data)

    new_training_data = train_data + new_samples

    modified_training_data_size = len(new_training_data)
    modified_held_out_data_size = len(later_usage)

    print("---------------------")
    print("Dataset size")
    print("\tOriginal training data: ", original_training_data_size)
    print("\tOriginal held out data: ", original_held_out_data_size)
    print("\tTotal: ", original_training_data_size + original_held_out_data_size)
    print("")
    print("\tNew training data: ", modified_training_data_size)
    print("\tNew held out data: ", modified_held_out_data_size)
    print("\tTotal: ", modified_training_data_size + modified_held_out_data_size)

    with open(os.path.join(args.dataset_dir, "data/model_data", 
                           "train_initial_model.json"), mode='w') as f:
        f.write(json.dumps(new_training_data))

    with open(os.path.join(args.dataset_dir, "data/model_data", 
                           "train_later_usage.json"), mode='w') as f:
        f.write(json.dumps(later_usage))

    # Changing dataset version
    dataset_metadata['dataset_version'] = args.new_dataset_version
    with open(os.path.join(args.dataset_dir, "metadata.json"), mode='w') as f:
        f.write(json.dumps(dataset_metadata))










if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--minimum-precision", default=0.95, type=float)
    parser.add_argument("--backup-precision", default=0.75, type=float)
    # Min samples per class
    parser.add_argument("--min-samples", default=50, type=int)

    parser.add_argument("--training-data-path")
    parser.add_argument("--held-out-data-path")
    parser.add_argument("--held-out-classification-dir")

    parser.add_argument("--dataset-dir")

    parser.add_argument("--original-dataset-version")
    parser.add_argument("--new-dataset-version")

    args = parser.parse_args()

    main(args)