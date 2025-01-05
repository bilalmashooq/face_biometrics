#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidated Face Verification Script

Functions:
1) assign_ids_to_dataset        (Rename subfolders by unique ID)
2) generate_results_single_model (Generate pairs & distances for one model)
3) generate_results_two_models   (Generate pairs & distances for two models)
4) Evaluation/metrics utilities
5) High-level evaluate scripts (plot ROC, compute EER, confusion matrix, etc.)

Author:  Muhammad Bilal
Date:    27/12/2024
"""

import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DeepFace can be commented out if not needed
from deepface import DeepFace

from sklearn.metrics import roc_curve, auc


###############################################################################
# 1) Renaming Subfolders with Unique IDs
###############################################################################
def assign_ids_to_dataset(dataset_path, start_id=1000):
    """
    Renames each subfolder in `dataset_path` to `<ID>_<originalFolderName>`.
    Returns a dictionary mapping {ID: "ID_OriginalName"}.
    """
    if not os.path.isdir(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")

    unique_id_map = {}
    current_id = start_id

    # List all subfolders
    folders = [
        f for f in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, f))
    ]

    for folder in sorted(folders):
        old_path = os.path.join(dataset_path, folder)
        new_folder_name = f"{current_id}_{folder}"
        new_path = os.path.join(dataset_path, new_folder_name)

        # Rename the folder
        os.rename(old_path, new_path)

        # Store mapping
        unique_id_map[current_id] = new_folder_name
        print(f"Renamed '{folder}' -> '{new_folder_name}' (ID={current_id})")

        current_id += 1

    return unique_id_map


###############################################################################
# 2) Generate Pairs & Distances for One Model
###############################################################################
def get_embedding(img_path, model_name='Facenet', enforce_detection=False):
    """
    Computes face embedding for a single image using DeepFace.
    Returns a NumPy array or None if it fails.
    """
    try:
        resp = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            enforce_detection=enforce_detection
        )
        # 'resp' is usually a list of dicts if enforce_detection=False
        # e.g. [{'embedding': [...], 'facial_area': {...}, ...}]
        if resp and len(resp) > 0:
            return np.array(resp[0]['embedding'], dtype=np.float32)
    except Exception as e:
        print(f"[{model_name}] Warning: embedding failed for {img_path}: {e}")

    return None


def euclidean_distance(emb1, emb2):
    """
    Computes Euclidean distance between two embeddings.
    """
    return np.linalg.norm(emb1 - emb2)


def generate_results_single_model(
        dataset_dir,
        results_csv="results.csv",
        model_name="Facenet",
        enforce_detection=False,
        max_same_pairs=5,
        max_diff_pairs=5
):
    """
    Traverses the dataset to gather embeddings for one model, then
    generates same-person (label=1) and different-person (label=0) pairs,
    computes distances, and writes to a CSV file.

    dataset_dir:     Path to the dataset root folder.
    results_csv:     Output CSV filename.
    model_name:      DeepFace model (e.g., "Facenet", "ArcFace", etc.).
    enforce_detection: if True, face detection must succeed or it raises errors.
    max_same_pairs:  max number of genuine pairs per person.
    max_diff_pairs:  max number of imposter pairs per person.
    """
    # 1) Gather embeddings
    embeddings_dict = {}
    person_folders = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for person in person_folders:
        folder_path = os.path.join(dataset_dir, person)
        emb_list = []
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, img_file)
                emb = get_embedding(img_path, model_name=model_name,
                                    enforce_detection=enforce_detection)
                if emb is not None:
                    emb_list.append(emb)

        if emb_list:
            embeddings_dict[person] = emb_list
        else:
            print(f"No valid embeddings for {person}")

    persons_data = list(embeddings_dict.items())
    print(f"Model={model_name}: Found embeddings for {len(persons_data)} persons.\n")

    # 2) Build pairs (same/different)
    results = []

    # A) Same-person pairs (label=1)
    for person, emb_list in persons_data:
        if len(emb_list) < 2:
            continue
        same_count = 0
        all_indices = list(range(len(emb_list)))
        attempts = 0
        while same_count < max_same_pairs and attempts < 1000:
            i1, i2 = random.sample(all_indices, 2)
            dist = euclidean_distance(emb_list[i1], emb_list[i2])
            results.append((dist, 1))
            same_count += 1
            attempts += 1

    # B) Different-person pairs (label=0)
    if len(persons_data) > 1:
        for idx, (personA, embA_list) in enumerate(persons_data):
            diff_count = 0
            attempts = 0
            while diff_count < max_diff_pairs and attempts < 1000:
                other_idx = random.randint(0, len(persons_data) - 1)
                if other_idx == idx:
                    attempts += 1
                    continue

                _, embB_list = persons_data[other_idx]
                if not embA_list or not embB_list:
                    attempts += 1
                    continue

                embA = random.choice(embA_list)
                embB = random.choice(embB_list)
                dist = euclidean_distance(embA, embB)
                results.append((dist, 0))
                diff_count += 1
                attempts += 1

    # 3) Write to CSV
    with open(results_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        for dist, lbl in results:
            writer.writerow([dist, lbl])

    print(f"Wrote {len(results)} pairs to '{results_csv}' (Model={model_name}).")


###############################################################################
# 3) Generate Pairs & Distances for Two Models
###############################################################################
def generate_results_two_models(
        dataset_dir,
        results_csv_1="results_model1.csv",
        results_csv_2="results_model2.csv",
        model1_name="Facenet",
        model2_name="VGG-Face",
        enforce_detection=False,
        max_same_pairs=5,
        max_diff_pairs=5
):
    """
    Gathers embeddings for two different DeepFace models, then
    generates same/different pairs and writes two CSV files:
      results_csv_1  (distances for model1)
      results_csv_2  (distances for model2)

    dataset_dir:       path to dataset
    results_csv_1:     output CSV for model1
    results_csv_2:     output CSV for model2
    model1_name:       e.g. "Facenet", "ArcFace", etc.
    model2_name:       e.g. "VGG-Face", etc.
    enforce_detection:  if True, face detection is strictly enforced
    max_same_pairs:     max same-person pairs per person
    max_diff_pairs:     max different-person pairs per person
    """
    embeddings_dict_1 = {}
    embeddings_dict_2 = {}

    # 1) Gather embeddings for each model
    person_folders = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ]

    for person in person_folders:
        folder_path = os.path.join(dataset_dir, person)
        emb_list_1, emb_list_2 = [], []

        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(folder_path, img_file)

                emb1 = get_embedding(img_path, model_name=model1_name,
                                     enforce_detection=enforce_detection)
                emb2 = get_embedding(img_path, model_name=model2_name,
                                     enforce_detection=enforce_detection)

                # Only add if we actually get an embedding
                if emb1 is not None:
                    emb_list_1.append(emb1)
                if emb2 is not None:
                    emb_list_2.append(emb2)

        if emb_list_1:
            embeddings_dict_1[person] = emb_list_1
        if emb_list_2:
            embeddings_dict_2[person] = emb_list_2

    persons_data_1 = list(embeddings_dict_1.items())
    persons_data_2 = list(embeddings_dict_2.items())

    print(f"[Model1={model1_name}] Found embeddings for {len(persons_data_1)} persons.")
    print(f"[Model2={model2_name}] Found embeddings for {len(persons_data_2)} persons.\n")

    # 2) Build pairs & compute distances for each model
    results_model1, results_model2 = [], []

    # A) Same-person pairs
    # Model1
    for person, emb_list in persons_data_1:
        if len(emb_list) < 2:
            continue
        same_count = 0
        all_indices = list(range(len(emb_list)))
        attempts = 0
        while same_count < max_same_pairs and attempts < 1000:
            i1, i2 = random.sample(all_indices, 2)
            dist = euclidean_distance(emb_list[i1], emb_list[i2])
            results_model1.append((dist, 1))
            same_count += 1
            attempts += 1

    # Model2
    for person, emb_list in persons_data_2:
        if len(emb_list) < 2:
            continue
        same_count = 0
        all_indices = list(range(len(emb_list)))
        attempts = 0
        while same_count < max_same_pairs and attempts < 1000:
            i1, i2 = random.sample(all_indices, 2)
            dist = euclidean_distance(emb_list[i1], emb_list[i2])
            results_model2.append((dist, 1))
            same_count += 1
            attempts += 1

    # B) Different-person pairs
    # Model1
    if len(persons_data_1) > 1:
        for idx, (_, embA_list) in enumerate(persons_data_1):
            diff_count = 0
            attempts = 0
            while diff_count < max_diff_pairs and attempts < 1000:
                other_idx = random.randint(0, len(persons_data_1) - 1)
                if other_idx == idx:
                    attempts += 1
                    continue
                _, embB_list = persons_data_1[other_idx]
                if not embA_list or not embB_list:
                    attempts += 1
                    continue

                embA = random.choice(embA_list)
                embB = random.choice(embB_list)
                dist = euclidean_distance(embA, embB)
                results_model1.append((dist, 0))
                diff_count += 1
                attempts += 1

    # Model2
    if len(persons_data_2) > 1:
        for idx, (_, embA_list) in enumerate(persons_data_2):
            diff_count = 0
            attempts = 0
            while diff_count < max_diff_pairs and attempts < 1000:
                other_idx = random.randint(0, len(persons_data_2) - 1)
                if other_idx == idx:
                    attempts += 1
                    continue
                _, embB_list = persons_data_2[other_idx]
                if not embA_list or not embB_list:
                    attempts += 1
                    continue

                embA = random.choice(embA_list)
                embB = random.choice(embB_list)
                dist = euclidean_distance(embA, embB)
                results_model2.append((dist, 0))
                diff_count += 1
                attempts += 1

    # 3) Write to CSV
    with open(results_csv_1, 'w', newline='') as f1:
        w1 = csv.writer(f1)
        for dist, lbl in results_model1:
            w1.writerow([dist, lbl])

    with open(results_csv_2, 'w', newline='') as f2:
        w2 = csv.writer(f2)
        for dist, lbl in results_model2:
            w2.writerow([dist, lbl])

    print(f"Model1: Wrote {len(results_model1)} pairs to '{results_csv_1}'.")
    print(f"Model2: Wrote {len(results_model2)} pairs to '{results_csv_2}'.\nDone.")


###############################################################################
# 4) Evaluation & Metrics Utilities
###############################################################################
def compute_far_frr(distances, labels):
    """
    Calculate FAR (False Acceptance Rate) and FRR (False Rejection Rate)
    for a range of thresholds derived from the distances.
    labels: 1 for genuine (same-person), 0 for imposter (different-person).

    Returns:
      thresholds: array of unique distance values (ascending)
      FAR:        array of false acceptance rates
      FRR:        array of false rejection rates
    """
    sorted_distances = np.sort(distances)
    thresholds = np.unique(sorted_distances)

    FAR_list = []
    FRR_list = []

    for th in thresholds:
        # "Predict same" if distance < th
        preds_same = (distances < th)

        # Among label=0, how many are incorrectly accepted?
        false_accepts = np.sum((preds_same == True) & (labels == 0))
        total_imposter = np.sum(labels == 0)
        far = false_accepts / float(total_imposter) if total_imposter > 0 else 0.0

        # Among label=1, how many are incorrectly rejected?
        false_rejects = np.sum((preds_same == False) & (labels == 1))
        total_genuine = np.sum(labels == 1)
        frr = false_rejects / float(total_genuine) if total_genuine > 0 else 0.0

        FAR_list.append(far)
        FRR_list.append(frr)

    return thresholds, np.array(FAR_list), np.array(FRR_list)


def find_eer(far, frr, thresholds):
    """
    Find the Equal Error Rate (EER) where FAR and FRR intersect or are closest.
    Returns (eer_value, threshold_at_eer).
    """
    diff = np.abs(far - frr)
    idx = np.argmin(diff)
    eer_value = (far[idx] + frr[idx]) / 2.0
    threshold_at_eer = thresholds[idx]
    return eer_value, threshold_at_eer


def plot_roc_curve(distances, labels, model_name='Model'):
    """
    Plots the ROC curve (TPR vs FPR) for a given set of distances/labels.
    labels=1 (same), 0 (different). Smaller distance => more likely same.
    """
    # Convert distances to "scores" (higher => more likely same)
    scores = -distances

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')


def compute_confusion_matrix(distances, labels, threshold):
    """
    Compute confusion matrix (TP, FP, TN, FN) at a given threshold.
    'Predict same' if distance < threshold.
    labels=1 => same-person, 0 => different-person
    """
    preds_same = (distances < threshold)
    tp = np.sum((preds_same == True) & (labels == 1))
    fp = np.sum((preds_same == True) & (labels == 0))
    tn = np.sum((preds_same == False) & (labels == 0))
    fn = np.sum((preds_same == False) & (labels == 1))
    return tp, fp, tn, fn


def compute_classification_metrics(tp, fp, tn, fn):
    """
    Compute Accuracy, Precision, Recall, Specificity, F1.
    """
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / float(total) if total > 0 else 0.0
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1
    }


def find_best_threshold_for_f1(distances, labels):
    """
    Brute force all unique distance values as thresholds, pick the one
    that yields the best F1 score.
    Returns (best_threshold, best_f1, best_metrics)
    """
    sorted_distances = np.sort(distances)
    thresholds = np.unique(sorted_distances)

    best_th = None
    best_f1 = -1
    best_metrics = {}

    for th in thresholds:
        tp, fp, tn, fn = compute_confusion_matrix(distances, labels, th)
        metrics = compute_classification_metrics(tp, fp, tn, fn)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_th = th
            best_metrics = metrics

    return best_th, best_f1, best_metrics


###############################################################################
# 5) High-Level Evaluation Examples
###############################################################################
def evaluate_single_csv(csv_file, model_name="Model"):
    """
    Loads (distance, label) from CSV, computes:
      - thresholds, FAR, FRR
      - EER
      - Plots ROC, FAR/FRR vs. threshold
    """
    data = np.loadtxt(csv_file, delimiter=",")
    distances = data[:, 0]
    labels = data[:, 1].astype(int)

    thresholds, FAR, FRR = compute_far_frr(distances, labels)
    eer_value, eer_threshold = find_eer(FAR, FRR, thresholds)

    # Plot ROC
    plt.figure(figsize=(8, 6))
    plot_roc_curve(distances, labels, model_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title(f"ROC Curve ({model_name})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot FAR/FRR
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, FAR, label='FAR', linestyle='--', color='orange')
    plt.plot(thresholds, FRR, label='FRR', linestyle='-', color='blue')
    plt.axvline(x=eer_threshold, linestyle='--', color='green', label='EER Threshold')
    plt.title(f"FAR and FRR vs. Threshold ({model_name})")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"{model_name}: EER = {eer_value:.3f} at threshold = {eer_threshold:.3f}")


def compare_two_models(csv_file_1, csv_file_2, model1_name="Model1", model2_name="Model2"):
    """
    Compare two models by loading their CSV files (distance,label),
    plot combined ROC, compute EER for each.
    """
    data1 = np.loadtxt(csv_file_1, delimiter=",")
    dist1 = data1[:, 0]
    labels1 = data1[:, 1].astype(int)

    data2 = np.loadtxt(csv_file_2, delimiter=",")
    dist2 = data2[:, 0]
    labels2 = data2[:, 1].astype(int)

    # Compute EER for Model1
    th1, FAR1, FRR1 = compute_far_frr(dist1, labels1)
    eer1, eer_th1 = find_eer(FAR1, FRR1, th1)

    # Compute EER for Model2
    th2, FAR2, FRR2 = compute_far_frr(dist2, labels2)
    eer2, eer_th2 = find_eer(FAR2, FRR2, th2)

    # Plot combined ROC
    plt.figure(figsize=(8, 6))
    plot_roc_curve(dist1, labels1, model1_name)
    plot_roc_curve(dist2, labels2, model2_name)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"{model1_name}: EER={eer1:.3f} at threshold={eer_th1:.3f}")
    print(f"{model2_name}: EER={eer2:.3f} at threshold={eer_th2:.3f}")


def evaluate_confusion_matrix_and_metrics(csv_file, threshold):
    """
    Loads (distance, label) from CSV, computes confusion matrix at a given threshold,
    prints metrics and shows a heatmap.
    """
    data = np.loadtxt(csv_file, delimiter=",")
    distances = data[:, 0]
    labels = data[:, 1].astype(int)

    tp, fp, tn, fn = compute_confusion_matrix(distances, labels, threshold)
    metrics = compute_classification_metrics(tp, fp, tn, fn)

    # Print
    print(f"\nConfusion Matrix (Threshold={threshold:.3f}):")
    print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}")

    # Plot heatmap
    cm = np.array([[tn, fp],
                   [fn, tp]])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred=0', 'Pred=1'],
                yticklabels=['True=0', 'True=1'])
    plt.title(f"Confusion Matrix @ threshold={threshold:.3f}")
    plt.show()


###############################################################################
# EXAMPLE MAIN SECTIONS (Uncomment if you want to run as script)
###############################################################################
if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example 1: Assign IDs to subfolders
    # -------------------------------------------------------------------------
    # dataset_path = r"C:\path\to\lfw_subset"
    # mapping = assign_ids_to_dataset(dataset_path, start_id=1000)
    # print("Folder renaming complete. ID Mapping:", mapping)

    # -------------------------------------------------------------------------
    # Example 2: Generate results for a single model
    # -------------------------------------------------------------------------
    DATASET_DIR = r"C:\Users\muham\PycharmProjects\pythonProject\Verification\lfw_subset"
    generate_results_single_model(
        dataset_dir=DATASET_DIR,
        results_csv="results_model1.csv",
        model_name="Facenet",   # or "VGG-Face", "ArcFace", etc.
        enforce_detection=False,
        max_same_pairs=5,
        max_diff_pairs=5
    )
    generate_results_single_model(
        dataset_dir=DATASET_DIR,
        results_csv="results_model2.csv",
        model_name="ArcFace",  # or "VGG-Face", "ArcFace", etc.
        enforce_detection=False,
        max_same_pairs=5,
        max_diff_pairs=5
    )
    # -------------------------------------------------------------------------
    # Example 3: Generate results for two models
    # -------------------------------------------------------------------------
    DATASET_DIR = r"C:\Users\muham\PycharmProjects\pythonProject\Verification\lfw_subset"
    generate_results_two_models(
        dataset_dir=DATASET_DIR,
        results_csv_1="results_model1.csv",
        results_csv_2="results_model2.csv",
        model1_name="Facenet",
        model2_name="VGG-Face",
        enforce_detection=False,
        max_same_pairs=5,
        max_diff_pairs=5
    )

    # -------------------------------------------------------------------------
    # Example 4: Evaluate single CSV (plot ROC, find EER, plot FAR/FRR)
    # -------------------------------------------------------------------------
    evaluate_single_csv("results.csv", model_name="Facenet")

    # -------------------------------------------------------------------------
    # Example 5: Compare two models
    # -------------------------------------------------------------------------
    compare_two_models("results_model1.csv", "results_model2.csv",
                       model1_name="Facenet",
                       model2_name="VGG-Face")

    # -------------------------------------------------------------------------
    # Example 6: Confusion matrix, classification metrics at a chosen threshold
    # -------------------------------------------------------------------------
    evaluate_confusion_matrix_and_metrics("results.csv", threshold=10.0)

   # Remove this 'pass' when you uncomment any of the examples
