
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.spatial.distance import cdist
from deepface import DeepFace

# ======================================
#   DOWNLOAD & EXTRACT LFW DATASET
# ======================================
# if not os.path.exists("lfw-deepfunneled"):
#     !wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
#     !tar xzf lfw-deepfunneled.tgz

# ======================================
#   CREATE CSV OF PERSON IMAGE COUNTS
# ======================================
def create_lfw_csv(dataset_dir="lfw-deepfunneled", csv_output="lfw_allnames.csv"):
    """
    Scans the 'lfw-deepfunneled' folder, counts how many images each person has,
    and saves the result to a CSV file with columns: ['person', 'images'].
    """
    rows = []
    for person_name in sorted(os.listdir(dataset_dir)):
        person_path = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_path):
            # Count valid image files
            img_count = sum(
                img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
                for img in os.listdir(person_path)
            )
            rows.append((person_name, img_count))

    df = pd.DataFrame(rows, columns=["person", "images"])
    df.to_csv(csv_output, index=False)
    print(f"[INFO] CSV created: {csv_output} with {len(df)} entries.")
    return df

_ = create_lfw_csv("lfw-deepfunneled", "lfw_allnames.csv")


# ======================================
#       FUNCTION DEFINITIONS
# ======================================
def filter_dataset(csv_path, min_images=35, max_images=80):
    """
    Filter the dataset based on the number of images per person.
    """
    print("\n[STEP] Filtering dataset based on image count criteria...")
    all_people = pd.read_csv(csv_path)
    filtered_people = all_people[
        (all_people['images'] >= min_images) & (all_people['images'] <= max_images)
    ].copy()
    total_images = filtered_people['images'].sum()
    total_people = len(filtered_people)

    print(f"Filtered People (Sample):\n{filtered_people.head()}")
    print(f"\nTotal People in filtered set: {total_people}")
    print(f"Total Images in filtered set: {total_images}\n")

    return filtered_people


def compute_embeddings(dataset_path, model_name="Facenet", people_list=None, enforce_detection=False):
    """
    Compute facial embeddings for each selected person using DeepFace.
    If 'people_list' is provided, only those persons are processed.
    """
    embeddings_dict = {}
    person_to_label = {}
    label_to_person = {}
    current_label = 0

    print(f"\n[STEP] Computing embeddings using {model_name} model...")

    # If no subset provided, do all
    if people_list is None:
        people_list = sorted(os.listdir(dataset_path))

    for person in sorted(people_list):
        person_path = os.path.join(dataset_path, person)
        if not os.path.isdir(person_path):
            continue  # Skip if not a directory

        embeddings = []
        for img in os.listdir(person_path):
            img_path = os.path.join(person_path, img)
            # Process only valid image files
            if img.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=img_path,
                        model_name=model_name,
                        enforce_detection=enforce_detection
                    )
                    if embedding_objs:
                        emb = embedding_objs[0]['embedding']
                        embeddings.append(emb)
                except Exception as e:
                    print(f"[WARNING] Error processing {img_path}: {e}")

        if embeddings:
            embeddings_dict[person] = embeddings
            person_to_label[person] = current_label
            label_to_person[current_label] = person
            current_label += 1

    print(f"[INFO] Total persons with embeddings: {len(embeddings_dict)}")
    return embeddings_dict, person_to_label, label_to_person


def evaluate_identification(embeddings_dict, person_to_label):
    """
    Evaluate identification performance by doing a proper leave-one-out comparison:
    For each embedding, we compute its distance to all embeddings EXCEPT itself.
    """
    # Flatten all embeddings
    combined_embeddings = []
    combined_labels = []
    for person, embs in embeddings_dict.items():
        label = person_to_label[person]
        for e in embs:
            combined_embeddings.append(e)
            combined_labels.append(label)

    combined_embeddings = np.array(combined_embeddings)
    combined_labels = np.array(combined_labels)

    ground_truth = []
    predictions = []

    for test_idx in range(len(combined_embeddings)):
        test_emb = combined_embeddings[test_idx]
        test_label = combined_labels[test_idx]

        # Distances to all embeddings
        distances = cdist([test_emb], combined_embeddings, metric='euclidean')[0]
        # Set the distance to itself to a very large number so it won't pick itself
        distances[test_idx] = np.inf

        # Find nearest index
        nearest_idx = np.argmin(distances)
        predicted_label = combined_labels[nearest_idx]

        ground_truth.append(test_label)
        predictions.append(predicted_label)

    return ground_truth, predictions, combined_embeddings, combined_labels


def plot_confusion_matrix(true_labels, pred_labels, label_to_person, model_name="Facenet"):
    """
    Plot a confusion matrix using seaborn.
    """
    print(f"\n[STEP] Plotting confusion matrix for {model_name}...")
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_to_person[i] for i in range(len(label_to_person))],
                yticklabels=[label_to_person[i] for i in range(len(label_to_person))])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_cmc_curve(true_labels, distances, labels, model_name="Facenet", max_rank=5):
    """
    Plot the Cumulative Match Characteristic (CMC) curve.
    """
    print(f"[STEP] Plotting CMC curve for {model_name}...")
    ranks = np.zeros(len(true_labels))

    # For each test embedding, find the rank of the correct label
    for idx, (label, dist_row) in enumerate(zip(true_labels, distances)):
        sorted_indices = np.argsort(dist_row)
        sorted_labels = labels[sorted_indices]
        # Exclude distance to itself if it occurs; or ensure we do leave-one-out in calls
        # We'll assume we already set self-distance to np.inf if we use the same matrix.
        # But let's do a small check:
        if sorted_labels[0] == label:
            # rank=1
            rank = 1
        else:
            # find where label occurs
            pos = np.where(sorted_labels == label)[0]
            if len(pos) > 0:
                rank = pos[0] + 1
            else:
                rank = max_rank + 1
        ranks[idx] = rank

    # Compute CMC up to max_rank
    cmc_vals = []
    for r in range(1, max_rank + 1):
        cmc_rate = np.mean(ranks <= r)
        cmc_vals.append(cmc_rate)

    # Plot
    plt.figure(figsize=(6,4))
    plt.plot(range(1, max_rank + 1), cmc_vals, marker='o', color='b')
    plt.title(f"CMC Curve - {model_name}")
    plt.xlabel("Rank")
    plt.ylabel("Identification Rate")
    plt.xticks(range(1, max_rank + 1))
    plt.grid(True)
    plt.ylim([0,1])
    plt.show()


# ======================================
#             MAIN SCRIPT
# ======================================
def main():
    # 1) Filter the dataset to a certain range of images
    CSV_PATH = "lfw_allnames.csv"
    filtered = filter_dataset(CSV_PATH, min_images=35, max_images=80)
    # This returns a DataFrame with columns ["person", "images"] in the desired range.

    # 2) From that filtered set, build a list of person folders
    dataset_dir = "lfw-deepfunneled"
    selected_people = filtered['person'].tolist()

    # 3) Compute embeddings for chosen model
    model_name = "ArcFace"
    embeddings_dict, person_to_label, label_to_person = compute_embeddings(
        dataset_dir,
        model_name=model_name,
        people_list=selected_people,
        enforce_detection=False  # LFW is aligned
    )

    # 4) Evaluate identification (now ignoring the sample itself = more realistic)
    ground_truth, predictions, all_embeddings, all_labels = evaluate_identification(
        embeddings_dict, person_to_label
    )

    # 5) Print accuracy and classification report
    accuracy = accuracy_score(ground_truth, predictions)
    print(f"\n[RESULT] Identification Accuracy ({model_name}): {accuracy*100:.2f}%")

    sorted_persons = [label_to_person[i] for i in range(len(label_to_person))]
    print("\nClassification Report:")
    print(classification_report(ground_truth, predictions, target_names=sorted_persons))

    # 6) Plot confusion matrix
    plot_confusion_matrix(ground_truth, predictions, label_to_person, model_name)

    # 7) Plot CMC curve
    # Build distance matrix again, but set diagonal to infinity
    # for the same leave-one-out approach
    distances = cdist(all_embeddings, all_embeddings, metric='euclidean')
    np.fill_diagonal(distances, np.inf)  # important step
    plot_cmc_curve(ground_truth, distances, all_labels, model_name=model_name, max_rank=5)


if __name__ == "__main__":
    main()
