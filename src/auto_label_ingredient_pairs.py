
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. File paths and settings
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_pair_review.csv"
)

MANUAL_LABELS_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_matching_tests.csv"
)

EMBEDDINGS_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_embeddings.npz"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_pair_auto_labels.csv"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

POSITIVE_SIMILARITY = 0.85



def normalize_name(name):
    name = str(name).lower().strip()
    name = re.sub(r"\s+", " ", name)
    return name


def get_tokens(name):
    return set(re.findall(r"\b\w+\b", normalize_name(name)))



def load_embeddings():
    data = np.load(EMBEDDINGS_FILE)

    names = data["ingredients"]
    vectors = data["embeddings"]

    return {
        normalize_name(name): vector
        for name, vector in zip(names, vectors)
    }




def calculate_features(name_a, name_b, embedding_lookup, model):

    name_a = normalize_name(name_a)
    name_b = normalize_name(name_b)

    tokens_a = get_tokens(name_a)
    tokens_b = get_tokens(name_b)

    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)

    word_overlap = len(intersection) / len(union) if union else 0

    smaller_count = min(len(tokens_a), len(tokens_b))

    token_containment = (
        len(intersection) / smaller_count
        if smaller_count else 0
    )

    character_similarity = SequenceMatcher(
        None, name_a, name_b
    ).ratio()

    length_difference = abs(len(name_a) - len(name_b))

    longest_length = max(len(name_a), len(name_b), 1)

    normalized_length_difference = (
        length_difference / longest_length
    )

    missing = [
        name for name in (name_a, name_b)
        if name not in embedding_lookup
    ]

    if missing:
        unique_missing = list(dict.fromkeys(missing))

        vectors = model.encode(
            unique_missing,
            normalize_embeddings=True
        )

        for name, vector in zip(unique_missing, vectors):
            embedding_lookup[name] = vector

    vector_a = embedding_lookup[name_a]
    vector_b = embedding_lookup[name_b]

    cosine_similarity = float(np.dot(vector_a, vector_b))

    return {
        "cosine_similarity": cosine_similarity,
        "embedding_distance": 1 - cosine_similarity,
        "word_overlap": word_overlap,
        "token_containment": token_containment,
        "character_similarity": character_similarity,
        "length_difference": length_difference,
        "normalized_length_difference": normalized_length_difference
    }


# --------------------------------------------------
# 5. Conservative automatic labelling
# --------------------------------------------------

def assign_label(name_a, name_b, features):

    name_a = normalize_name(name_a)
    name_b = normalize_name(name_b)

    similarity = features["cosine_similarity"]
    tokens_a = get_tokens(name_a)
    tokens_b = get_tokens(name_b)

    if name_a == name_b:
        return 1, "Identical ingredient names"

    # Explicit modifiers can distinguish ingredient types.
    distinguishing_modifiers = {
        "purple", "sweet", "coconut", "brown",
        "white", "black", "red", "green",
        "yellow", "ground", "whole", "dried",
        "fresh", "smoked", "powdered"
    }

    shared_tokens = tokens_a.intersection(tokens_b)

    modifiers_a = tokens_a.intersection(distinguishing_modifiers)
    modifiers_b = tokens_b.intersection(distinguishing_modifiers)

    if modifiers_a != modifiers_b and (
        modifiers_a or modifiers_b
    ):
        return 0, "Different ingredient modifiers"

    # High similarity and matching wording.
    if (
        similarity >= POSITIVE_SIMILARITY
        and features["token_containment"] >= 0.80
    ):
        return 1, "High semantic similarity and token containment"

    # Very low similarity is not sufficient to reject a pair.
    # Unknown pairs remain REVIEW.
    return "REVIEW", "Requires further verification"



def load_manual_labels():

    manual_df = pd.read_csv(MANUAL_LABELS_FILE)

    manual_df["ingredient_a"] = (
        manual_df["ingredient_a"].apply(normalize_name)
    )

    manual_df["ingredient_b"] = (
        manual_df["ingredient_b"].apply(normalize_name)
    )

    manual_labels = {}

    for _, row in manual_df.iterrows():

        pair = tuple(sorted([
            row["ingredient_a"],
            row["ingredient_b"]
        ]))

        manual_labels[pair] = int(row["label"])

    return manual_labels



def main():

    df = pd.read_csv(INPUT_FILE)

    embedding_lookup = load_embeddings()

    manual_labels = load_manual_labels()

    model = SentenceTransformer(MODEL_NAME)

    results = []

    for _, row in df.iterrows():

        name_a = normalize_name(row["ingredient_a"])
        name_b = normalize_name(row["ingredient_b"])

        features = calculate_features(
            name_a,
            name_b,
            embedding_lookup,
            model
        )

        auto_label, reason = assign_label(
            name_a,
            name_b,
            features
        )

        pair = tuple(sorted([name_a, name_b]))

        manual_label = manual_labels.get(pair)

        final_label = (
            manual_label
            if manual_label is not None
            else auto_label
        )

        results.append({
            "ingredient_a": name_a,
            "ingredient_b": name_b,
            **features,
            "auto_label": auto_label,
            "manual_label": (
                manual_label if manual_label is not None else ""
            ),
            "final_label": final_label,
            "reason": (
                "Trusted manual label"
                if manual_label is not None
                else reason
            )
        })

    output_df = pd.DataFrame(results)

    output_df.to_csv(OUTPUT_FILE, index=False)

    print("\nINGREDIENT LABELLING RESULTS")
    print("-" * 45)

    print(f"Total pairs: {len(output_df)}")

    print(
        "Manually labelled:",
        output_df["manual_label"].ne("").sum()
    )

    print(
        "Automatically equivalent:",
        (output_df["auto_label"] == 1).sum()
    )

    print(
        "Automatically different:",
        (output_df["auto_label"] == 0).sum()
    )

    print(
        "Needs review:",
        (output_df["final_label"] == "REVIEW").sum()
    )

    print(f"\nSaved results to:\n{OUTPUT_FILE}")

    print("\nFirst 15 results:")

    print(
        output_df[
            [
                "ingredient_a",
                "ingredient_b",
                "auto_label",
                "manual_label",
                "final_label"
            ]
        ].head(15).to_string(index=False)
    )


if __name__ == "__main__":
    main()
