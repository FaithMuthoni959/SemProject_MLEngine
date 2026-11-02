
import re
import random
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer



BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_embeddings.npz"
)

EXISTING_REVIEW_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_pair_review.csv"
)

MANUAL_LABELS_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_matching_tests.csv"
)

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed" /
    "ingredient_pair_candidates_200.csv"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TARGET_PAIRS = 200
ALIAS_TARGET = 50
TOP_NEIGHBOURS = 20
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Load existing embeddings
# --------------------------------------------------

def load_embeddings():

    data = np.load(EMBEDDINGS_FILE)

    names = data["ingredients"]
    vectors = data["embeddings"]

    lookup = {
        str(name).lower().strip(): vector
        for name, vector in zip(names, vectors)
    }

    print(f"Loaded {len(lookup)} ingredient embeddings.")

    return lookup


# --------------------------------------------------
# 3. Load existing pairs to avoid duplicates
# --------------------------------------------------

def get_existing_pairs():

    existing_pairs = set()

    files = [
        EXISTING_REVIEW_FILE,
        MANUAL_LABELS_FILE
    ]

    for file_path in files:

        if not file_path.exists():
            continue

        df = pd.read_csv(file_path)

        for _, row in df.iterrows():

            a = str(row["ingredient_a"]).lower().strip()
            b = str(row["ingredient_b"]).lower().strip()

            if a and b and a != "nan" and b != "nan":
                existing_pairs.add(tuple(sorted((a, b))))

    print(f"Found {len(existing_pairs)} previously used pairs.")

    return existing_pairs


# --------------------------------------------------
# 4. Generate candidates from parenthetical names
# --------------------------------------------------

def generate_alias_candidates(ingredient_names):

    candidates = {}

    for name in ingredient_names:

        match = re.fullmatch(r"(.+?)\s*\(([^()]+)\)", name)

        if not match:
            continue

        main_name = match.group(1).strip()
        alias_name = match.group(2).strip()

        if not main_name or not alias_name:
            continue

        if main_name == alias_name:
            continue

        pair = tuple(sorted((main_name, alias_name)))

        candidates[pair] = "Parenthetical name candidate"

    return candidates


# --------------------------------------------------
# 5. Generate semantic-neighbour candidates
# --------------------------------------------------

def generate_semantic_candidates(names, vectors):

    similarity_matrix = np.dot(vectors, vectors.T)

    candidates = {}

    for index, name in enumerate(names):

        similarities = similarity_matrix[index]

        # Find the nearest ingredients, excluding itself.
        neighbour_indices = np.argsort(similarities)[
            -(TOP_NEIGHBOURS + 1):-1
        ]

        for neighbour_index in neighbour_indices:

            other_name = names[neighbour_index]

            if name == other_name:
                continue

            similarity = float(similarities[neighbour_index])

            if similarity < 0.30:
                continue

            pair = tuple(sorted((name, other_name)))

            candidates[pair] = similarity

    return candidates


# --------------------------------------------------
# 6. Calculate alias-pair similarity
# --------------------------------------------------

def calculate_alias_similarities(alias_candidates, lookup, model):

    missing_names = set()

    for pair in alias_candidates:

        for name in pair:

            if name not in lookup:
                missing_names.add(name)

    if missing_names:

        missing_names = sorted(missing_names)

        new_vectors = model.encode(
            missing_names,
            normalize_embeddings=True
        )

        for name, vector in zip(missing_names, new_vectors):
            lookup[name] = vector

    results = {}

    for pair, reason in alias_candidates.items():

        a, b = pair

        similarity = float(np.dot(lookup[a], lookup[b]))

        results[pair] = {
            "cosine_similarity": similarity,
            "candidate_source": reason
        }

    return results


# --------------------------------------------------
# 7. Select exactly 200 new pairs
# --------------------------------------------------

def select_candidates(alias_candidates, semantic_candidates, existing):

    random.seed(RANDOM_SEED)

    alias_available = [
        pair for pair in alias_candidates
        if pair not in existing
    ]

    semantic_available = [
        pair for pair in semantic_candidates
        if pair not in existing
    ]

    random.shuffle(alias_available)
    random.shuffle(semantic_available)

    selected = {}

    # Prioritize parenthetical-name candidates.
    alias_count = min(ALIAS_TARGET, len(alias_available))

    for pair in alias_available[:alias_count]:

        selected[pair] = {
            "candidate_source": "Parenthetical name candidate"
        }

    # Fill the remaining spaces with semantic neighbours.
    remaining = TARGET_PAIRS - len(selected)

    for pair in semantic_available:

        if remaining == 0:
            break

        if pair not in selected:

            selected[pair] = {
                "candidate_source": "Semantic neighbour candidate"
            }

            remaining -= 1

    # If semantic candidates overlap with alias candidates,
    # or there are not enough, try additional alias candidates.
    if remaining > 0:

        for pair in alias_available[alias_count:]:

            if remaining == 0:
                break

            if pair not in selected:

                selected[pair] = {
                    "candidate_source": "Parenthetical name candidate"
                }

                remaining -= 1

    if len(selected) < TARGET_PAIRS:

        raise ValueError(
            f"Only {len(selected)} new pairs were found. "
            "More candidates are needed."
        )

    return selected




def main():

    random.seed(RANDOM_SEED)

    lookup = load_embeddings()

    existing_pairs = get_existing_pairs()

    names = list(lookup.keys())

    vectors = np.array([
        lookup[name] for name in names
    ])

    print("\nGenerating parenthetical-name candidates...")

    alias_candidates = generate_alias_candidates(names)

    print(f"Found {len(alias_candidates)} alias candidates.")

    print("\nGenerating semantic-neighbour candidates...")

    semantic_candidates = generate_semantic_candidates(
        names,
        vectors
    )

    print(
        f"Found {len(semantic_candidates)} semantic candidates."
    )

    selected = select_candidates(
        alias_candidates,
        semantic_candidates,
        existing_pairs
    )

    # Calculate similarities for all selected pairs.
    model = SentenceTransformer(MODEL_NAME)

    alias_similarities = calculate_alias_similarities(
        alias_candidates,
        lookup,
        model
    )

    results = []

    for pair, information in selected.items():

        a, b = pair

        if pair in alias_similarities:

            similarity = alias_similarities[pair][
                "cosine_similarity"
            ]

        else:

            similarity = float(np.dot(lookup[a], lookup[b]))

        results.append({
            "ingredient_a": a,
            "ingredient_b": b,
            "cosine_similarity": similarity,
            "candidate_source": information["candidate_source"],
            "label": ""
        })

    output_df = pd.DataFrame(results)

    output_df = output_df.sort_values(
        by=["candidate_source", "cosine_similarity"],
        ascending=[True, False]
    )

    output_df.to_csv(OUTPUT_FILE, index=False)

    print("\nCANDIDATE GENERATION COMPLETE")
    print("-" * 45)

    print(f"New candidate pairs: {len(output_df)}")

    print(
        "Parenthetical candidates:",
        (output_df["candidate_source"] ==
         "Parenthetical name candidate").sum()
    )

    print(
        "Semantic candidates:",
        (output_df["candidate_source"] ==
         "Semantic neighbour candidate").sum()
    )

    print(f"\nSaved to:\n{OUTPUT_FILE}")

    print("\nFirst 20 candidates:")

    print(
        output_df.head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
