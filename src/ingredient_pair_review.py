
import numpy as np
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = Path("data/processed/ingredient_embeddings.npz")

EXISTING_LABELS_PATH = Path("data/processed/ingredient_matching_tests.csv")

OUTPUT_PATH = Path("data/processed/ingredient_pair_review.csv")

# Number of new candidate pairs to generate
TARGET_PAIRS = 150

# Number of nearest neighbours to consider per ingredient
TOP_K = 8


#Load the ingredient embeddings

data = np.load(EMBEDDINGS_PATH)

ingredients = data["ingredients"].tolist()
embeddings = data["embeddings"]

print(f"Loaded {len(ingredients)} ingredients.")



# 3. Load existing labelled pairs
existing_df = pd.read_csv(EXISTING_LABELS_PATH)

existing_pairs = set()
review_records = []

for _, row in existing_df.iterrows():

    ingredient_a = str(row["ingredient_a"]).strip().lower()
    ingredient_b = str(row["ingredient_b"]).strip().lower()

    pair_key = tuple(sorted((ingredient_a, ingredient_b)))

    existing_pairs.add(pair_key)

    review_records.append({
        "ingredient_a": ingredient_a,
        "ingredient_b": ingredient_b,
        "cosine_similarity": np.nan,
        "label": int(row["label"])
    })



# 4. Find semantically similar ingredient pairs
print("Finding similar ingredient candidates...")


# Matrix multiplication calculates cosine similarities.
similarity_matrix = embeddings @ embeddings.T

candidate_pairs = {}

for index, ingredient in enumerate(ingredients):

    # Get indices of the most similar ingredients
    nearest_indices = np.argsort(
        similarity_matrix[index]
    )[-(TOP_K + 1):][::-1]

    for neighbour_index in nearest_indices:

        if index == neighbour_index:
            continue

        neighbour = ingredients[neighbour_index]

        pair_key = tuple(sorted((ingredient, neighbour)))

        if pair_key in existing_pairs:
            continue

        similarity = float(
            similarity_matrix[index, neighbour_index]
        )

        # Keep only potentially useful candidates
        if similarity < 0.30:
            continue

        candidate_pairs[pair_key] = similarity



candidate_df = pd.DataFrame([
    {
        "ingredient_a": pair[0],
        "ingredient_b": pair[1],
        "cosine_similarity": similarity
    }
    for pair, similarity in candidate_pairs.items()
])

# Sort candidates by similarity before sampling.
candidate_df = candidate_df.sort_values(
    "cosine_similarity",
    ascending=False
).reset_index(drop=True)

if len(candidate_df) < TARGET_PAIRS:
    print(
        f"Warning: Only {len(candidate_df)} new candidates found."
    )
    selected_df = candidate_df
else:
    # Sample from the candidate pool with a fixed random seed
    selected_df = candidate_df.sample(
        n=TARGET_PAIRS,
        random_state=42
    )

# Leave labels blank for manual review
selected_df["label"] = np.nan


existing_output = pd.DataFrame(review_records)

final_df = pd.concat(
    [existing_output, selected_df],
    ignore_index=True
)

final_df = final_df.drop_duplicates(
    subset=["ingredient_a", "ingredient_b"]
)

final_df = final_df[
    [
        "ingredient_a",
        "ingredient_b",
        "cosine_similarity",
        "label"
    ]
]


OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print("\nCandidate generation completed!")

print(f"Existing labelled pairs: {len(existing_output)}")
print(f"New candidate pairs: {len(selected_df)}")
print(f"Total pairs for review: {len(final_df)}")

print(f"\nSaved to: {OUTPUT_PATH}")

print("\nFirst 15 candidate pairs:")
print(final_df.head(15).to_string(index=False))
