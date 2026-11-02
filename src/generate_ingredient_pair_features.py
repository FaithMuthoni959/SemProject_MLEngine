
import numpy as np
import pandas as pd

from pathlib import Path
from sentence_transformers import SentenceTransformer




BASE_DIR = Path(__file__).resolve().parent.parent

TEST_DATA_PATH = Path("data/processed/ingredient_matching_tests.csv")

EMBEDDINGS_PATH = Path ("data/processed/ingredient_embeddings.npz")

OUTPUT_PATH = Path("data/processed/ingredient_pair_features.csv")


# 2. Load ingredient pair data
if not TEST_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_DATA_PATH}"
    )

pairs_df = pd.read_csv(TEST_DATA_PATH)

required_columns = ["ingredient_a", "ingredient_b", "label"]

if not all(column in pairs_df.columns for column in required_columns):
    raise ValueError(
        f"Dataset must contain these columns: {required_columns}"
    )

if pairs_df[required_columns].isnull().any().any():
    raise ValueError("The ingredient pair dataset contains missing values.")

if not pairs_df["label"].isin([0, 1]).all():
    raise ValueError("Labels must contain only 0 or 1.")



# 3. Load existing ingredient embeddings
if not EMBEDDINGS_PATH.exists():
    raise FileNotFoundError(
        f"Embedding file not found: {EMBEDDINGS_PATH}"
    )

embedding_data = np.load(EMBEDDINGS_PATH)

ingredients = embedding_data["ingredients"].tolist()
embeddings = embedding_data["embeddings"]

# Map each ingredient to its embedding
ingredient_index = {
    ingredient: index
    for index, ingredient in enumerate(ingredients)
}

print(f"Loaded {len(ingredients)} ingredient embeddings.")



# 4. Load model 

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Cache embeddings generated for unseen ingredients
embedding_cache = {}


def get_embedding(ingredient):
    """
    Retrieve an existing embedding or generate one
    if the ingredient is not in the vocabulary.
    """

    ingredient = str(ingredient).strip().lower()

    if ingredient in ingredient_index:
        return embeddings[ingredient_index[ingredient]]

    if ingredient not in embedding_cache:
        embedding_cache[ingredient] = model.encode(
            ingredient,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    return embedding_cache[ingredient]



# 5. Calculate ingredient pair features
def calculate_features(ingredient_a, ingredient_b):

    ingredient_a = str(ingredient_a).strip().lower()
    ingredient_b = str(ingredient_b).strip().lower()

    embedding_a = get_embedding(ingredient_a)
    embedding_b = get_embedding(ingredient_b)

    # Embeddings are normalized, so their dot product
    # gives cosine similarity.
    cosine_similarity = float(
        np.dot(embedding_a, embedding_b)
    )

    # Calculate embedding distance
    embedding_distance = 1.0 - cosine_similarity

    # Split ingredient names into words
    words_a = set(ingredient_a.split())
    words_b = set(ingredient_b.split())

    # Calculate word overlap using Jaccard similarity
    union = words_a | words_b
    intersection = words_a & words_b

    word_overlap = (
        len(intersection) / len(union)
        if union else 0.0
    )

    # Calculate difference in ingredient name lengths
    length_difference = abs(
        len(ingredient_a) - len(ingredient_b)
    )

    # Calculate relative length difference
    maximum_length = max(
        len(ingredient_a),
        len(ingredient_b),
        1
    )

    normalized_length_difference = (
        length_difference / maximum_length
    )

    return {
        "cosine_similarity": cosine_similarity,
        "embedding_distance": embedding_distance,
        "word_overlap": word_overlap,
        "length_difference": length_difference,
        "normalized_length_difference": normalized_length_difference
    }



# 6. Generate features for every ingredient pair
feature_records = []

for _, row in pairs_df.iterrows():

    ingredient_a = row["ingredient_a"]
    ingredient_b = row["ingredient_b"]

    features = calculate_features(
        ingredient_a,
        ingredient_b
    )

    feature_records.append({
        "ingredient_a": ingredient_a,
        "ingredient_b": ingredient_b,
        **features,
        "label": int(row["label"])
    })



features_df = pd.DataFrame(feature_records)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

features_df.to_csv(OUTPUT_PATH, index=False)

print("\nFeature generation completed")

print(f"Ingredient pairs processed {len(features_df)}")
print(f"Equivalent pairs {(features_df['label'] == 1).sum()}")
print(f"Non-equivalent pairs {(features_df['label'] == 0).sum()}")

print(f"\nSaved features to{OUTPUT_PATH}")

print("\nGenerated features")
print(features_df.to_string(index=False))
