
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

EMBEDDINGS_PATH = (
    BASE_DIR / "data" / "processed" / "ingredient_embeddings.npz"
)

# Load ingredients and embeddings
data = np.load(EMBEDDINGS_PATH)

ingredients = data["ingredients"].tolist()
embeddings = data["embeddings"]

# Create ingredient-to-index mapping
ingredient_index = {
    ingredient: index
    for index, ingredient in enumerate(ingredients)
}

# Load model for ingredients not in the vocabulary
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def get_embedding(ingredient):
    ingredient = ingredient.strip().lower()

    if ingredient in ingredient_index:
        return embeddings[ingredient_index[ingredient]]

    return model.encode(
        ingredient,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


def compare_ingredients(ingredient_a, ingredient_b):
    embedding_a = get_embedding(ingredient_a)
    embedding_b = get_embedding(ingredient_b)

    similarity = np.dot(embedding_a, embedding_b)

    print(
        f"{ingredient_a} <-> {ingredient_b}: "
        f"{similarity:.4f}"
    )



test_pairs = [
    ("carrot", "carrots (gajjar)"),
    ("cow peas", "black eyed peas (lobia)"),
    ("aubergine", "eggplant"),
    ("minced beef", "ground beef"),
    ("milk", "coconut milk"),
    ("cow milk", "milk"),
    ("potato", "sweet potato"),
    ("green peas", "black eyed peas")
]

print("\nSimilarity results\n")

for ingredient_a, ingredient_b in test_pairs:
    compare_ingredients(ingredient_a, ingredient_b)
