
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

VOCAB_PATH = Path("data/processed/ingredient_vocabulary.json")
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ingredient_embeddings.npz"

# Load ingredient vocabulary
with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    ingredients = json.load(file)

print(f"Loaded {len(ingredients)} unique ingredients.")

#Load the Sentence Transformer model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings

embeddings = model.encode(
    ingredients,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Save ingredients and embeddings together
np.savez_compressed(
    OUTPUT_PATH,
    ingredients=np.array(ingredients),
    embeddings=embeddings
)

print(f"Number of ingredients: {len(ingredients)}")
print(f"Embedding dimensions: {embeddings.shape}")
print(f"Saved to: {OUTPUT_PATH}")
