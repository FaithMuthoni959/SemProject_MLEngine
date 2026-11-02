
import pandas as pd
import ast
import json
from pathlib import Path

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = Path("data/processed/Recipes_cleaned.csv")
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "ingredient_vocabulary.json"

# Load cleaned recipes
df = pd.read_csv(DATASET_PATH)

# Store unique ingredients
ingredient_vocabulary = set()

# Extract ingredients from every recipe
for ingredients in df["ingredients_normalized"].dropna():

    try:
        # Convert the stored list string into a Python list
        parsed_ingredients = ast.literal_eval(ingredients)

        if isinstance(parsed_ingredients, list):

            for ingredient in parsed_ingredients:

                # Normalize whitespace and capitalization
                ingredient = str(ingredient).strip().lower()

                if ingredient:
                    ingredient_vocabulary.add(ingredient)

    except (ValueError, SyntaxError):
        print(f"Could not parse ingredient list: {ingredients}")

# Sort the ingredients alphabetically
ingredient_vocabulary = sorted(ingredient_vocabulary)

# Save vocabulary as JSON
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
    json.dump(ingredient_vocabulary, file, indent=4, ensure_ascii=False)

# Display results
print("Ingredient vocabulary created successfully!")
print(f"Total unique ingredients: {len(ingredient_vocabulary)}")
print(f"Saved a copy at {OUTPUT_PATH}")

print("\nFirst 20 ingredients:")

for ingredient in ingredient_vocabulary[:20]:
    print("-", ingredient)
