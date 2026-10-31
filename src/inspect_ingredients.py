import ast
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RECIPE_FILE = Path("data/processed/Recipes_cleaned.csv")

df = pd.read_csv(RECIPE_FILE)

keywords = ["beef", "carrot", "cow pea", "pork", "milk"]

for _, row in df.iterrows():
    title = str(row.get("title_clean", row.get("title", "Unknown recipe")))
    raw_ingredients = row.get("ingredients_normalized", "")

    if pd.isna(raw_ingredients):
        continue

    try:
        ingredients = ast.literal_eval(str(raw_ingredients))
        if not isinstance(ingredients, list):
            ingredients = str(raw_ingredients).split("|")
    except (ValueError, SyntaxError):
        ingredients = str(raw_ingredients).split("|")

    ingredients = [str(item).strip() for item in ingredients]

    matched_keywords = [
        keyword
        for keyword in keywords
        if any(keyword in ingredient.lower() for ingredient in ingredients)
    ]

    if matched_keywords:
        print(f"\nRecipe: {title}")
        print(f"Keywords found: {', '.join(matched_keywords)}")
        print(f"Ingredients: {ingredients}")