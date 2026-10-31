
from pathlib import Path
import json
import re
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = Path("C:/Users/User/Documents/3RD YEAR/3.2/Artificial Intelligence/SemProject_MLEngine/data/raw/combined_recipes.csv")
OUTPUT_PATH = Path("C:/Users/User/Documents/3RD YEAR/3.2/Artificial Intelligence/SemProject_MLEngine/data/processed/Recipes_cleaned.csv")

INGREDIENT_ALIASES = {
    "cow milk": "milk",
    "cardamoms powder": "cardamom",
    "cardamoms": "cardamom",
    "spring onions": "spring onion",
    "green bananas": "green banana",
    "potatoes": "potato",
    "onions": "onion",
    "eggs": "egg",
}

SPECIAL_INGREDIENTS = {
    "wheat flour, refined": ["wheat flour"],
    "water, vanilla essence, clear": ["water", "vanilla essence"],
    "yeast, cooking oil, sugar": ["yeast", "cooking oil", "sugar"],
    "coriander, fresh": ["coriander"],
    "sugar, white": ["sugar"],
    "food colour powder, egg yellow": ["food colour powder"],
    "njahi, dolichos lablab": ["njahi"],
    "black beans, dolichos": ["njahi"],
    "rind from lemons sugar": ["lemon rind", "sugar"],
}


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    required_columns = {"title", "ingredients"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if df.empty:
        raise ValueError("The dataset contains no recipes.")

    return df


def normalize_ingredient(ingredient):
    ingredient = str(ingredient).lower().strip()
    ingredient = re.sub(r"\s+", " ", ingredient)
    ingredient = ingredient.strip(" ,;.")

    return INGREDIENT_ALIASES.get(
        ingredient,
        ingredient
    )


def parse_ingredients(raw_ingredients):
    if pd.isna(raw_ingredients):
        return []

    raw = str(raw_ingredients).strip()

    if not raw:
        return []

    ingredients = []

    if "|" in raw:
        parts = raw.split("|")

    else:
        parts = []

        for line in raw.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            line = line.strip(" ,;.")

            if not line:
                continue

            if line in SPECIAL_INGREDIENTS:
                parts.extend(SPECIAL_INGREDIENTS[line])
            else:
                parts.append(line)

    for part in parts:
        ingredient = normalize_ingredient(part)

        if ingredient:
            ingredients.append(ingredient)

    unique_ingredients = []
    seen = set()

    for ingredient in ingredients:
        if ingredient not in seen:
            unique_ingredients.append(ingredient)
            seen.add(ingredient)

    return unique_ingredients


def normalize_title(title):
    title = str(title).lower().strip()
    title = re.sub(r"\s+", " ", title)
    return title


def preprocess_dataset(df):
    df = df.copy()

    df["title_clean"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df["ingredients_list"] = df["ingredients"].apply(
        parse_ingredients
    )

    df["ingredients_normalized"] = df[
        "ingredients_list"
    ].apply(
        lambda items: json.dumps(
            items,
            ensure_ascii=False
        )
    )

    df["title_key"] = df["title_clean"].apply(
        normalize_title
    )

    df["ingredient_set_key"] = df[
        "ingredients_list"
    ].apply(
        lambda items: json.dumps(
            sorted(set(items)),
            ensure_ascii=False
        )
    )

    return df


def remove_empty_recipes(df):
    empty_mask = df["ingredients_list"].apply(
        lambda items: len(items) == 0
    )

    removed_count = int(empty_mask.sum())

    cleaned_df = df.loc[~empty_mask].copy()

    print(
        f"Recipes removed because they have no ingredients: "
        f"{removed_count}"
    )

    return cleaned_df


def remove_exact_duplicate_recipes(df):
    duplicate_mask = df.duplicated(
        subset=["title_key", "ingredient_set_key"],
        keep="first"
    )

    removed_count = int(duplicate_mask.sum())

    cleaned_df = df.loc[~duplicate_mask].copy()

    print(
        f"Exact duplicate recipes removed: {removed_count}"
    )

    return cleaned_df


def report_results(df, original_count):
    df["title_key"] = (
    df["title_clean"]
    .fillna("")
    .astype(str)
    .str.lower()
    .str.strip()
)
    print("\n Cleaning Results ")

    print(f"Original recipe count: {original_count}")
    print(f"Remaining recipe count: {len(df)}")

    print(
        "Recipes with no parsed ingredients:",
        int(df["ingredients_list"].apply(len).eq(0).sum())
    )

    total_ingredient_occurrences = sum(
        len(items) for items in df["ingredients_list"]
    )

    unique_ingredient_occurrences = sum(
        len(set(items)) for items in df["ingredients_list"]
    )

    print(
        "Ingredient occurrences after deduplication:",
        total_ingredient_occurrences
    )

    print(
        "Unique ingredient occurrences within recipes:",
        unique_ingredient_occurrences
    )

    duplicate_titles = df[
        df["title_key"].duplicated(keep=False)
    ].sort_values("title_key")

    print(
        "Rows with titles still duplicated:",
        len(duplicate_titles)
    )

    if not duplicate_titles.empty:
        print("\nDuplicate titles requiring review:")

        for _, row in duplicate_titles.iterrows():
            print(f"\nRecipe: {row['title_clean']}")
            print(f"Ingredients: {row['ingredients_list']}")

    all_ingredients = sorted({
        ingredient
        for items in df["ingredients_list"]
        for ingredient in items
    })

    print(
        "\nNumber of distinct normalized ingredient names:",
        len(all_ingredients)
    )

    print("\nExamples of cleaned recipes")

    for _, row in df.head(5).iterrows():
        print(f"\nRecipe: {row['title_clean']}")
        print(f"Ingredients: {row['ingredients_list']}")

    print("\n First 30 ingredients")

    for ingredient in all_ingredients[:30]:
        print(f"- {ingredient}")


def main():
    df = load_dataset()
    original_count = len(df)

    df = preprocess_dataset(df)

    df = remove_empty_recipes(df)

    df = remove_exact_duplicate_recipes(df)

    df = df.drop(
        columns=["title_key", "ingredient_set_key"]
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    report_results(df, original_count)

    print(f"\nCleaned dataset saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()