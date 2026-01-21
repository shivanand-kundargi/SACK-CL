import openai
import glob
import os
import json
from tqdm import tqdm
import time
# openai.api_key = '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'

from openai import OpenAI
client = OpenAI(api_key="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")



# print(response.output_text)


# def get_descriptors_for_category(category_name):
#     prompt = (
#         f"Q: List out short (1-2 words) visual finegrained features that can describe a {category_name}. "
#         f"Consider its appearance, structure, and surrounding environmental elements."
#         f"\nA: The visual features are \n-"
#     )

#     response = client.chat.completions.create(
#     model="gpt-5",
#     prompt=prompt
# )

#     features = [
#         line.strip('- ').strip()
#         for line in response.choices[0].text.split('\n')
#         if line.strip().startswith('-')
#     ]
#     return features
def get_descriptors_for_category(category_name: str):
    """Ask the model for short visual features describing a given category."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that lists short, fine-grained visual features."},
        {"role": "user", "content": (
            f"List short (1-2 words) visual fine-grained features that describe a {category_name}. "
            f"Focus and pay attention to detail on its appearance, structure, texture, and environment. "
            f"Give 10-20 concepts per {category_name} such that the concepts strongly define {category_name}"
        )}
    ]

    response = client.chat.completions.create(
        model="gpt-5",     # or "gpt-4o"
        messages=messages,
        # temperature=0.2,
        # max_tokens=150
    )

    text = response.choices[0].message.content
    features = [
        line.strip("- ").strip()
        for line in text.split("\n")
        if line.strip().startswith("-")
    ]
    return features


def save_descriptors_to_file(data, filename="/p/lustre1/kundargi1/SACK/concept_sets/descriptors_inaturalist1000.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def read_categories_from_file(filepath):
    """Reads class names from a text file (one per line)."""
    with open(filepath, 'r') as f:
        categories = [line.strip() for line in f if line.strip()]
    return categories

if __name__ == "__main__":
    # Path to the text file containing class names (one per line)
    categories_file = "/p/lustre1/kundargi1/SACK/data/inaturalist1000_classes.txt"

    # Read categories from file
    categories = read_categories_from_file(categories_file)
    categories = sorted(categories, key=lambda x: x.lower())    

    num_iterations = 1  # Number of unique descriptor sets per category
    all_descriptors = {}

    for category in tqdm(categories, desc="Processing categories"):
        category_descriptors = set()

        for _ in range(num_iterations):
            features = get_descriptors_for_category(category)
            category_descriptors.update(features)

        all_descriptors[category] = list(category_descriptors)
        save_descriptors_to_file(all_descriptors)
    time.sleep(10)
    print("Descriptors saved to file!")