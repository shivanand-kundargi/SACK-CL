import json
from conceptset_utils import remove_too_long, filter_too_similar_to_cls, filter_too_similar

# --- Configuration ---
INPUT_JSON  = "/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/concept_sets/gpt3_init_dict/gpt3_imagenet_r_important_new.json"
OUTPUT_JSON = "/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/concept_sets/gpt3_init_dict/Gemini_filtered_concepts_imagenet-r.json"

# Maximum character length for any concept
MAX_CONCEPT_LEN = 20

# Similarity thresholds (experiment to find what works best for you)
SIM_CUTOFF_CLS     = 0.75   # between concept and its class name
SIM_CUTOFF_GLOBAL  = 0.80   # between any two concepts
DEVICE             = "cuda" # or "cpu" if you don’t have a GPU

# --- Load original data ---
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

filtered = {}


for cls, concepts in data.items():
    # 1) prune by length—but allow up to MAX_CONCEPT_LEN chars
    c1 = remove_too_long(concepts, max_len=MAX_CONCEPT_LEN, print_prob=0)

    # 2) if c1 is empty, skip the rest
    if not c1:
        filtered[cls] = []
        continue

    # 3) filter out anything too similar to the class name
    c2 = filter_too_similar_to_cls(
        concepts=c1,
        classes=[cls],
        sim_cutoff=SIM_CUTOFF_CLS,
        device=DEVICE,
        print_prob=0
    )

    if not c2:
        filtered[cls] = []
        continue

    # 4) filter out near-duplicates among the remaining concepts
    c3 = filter_too_similar(
        concepts=c2,
        sim_cutoff=SIM_CUTOFF_GLOBAL,
        device=DEVICE,
        print_prob=0
    )

    filtered[cls] = c3

# 5) write your new JSON
with open(OUTPUT_JSON, "w") as f:
    json.dump(filtered, f, indent=2)

print("Wrote filtered concepts to", OUTPUT_JSON)