import os
import json

# ——— CONFIG ———
INPUT_DIR  = "/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/concept_sets/core50_sessions_json_with_img"   # where s1.json … s11.json live
OUTPUT_DIR = "/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/concept_sets/core50_concepts_SACK"    # where you want s1.txt … s11.txt
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ——— PROCESS EACH SESSION JSON ———
for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".json"):
        continue

    session = os.path.splitext(fname)[0]     # "s1", "s2", …
    json_path = os.path.join(INPUT_DIR, fname)

    # load the JSON mapping class -> [concepts]
    with open(json_path, "r") as f:
        data = json.load(f)

    # flatten & dedupe while preserving first‐seen order
    seen = set()
    all_concepts = []
    for feats in data.values():
        for feat in feats:
            if feat not in seen:
                seen.add(feat)
                all_concepts.append(feat)

    # write out session.txt
    txt_path = os.path.join(OUTPUT_DIR, f"{session}.txt")
    with open(txt_path, "w") as out:
        for concept in all_concepts:
            out.write(concept + "\n")

    print(f"Wrote {txt_path} ({len(all_concepts)} concepts)")






# import os
# import json

# INPUT_DIR   = "/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/concept_sets/core50_sessions_json"   # where s1.json … s11.json live
# OUTPUT_PATH = "/umbc/rs/gokhale/users/shivank2/shivanand/mammoth/concept_sets/core50_sessions_json/all_session_CORE50.json"        # path for the merged output

# merged = {}
# for i in range(1, 12):
#     session = f"s{i}"
#     fname = f"{session}.json"
#     path = os.path.join(INPUT_DIR, fname)
#     if not os.path.isfile(path):
#         print(f"⚠️  {fname} not found, skipping")
#         continue

#     with open(path, "r") as f:
#         data = json.load(f)  # { class_name: [concepts] }

#     # optionally sort classes alphabetically per session
#     for cls, concepts in sorted(data.items()):
#         key = f"{session}_{cls.replace(' ', '_')}"
#         merged[key] = concepts

# # write out; insertion order preserved (s1_…, s2_…, …, s11_…)
# with open(OUTPUT_PATH, "w") as out:
#     json.dump(merged, out, indent=2)

# print(f"✔ Merged {len(merged)} keys into {OUTPUT_PATH}")