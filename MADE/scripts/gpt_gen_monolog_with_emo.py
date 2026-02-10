from openai import OpenAI
import json
import re
import os
import hashlib
from collections import Counter

API_KEY    = None


client = OpenAI(api_key=API_KEY)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
desc_path = os.path.join(BASE_DIR, "cookie_theft_picture.txt")

with open(desc_path, "r", encoding="utf-8") as f:
    MASTER_DESCRIPTION = f.read()

PROMPT_TEMPLATE = f"""
You are an expert narrator.
Below is a detailed master description of a scene.
Your task is to create 7 unique monologues, each one 10–15 sentences long,
from the perspective of a single speaker observing the scene.

Constraints:
- Each monologue must contain between 10 and 15 sentences.
- Do not join multiple sentences in one string.
- Use first-person narration as an observer ("I see...", "I notice...", "I think...").
- Make each monologue sound natural and spontaneous.
- For each sentence, assign one emotion label from this set:
  [happy, sad, angry, fearful, disgusted, surprised, neutral].
- Do not invent new labels.
- Each monologue must contain at least one non-neutral emotion.
- Across all 7 monologues, each of the 7 emotions above must appear at least once.

Return the result only as a valid JSON object with this structure:

{{
  "monologues": [
    {{
      "sentences": [
        {{"text": "sentence1", "emotion": "neutral"}},
        {{"text": "sentence2", "emotion": "surprised"}}
      ]
    }},
    ...
  ]
}}

Do not add any explanation outside the JSON.

Master Description:
{MASTER_DESCRIPTION}
"""

MAX_DUPLICATE_WITHIN_MONOLOGUE = 3    # same monologue has 3+ repetitions, skip
MAX_DUPLICATE_GLOBAL_ALLOWED = 3       # if 3+ sentences match previous monologues, skip

def md5_text(text: str) -> str:
    """Normalize text for duplicate detection"""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def get_next_id(output_dir):
    """
    Find the current largest monologue_x id in the gpt_monologues folder
    and return the next id.
    """
    existing = []
    for fname in os.listdir(output_dir):
        match = re.match(r"monologue_(\d+)\.json", fname)
        if match:
            existing.append(int(match.group(1)))
    if existing:
        return max(existing) + 1
    else:
        return 1

def load_existing_hashes(output_dir: str):
    """Load all existing sentence hashes to avoid global repeats"""
    seen = Counter()
    total_sentences = 0
    if not os.path.exists(output_dir):
        return seen

    for fname in os.listdir(output_dir):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(output_dir, fname), encoding="utf-8") as f:
                data = json.load(f)
            for seg in data.get("segments", []):
                txt = seg.get("text", "").strip()
                if txt:
                    seen[md5_text(txt)] += 1
                    total_sentences += 1
        except Exception as e:
            print(f"Could not read {fname}: {e}")
    print(f"Loaded {len(seen)} unique sentence hashes ({total_sentences} total sentences).")
    return seen

def generate_monologues():
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT_TEMPLATE}],
        temperature=0.9,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    data = json.loads(content)  # JSON array
    return data["monologues"]

if __name__ == "__main__":
    output_dir = "dataset/MADE/gpt_monologues_with_emo"
    os.makedirs(output_dir, exist_ok=True)

    max_retries = 3
    target_total = 2500
    global_hashes = load_existing_hashes(output_dir)
    skip_count_local = 0
    skip_count_global = 0

    while True:
        next_id = get_next_id(output_dir)
        produced_count = next_id - 1

        if produced_count >= target_total:
            break

        monologues = None
        for attempt in range(max_retries):
            try:
                monologues = generate_monologues()
                if monologues:
                    break
            except Exception as e:
                print(f"Error: {e}")
            print(f"generate_monologues returned empty/invalid, retrying ({attempt+1}/{max_retries})")

        if not monologues:
            print("Monologue could not be generated, skipping this batch.")
            continue

        for mono in monologues:
            # tekrar produced_count güncelle
            next_id = get_next_id(output_dir)
            produced_count = next_id - 1
            if produced_count >= target_total:
                break

            monologue_id = f"monologue_{next_id}"
            sentences = mono["sentences"]

            # ======================================
            # LOCAL DUPLICATE CHECK
            # ======================================
            local_counts = Counter()
            local_hashes = []
            local_reject = False

            for s in sentences:
                txt = s.get("text", "").strip()
                if not txt:
                    continue
                h = md5_text(txt)
                local_counts[h] += 1
                if local_counts[h] >= MAX_DUPLICATE_WITHIN_MONOLOGUE:
                    local_reject = True
                    break
                local_hashes.append(h)

            if local_reject:
                skip_count_local += 1
                print(f"Skipped {monologue_id} (local repetition ≥ {MAX_DUPLICATE_WITHIN_MONOLOGUE}).")
                continue

            # ======================================
            # GLOBAL DUPLICATE CHECK
            # ======================================
            duplicate_count = sum(1 for h in local_hashes if h in global_hashes)
            if duplicate_count >= MAX_DUPLICATE_GLOBAL_ALLOWED:
                skip_count_global += 1
                print(f"Skipped {monologue_id}: {duplicate_count} global duplicates (≥ {MAX_DUPLICATE_GLOBAL_ALLOWED}).")
                continue

            # ======================================
            # SAVE MONOLOGUE
            # ======================================
            global_hashes.update(local_hashes)
            data = {
                "id": monologue_id,
                "label": "gpt-4o-mini",
                "segments": [
                    {
                        "text": s["text"].strip(),
                        "predicted_emotion": s["emotion"],
                        "speaker": "PAR",
                    }
                    for s in sentences
                    if s.get("text") and s.get("emotion")
                ],
            }

            filename = os.path.join(output_dir, f"{monologue_id}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Saved {filename} (Total: {produced_count + 1})")

    print(f"\nCompleted! Total produced monologues: {produced_count}")
    print(f"Skipped (local): {skip_count_local}")
    print(f"Skipped (global): {skip_count_global}")
