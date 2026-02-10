from google import genai
import json
import re
import os
import hashlib
from collections import Counter
from pydantic import BaseModel
from enum import Enum

# =====================================================
# Config
# =====================================================
API_KEY = None
client = genai.Client(api_key=API_KEY)

OUTPUT_DIR = "dataset/MADE/genai_monologues_with_emo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TOTAL = 2500
MAX_RETRIES = 3
MAX_DUPLICATE_WITHIN_MONOLOGUE = 3 # same monologue has 3+ repetitions, skip
MAX_DUPLICATE_GLOBAL_ALLOWED = 3   # if 3+ sentences match previous monologues, skip

# =====================================================
# Emotion Enum & Models
# =====================================================
class Emotion(str, Enum):
    happy = "happy"
    sad = "sad"
    angry = "angry"
    fearful = "fearful"
    disgusted = "disgusted"
    surprised = "surprised"
    neutral = "neutral"

class Sentence(BaseModel):
    text: str
    emotion: Emotion

class Monologue(BaseModel):
    sentences: list[Sentence]

# =====================================================
# Utilities
# =====================================================
def md5_text(text: str) -> str:
    """Normalize text for duplicate detection"""
    return hashlib.md5(text.strip().lower().encode("utf-8")).hexdigest()

def get_next_id(output_dir: str) -> int:
    existing = []
    for fname in os.listdir(output_dir):
        match = re.match(r"monologue_(\d+)\.json", fname)
        if match:
            existing.append(int(match.group(1)))
    return max(existing) + 1 if existing else 1

def load_existing_hashes(output_dir: str) -> Counter:
    """Load all existing sentence hashes"""
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

# =====================================================
# Prompt Template
# =====================================================
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

Return the result only as valid JSON:
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

Master Description:
{MASTER_DESCRIPTION}
"""

# =====================================================
# Model Request
# =====================================================
def generate_monologues() -> list[Monologue]:
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=PROMPT_TEMPLATE,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[Monologue],
            },
        )
        return response.parsed
    except Exception as e:
        print("Parse error:", e)
        return []

# =====================================================
# Main Generation Loop (Local + Global Duplicate Filter)
# =====================================================
if __name__ == "__main__":
    global_hashes = load_existing_hashes(OUTPUT_DIR)
    skip_count_local, skip_count_global = 0, 0

    print("Starting Gemini generation with local + global duplicate filtering...\n")

    while True:
        next_id = get_next_id(OUTPUT_DIR)
        produced_count = next_id - 1
        if produced_count >= TARGET_TOTAL:
            break

        monologues = None
        for attempt in range(MAX_RETRIES):
            monologues = generate_monologues()
            if monologues:
                break
            print(f"Empty response, retrying ({attempt+1}/{MAX_RETRIES})")

        if not monologues:
            print("Generation failed, skipping batch.")
            continue

        for mono in monologues:
            next_id = get_next_id(OUTPUT_DIR)
            produced_count = next_id - 1
            if produced_count >= TARGET_TOTAL:
                break

            sentences = mono.sentences
            if not sentences:
                continue

            # ================================
            # LOCAL DUPLICATE CHECK
            # ================================
            local_counts = Counter()
            local_hashes = []
            local_reject = False

            for s in sentences:
                txt = s.text.strip()
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
                print(f"Skipped monologue_{next_id} (local repetition ≥ {MAX_DUPLICATE_WITHIN_MONOLOGUE}).")
                continue

            # ================================
            # GLOBAL DUPLICATE CHECK
            # ================================
            duplicate_count = sum(1 for h in local_hashes if h in global_hashes)
            if duplicate_count >= MAX_DUPLICATE_GLOBAL_ALLOWED:
                skip_count_global += 1
                print(f"Skipped monologue_{next_id}: {duplicate_count} global duplicates (≥ {MAX_DUPLICATE_GLOBAL_ALLOWED}).")
                continue

            # ================================
            # SAVE MONOLOGUE
            # ================================
            global_hashes.update(local_hashes)

            monologue_id = f"monologue_{next_id}"
            data = {
                "id": monologue_id,
                "label": "gemini-2.5-flash",
                "segments": [
                    {"text": s.text.strip(), "predicted_emotion": s.emotion, "speaker": "PAR"}
                    for s in sentences
                    if s.text and s.emotion
                ],
            }

            filename = os.path.join(OUTPUT_DIR, f"{monologue_id}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            print(f"Saved {filename} (Total: {produced_count + 1})")

    print(f"\nCompleted! Total produced monologues: {produced_count}")
    print(f"Skipped (local): {skip_count_local}")
    print(f"Skipped (global): {skip_count_global}")