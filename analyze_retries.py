import re
from collections import Counter
from pathlib import Path

log = Path(r"C:\Users\const\mininio-ai-finetuning\data\output\logs\generation_20260717_193317.log").read_text(encoding="utf-8")

lines = log.splitlines()
print(f"Total lines: {len(lines)}")

retry_lines = [l for l in lines if "retry " in l and "/3:" in l]
error_lines = [l for l in lines if "| ERROR" in l and "FAIL:" in l]

print(f"Retry WARNING lines: {len(retry_lines)}")
print(f"FAIL ERROR lines (exhausted all 3 retries): {len(error_lines)}")
print(f"Total retry events: {len(retry_lines) + len(error_lines)}")
print()

def extract_errors(msg):
    m = re.search(r"Validation errors: (.+?)(?:\s*\(backoff|\s*$)", msg)
    if not m:
        m = re.search(r"FAIL:.*?-- (.+?)$", msg)
    if m:
        return m.group(1).strip()
    return msg

def normalize_error(e):
    e = re.sub(r"Turn \d+: ", "", e)
    e = re.sub(r"number [\d.]+", "number N", e)
    e = re.sub(r"food_id \d+", "food_id N", e)
    e = re.sub(r"\([^)]*\)", "(...)", e)
    e = re.sub(r"IDs: set\(\)", "IDs: set()", e)
    e = re.sub(r"IDs: \{[^}]+\}", "IDs: {...}", e)
    e = re.sub(r"~\d+ tokens", "~N tokens", e)
    e = re.sub(r"line \d+ column \d+", "line N column N", e)
    return e.strip()

all_retry_lines = retry_lines + error_lines
individual_errors = Counter()
error_categories = Counter()

for line in all_retry_lines:
    raw = extract_errors(line)
    parts = [p.strip() for p in raw.split("; ")]
    for part in parts:
        norm = normalize_error(part)
        if "Validation errors: " in norm:
            norm = norm.replace("Validation errors: ", "")
        individual_errors[norm] += 1
        if "does not match any tool result" in norm:
            error_categories["Hallucinated numbers"] += 1
        elif "not in known IDs" in norm:
            error_categories["Invalid food_id (not in DB)"] += 1
        elif "unit 'pcs' used but food has no piece mode" in norm:
            error_categories["Unit mismatch: 'pcs' on non-piece food"] += 1
        elif "unit 'g' used but food has no grams/ml mode" in norm:
            error_categories["Unit mismatch: 'g' on non-grams food"] += 1
        elif "Too short" in norm:
            error_categories["Conversation too short"] += 1
        elif "WinError" in norm or "connection" in norm.lower():
            error_categories["Network error"] += 1
        elif "delimiter" in norm or "Expecting" in norm:
            error_categories["JSON parse error"] += 1
        elif "missing context_block" in norm:
            error_categories["Missing context_block"] += 1
        else:
            error_categories[f"Other: {norm}"] += 1

print("=" * 70)
print("ERROR CATEGORIES (aggregated across all individual validation errors)")
print("=" * 70)
for cat, count in error_categories.most_common():
    print(f"  {count:>5}  {cat}")

print()
print("=" * 70)
print("DISTINCT ERROR PATTERNS (normalized)")
print("=" * 70)
for err, count in individual_errors.most_common():
    print(f"  {count:>5}  {err}")

print()
print("=" * 70)
print("RETRIES BY SCENARIO TYPE")
print("=" * 70)
scenario_counter = Counter()
for line in all_retry_lines:
    m = re.search(r"\] (\w+) --", line)
    if m:
        scenario_counter[m.group(1)] += 1
for sc, count in scenario_counter.most_common():
    print(f"  {count:>5}  {sc}")

print()
print("=" * 70)
print("RETRIES BY LANGUAGE")
print("=" * 70)
lang_counter = Counter()
for line in all_retry_lines:
    m = re.search(r"\[(\w{2})\]", line)
    if m:
        lang_counter[m.group(1)] += 1
for lang, count in lang_counter.most_common():
    print(f"  {count:>5}  {lang}")

print()
print("=" * 70)
print("RETRY DEPTH DISTRIBUTION")
print("=" * 70)
depth_counter = Counter()
for line in retry_lines:
    m = re.search(r"retry (\d+)/3", line)
    if m:
        depth_counter[f"retry {m.group(1)}/3"] += 1
depth_counter["FAIL (all 3 exhausted)"] = len(error_lines)
for depth, count in sorted(depth_counter.items()):
    print(f"  {count:>5}  {depth}")

print()
total_convos = 0
m = re.findall(r"Saved (\d+)/", log)
if m:
    total_convos = sum(int(x) for x in m)
print(f"Total successful conversations saved: {total_convos}")

success_lines = [l for l in lines if "| SUCCESS" in l or ("OK:" in l and "| INFO" in l)]
api_error_lines = [l for l in lines if "429" in l or "500" in l or "503" in l or "ResourceExhausted" in l or "rate limit" in l.lower()]
print(f"API rate-limit / server error lines: {len(api_error_lines)}")
if api_error_lines:
    print("  Samples:")
    for l in api_error_lines[:5]:
        print(f"    {l.strip()[:200]}")
