import json

def extract_bug_fix_from_patch(patch: str):
    buggy_lines, fixed_lines = [], []
    for line in patch.splitlines():
        if line.startswith('-') and not line.startswith('---'):
            buggy_lines.append(line[1:])
        elif line.startswith('+') and not line.startswith('+++'):
            fixed_lines.append(line[1:])
    return '\n'.join(buggy_lines).strip(), '\n'.join(fixed_lines).strip()

with open("github_issues_with_fixes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

final_dataset = []

for item in data:
    buggy_code, fixed_code = extract_bug_fix_from_patch(item["patch"])
    
    if not buggy_code.strip() or not fixed_code.strip():
        continue

    prompt = f"""### Repository:
                {item['repo']}

                ### Issue:
                {item['issue_title']}

                ### Buggy Code Patch:
                {item['patch']}

                ### Fix the Code:
            """

    completion = fixed_code
    final_dataset.append({
        "prompt": prompt.strip(),
        "completion": completion.strip()
    })

# Save to JSONL format
with open("processed_dataset.jsonl", "w", encoding="utf-8") as f:
    for item in final_dataset:
        f.write(json.dumps(item) + "\n")

print(f"✅ Generated {len(final_dataset)} prompt-completion pairs.")
