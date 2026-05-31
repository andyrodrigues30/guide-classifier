import os
import json
from classifier import Guide, classify_guide, embed, build_text


# parsing helpers
def extract_title(markdown: str) -> str:
    for line in markdown.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled"


# data loading
def load_guides(folder_path: str, start_id: int = 1):
    guides = []
    current_id = start_id

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".md"):
            continue

        file_path = os.path.join(folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        guides.append({
            "id": str(current_id),
            "file_name": filename,
            "title": extract_title(content),
            "content": content,
            "embedding": None
        })

        current_id += 1

    return guides


# embeddings
def build_existing_embeddings(existing_guides: list[dict]) -> None:
    for g in existing_guides:
        g["embedding"] = embed(
            build_text(
                Guide(
                    id=g["id"],
                    title=g["title"],
                    content=g["content"]
                )
            )
        )


# conversion helper
def to_guide(g: dict) -> Guide:
    return Guide(
        id=g["id"],
        title=g["title"],
        content=g["content"],
        embedding=g.get("embedding")
    )


# classification
def batch_classify(new_guides: list[dict], existing_guides: list[dict]):
    existing_models = [to_guide(g) for g in existing_guides]

    results = []

    for new_g in new_guides:
        result = classify_guide(to_guide(new_g), existing_models)
        results.append({
            "guide": {
                "id": new_g["id"],
                "file_name": new_g["file_name"],
                "title": new_g["title"] or "Untitled"
            },
            "classification": {
                "label": result.label,
                "confidence": result.confidence,
                "related_guide_id": result.related_guide_id,
                "reason": result.reason
            }
        })

    return results


# export
def export_results(results: list[dict], output_path: str = "output/classification_results.json"):
    # ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# pipeline
def run_pipeline(base_path: str):
    existing_path = os.path.join(base_path, "existing")
    new_path = os.path.join(base_path, "new")

    existing = load_guides(existing_path, start_id=1)
    new_guides = load_guides(new_path, start_id=1000)

    build_existing_embeddings(existing)

    results = batch_classify(new_guides, existing)

    export_results(results)


# entry point
def main():
    run_pipeline("test_data")


if __name__ == "__main__":
    main()