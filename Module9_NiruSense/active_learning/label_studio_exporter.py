import json
import os
from datetime import datetime


class LabelStudioExporter:
    def __init__(self, output_dir="data/label_studio"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self, samples, filename=None):
        """
        Exports samples to Label Studio JSON format.
        """
        if not filename:
            filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.output_dir, filename)

        tasks = []
        for s in samples:
            task = {
                "data": {
                    "text": s.get("text"),
                    "meta_info": {
                        "original_score": s.get("score"),
                        "qdrant_id": str(s.get("id")),
                        "created_at": s.get("created_at"),
                    },
                }
            }
            tasks.append(task)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2, ensure_ascii=False)

        print(f"Exported {len(tasks)} tasks to {filepath}")
        return filepath


if __name__ == "__main__":
    # Mock data
    samples = [
        {"text": "Sample text", "score": 0.1, "id": 123, "created_at": "2024-01-01"}
    ]
    exporter = LabelStudioExporter()
    exporter.export(samples)
