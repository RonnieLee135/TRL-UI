"""Read-only access to catalog/catalog.json (the machine-readable result of the TRL analysis)."""

from __future__ import annotations

import json
from pathlib import Path


class Catalog:
    def __init__(self, path: Path):
        self.path = Path(path)
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.meta = raw["meta"]
        self.stages = raw["stages"]
        self.families = raw["families"]
        self.planes = raw.get("planes", [])
        self.presets = raw["presets"]
        self.dataset_examples = raw.get("dataset_examples", {})
        self.common_training_args = raw["common_training_args"]
        self.script_fields = raw.get("script_fields", [])
        self.model_fields = raw.get("model_fields", [])
        self.algorithms = raw["algorithms"]
        self._by_id = {a["id"]: a for a in self.algorithms}

    # ------------------------------------------------------------------ views
    def summary(self) -> dict:
        """Everything except the per-field parameter lists (fast first paint)."""

        family_order = {f["id"]: i for i, f in enumerate(self.families)}
        algorithms = []
        for algo in sorted(self.algorithms, key=lambda a: (family_order[a["family"]], a["label"])):
            algorithms.append(
                {
                    "id": algo["id"],
                    "label": algo["label"],
                    "full_name": algo["full_name"],
                    "family": algo["family"],
                    "status": algo["status"],
                    "summary": algo["summary"],
                    "launch": algo["launch"],
                    "config_class": algo["config_class"],
                    "trainer_class": algo["trainer_class"],
                    "requires": algo["requires"],
                    "data": algo["data"],
                    "counts": algo["counts"],
                    "key_params": algo.get("key_params", []),
                    "pros": algo.get("pros", []),
                    "cons": algo.get("cons", []),
                    "pitfalls": algo.get("pitfalls", []),
                    "paper": algo.get("paper"),
                    "doc": algo.get("doc"),
                }
            )
        return {
            "meta": self.meta,
            "stages": self.stages,
            "families": self.families,
            "planes": self.planes,
            "presets": self.presets,
            "dataset_examples": self.dataset_examples,
            "common_training_args": self.common_training_args,
            "algorithms": algorithms,
        }

    def algorithm(self, algo_id: str) -> dict | None:
        algo = self._by_id.get(algo_id)
        if algo is None:
            return None
        payload = {k: v for k, v in algo.items() if k != "fields"}
        payload["fields"] = algo["fields"]
        payload["script_fields"] = [f for f in self.script_fields if f["name"] in algo["script_fields"]]
        payload["model_fields"] = self.model_fields
        return payload

    def field(self, algo_id: str, name: str) -> dict | None:
        for field in self.schema_for(algo_id)["fields"]:
            if field["name"] == name:
                return field
        return None

    def schema_for(self, algo_id: str) -> dict:
        """The flat parameter schema the UI renders and the server validates against.

        Order matters: script plane (data + reward) → algorithm plane → model plane →
        runtime plane, i.e. top-down through the wizard's steps.
        """

        algo = self._by_id[algo_id]
        own = {f["name"] for f in algo["fields"]}
        fields: list[dict] = []
        seen: set[str] = set()

        def add(candidates: list[dict]) -> None:
            for field in candidates:
                if field["tier"] == "deprecated" or field["name"] in seen:
                    continue
                seen.add(field["name"])
                fields.append(field)

        add([f for f in self.script_fields if f["name"] in algo["script_fields"]])
        add(algo["fields"])
        add(self.model_fields)
        add([c for c in self.common_training_args if c["name"] not in own])
        return {
            "algorithm": algo_id,
            "title": algo["label"],
            "stages": self.stages,
            "planes": self.planes,
            "requires": algo["requires"],
            "data": algo["data"],
            "launch": algo["launch"],
            "preset": self.presets.get(algo_id, {}),
            "fields": fields,
        }
