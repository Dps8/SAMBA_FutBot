import json
import tempfile
import unittest
from pathlib import Path

from samba_futbot.finetune_preflight import (
    analyze_finetune_preflight,
    run_finetune_preflight,
    write_finetune_preflight,
)


def sam3_dataset(file_name: str, *, phrase: str = "orange ball") -> dict:
    return {
        "images": [
            {
                "id": 1,
                "file_name": file_name,
                "text_input": phrase,
                "is_instance_exhaustive": True,
                "is_pixel_exhaustive": True,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0.1, 0.2, 0.3, 0.4],
                "segmentation": {"size": [2, 2], "counts": [1, 2, 1]},
            }
        ],
        "categories": [{"id": 1, "name": "ball"}],
    }


def ready_snapshot() -> dict:
    return {
        "paths": {
            "sam3_root": "/sam3",
            "train_script": "/sam3/sam3/train/train.py",
            "training_readme": "/sam3/README_TRAIN.md",
            "vocab": "/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            "checkpoint": "/models/sam3.pt",
            "train_json": "/data/train.json",
            "val_json": "/data/val.json",
            "train_images": "/data/train",
            "val_images": "/data/val",
        },
        "exists": {
            "sam3_root": True,
            "train_script": True,
            "training_readme": True,
            "vocab": True,
            "checkpoint": True,
            "train_json": True,
            "val_json": True,
            "train_images": True,
            "val_images": True,
        },
        "datasets": {
            "train": sam3_dataset("train_001.jpg"),
            "val": sam3_dataset("val_001.jpg"),
        },
        "image_checks": {
            "train": {"checked": True, "missing": []},
            "val": {"checked": True, "missing": []},
        },
        "cuda": {"checked": False},
        "python_executable": "python",
    }


class FinetunePreflightTest(unittest.TestCase):
    def test_ready_snapshot_reports_counts_classes_phrases_and_command(self):
        report = analyze_finetune_preflight(ready_snapshot())

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["summary"]["train_images"], 1)
        self.assertEqual(report["summary"]["val_annotations"], 1)
        self.assertEqual(report["datasets"]["train"]["classes"], ["ball"])
        self.assertEqual(report["datasets"]["train"]["phrases"], {"orange ball": 1})
        self.assertIn("sam3/train/train.py", report["suggested_command"])
        self.assertNotIn("file_names_normalized", report["datasets"]["train"])

    def test_missing_required_assets_fail(self):
        snapshot = ready_snapshot()
        snapshot["exists"]["checkpoint"] = False
        snapshot["exists"]["vocab"] = False

        report = analyze_finetune_preflight(snapshot)

        self.assertEqual(report["status"], "fail")
        self.assertEqual(
            {issue["code"] for issue in report["issues"]},
            {"missing_checkpoint", "missing_bpe_vocab"},
        )

    def test_invalid_sam3_annotation_fields_fail(self):
        snapshot = ready_snapshot()
        image = snapshot["datasets"]["train"]["images"][0]
        image.pop("text_input")
        image["is_pixel_exhaustive"] = "yes"
        annotation = snapshot["datasets"]["train"]["annotations"][0]
        annotation["bbox"] = [0.9, 0.1, 0.3, 0.2]
        annotation["segmentation"] = {"size": [2, 2], "counts": [3]}

        report = analyze_finetune_preflight(snapshot)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertEqual(report["status"], "fail")
        self.assertIn("invalid_text_input", codes)
        self.assertIn("invalid_is_pixel_exhaustive", codes)
        self.assertIn("invalid_normalized_bbox", codes)
        self.assertIn("invalid_segmentation_rle", codes)

    def test_train_val_file_name_overlap_fails_case_insensitively(self):
        snapshot = ready_snapshot()
        snapshot["datasets"]["train"] = sam3_dataset("Frames/Shared.JPG")
        snapshot["datasets"]["val"] = sam3_dataset("frames\\shared.jpg")

        report = analyze_finetune_preflight(snapshot)

        self.assertEqual(report["status"], "fail")
        overlap = next(issue for issue in report["issues"] if issue["code"] == "train_val_file_overlap")
        self.assertEqual(overlap["examples"], ["frames/shared.jpg"])

    def test_cuda_diagnostic_is_optional_and_unavailable_cuda_needs_review(self):
        snapshot = ready_snapshot()
        self.assertEqual(analyze_finetune_preflight(snapshot)["status"], "ready")

        snapshot["cuda"] = {
            "checked": True,
            "torch_available": True,
            "cuda_available": False,
            "torch_version": "2.7",
        }
        report = analyze_finetune_preflight(snapshot)

        self.assertEqual(report["status"], "review")
        self.assertEqual(report["issues"][0]["code"], "cuda_unavailable")

    def test_wrapper_accepts_alternate_official_train_script_and_checks_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sam3_root = root / "SAM3"
            (sam3_root / "sam3").mkdir(parents=True)
            (sam3_root / "sam3" / "train.py").write_text("# official entry\n", encoding="utf-8")
            (sam3_root / "README_TRAIN.md").write_text("train\n", encoding="utf-8")
            (sam3_root / "assets").mkdir()
            (sam3_root / "assets" / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"vocab")
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            train_images = root / "images" / "train"
            val_images = root / "images" / "val"
            train_images.mkdir(parents=True)
            val_images.mkdir(parents=True)
            (train_images / "train.jpg").write_bytes(b"image")
            (val_images / "val.jpg").write_bytes(b"image")
            train_json = root / "train.json"
            val_json = root / "val.json"
            train_json.write_text(json.dumps(sam3_dataset("train.jpg")), encoding="utf-8")
            val_json.write_text(json.dumps(sam3_dataset("val.jpg")), encoding="utf-8")

            report = run_finetune_preflight(
                sam3_root=sam3_root,
                checkpoint=checkpoint,
                train_json=train_json,
                val_json=val_json,
                train_images=train_images,
                val_images=val_images,
            )

        self.assertEqual(report["status"], "ready")
        self.assertTrue(
            Path(report["paths"]["train_script"]).as_posix().endswith("sam3/train.py")
        )
        self.assertEqual(report["datasets"]["train"]["missing_image_files"], 0)

    def test_wrapper_reports_invalid_json_and_missing_referenced_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sam3_root = root / "SAM3"
            (sam3_root / "sam3" / "train").mkdir(parents=True)
            (sam3_root / "sam3" / "train" / "train.py").write_text("", encoding="utf-8")
            (sam3_root / "README_TRAIN.md").write_text("", encoding="utf-8")
            (sam3_root / "bpe_simple_vocab_16e6.txt.gz").write_bytes(b"vocab")
            checkpoint = root / "sam3.pt"
            checkpoint.write_bytes(b"checkpoint")
            train_images = root / "train_images"
            val_images = root / "val_images"
            train_images.mkdir()
            val_images.mkdir()
            train_json = root / "train.json"
            val_json = root / "val.json"
            train_json.write_text("{broken", encoding="utf-8")
            val_json.write_text(json.dumps(sam3_dataset("missing.jpg")), encoding="utf-8")

            report = run_finetune_preflight(
                sam3_root=sam3_root,
                checkpoint=checkpoint,
                train_json=train_json,
                val_json=val_json,
                train_images=train_images,
                val_images=val_images,
            )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "fail")
        self.assertIn("invalid_train_json", codes)
        self.assertIn("missing_referenced_images", codes)

    def test_write_wrapper_persists_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "reports" / "preflight.json"
            report = write_finetune_preflight(
                output,
                sam3_root=root / "missing-sam3",
                checkpoint=root / "missing.pt",
                train_json=root / "train.json",
                val_json=root / "val.json",
                train_images=root / "train",
                val_images=root / "val",
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["status"], "fail")
        self.assertEqual(persisted["schema"], "samba_futbot.sam3_finetune_preflight.v1")


if __name__ == "__main__":
    unittest.main()
