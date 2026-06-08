import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import read_json, write_json
from samba_futbot.training_export import (
    export_coco_detection,
    export_yolo_detection,
    manifest_to_coco_detection,
    yolo_lines_for_image,
)


def synthetic_manifest() -> dict:
    return {
        "schema": "samba_futbot.frame_dataset.v1",
        "images": [
            {
                "image_path": "frames/clip/clip_f000000.jpg",
                "width": 100,
                "height": 50,
                "split": "train",
                "detections": [
                    {"class_name": "robots", "box": [10, 5, 40, 25]},
                    {"class_name": "ball", "box": [50, 20, 60, 30]},
                ],
            },
            {
                "image_path": "frames/clip/clip_f000001.jpg",
                "width": 80,
                "height": 80,
                "split": "val",
                "detections": [
                    {"class_name": "robots", "box": [0, 0, 20, 20]},
                ],
            },
            {
                "image_path": "frames/clip/clip_f000002.jpg",
                "width": 40,
                "height": 40,
                "split": "test",
                "detections": [],
            },
        ],
    }


class TrainingExportTest(unittest.TestCase):
    def test_manifest_to_coco_detection_uses_stable_categories_and_xywh_boxes(self):
        coco = manifest_to_coco_detection(synthetic_manifest())

        self.assertEqual(coco["categories"], [{"id": 1, "name": "ball"}, {"id": 2, "name": "robots"}])
        self.assertEqual(coco["images"][0]["split"], "train")
        self.assertEqual(coco["annotations"][0]["category_id"], 2)
        self.assertEqual(coco["annotations"][0]["bbox"], [10.0, 5.0, 30.0, 20.0])
        self.assertEqual(coco["annotations"][0]["area"], 600.0)
        self.assertEqual(coco["annotations"][1]["category_id"], 1)
        self.assertEqual(coco["annotations"][1]["bbox"], [50.0, 20.0, 10.0, 10.0])

    def test_export_coco_detection_writes_split_files_with_consistent_category_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "dataset" / "manifest.json"
            write_json(manifest_path, synthetic_manifest())

            paths = export_coco_detection(manifest_path, root / "coco")
            train = read_json(paths["train"])
            val = read_json(paths["val"])
            test = read_json(paths["test"])

        self.assertEqual(sorted(paths), ["all", "test", "train", "val"])
        self.assertEqual([image["split"] for image in train["images"]], ["train"])
        self.assertEqual([annotation["category_id"] for annotation in val["annotations"]], [2])
        self.assertEqual(test["images"][0]["split"], "test")
        self.assertEqual(test["annotations"], [])
        self.assertEqual(val["categories"], [{"id": 1, "name": "ball"}, {"id": 2, "name": "robots"}])

    def test_yolo_lines_for_image_normalizes_xywh_to_zero_one(self):
        image = synthetic_manifest()["images"][0]
        lines = yolo_lines_for_image(image, {"ball": 0, "robots": 1})

        self.assertEqual(lines[0], "1 0.250000 0.300000 0.300000 0.400000")
        self.assertEqual(lines[1], "0 0.550000 0.500000 0.100000 0.200000")
        for line in lines:
            values = [float(part) for part in line.split()[1:]]
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_export_yolo_detection_writes_labels_lists_and_data_yaml_by_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "dataset"
            manifest_path = dataset_dir / "manifest.json"
            write_json(manifest_path, synthetic_manifest())

            exported = export_yolo_detection(manifest_path, root / "yolo")
            classes = exported["classes"].read_text(encoding="utf-8").splitlines()
            train_list = exported["image_lists"]["train"].read_text(encoding="utf-8").splitlines()
            data_yaml = exported["data_yaml"].read_text(encoding="utf-8")
            label_manifest = read_json(exported["manifest"])
            train_label = next(path for path in exported["labels"] if "\\train\\" in str(path) or "/train/" in str(path))
            train_label_lines = train_label.read_text(encoding="utf-8").splitlines()

        self.assertEqual(classes, ["ball", "robots"])
        self.assertEqual(train_list, [str((dataset_dir / "frames/clip/clip_f000000.jpg").resolve())])
        self.assertIn("train: images/train.txt", data_yaml)
        self.assertIn("val: images/val.txt", data_yaml)
        self.assertIn("test: images/test.txt", data_yaml)
        self.assertIn("names: ['ball', 'robots']", data_yaml)
        self.assertEqual(
            train_label_lines,
            [
                "1 0.250000 0.300000 0.300000 0.400000",
                "0 0.550000 0.500000 0.100000 0.200000",
            ],
        )
        self.assertEqual(label_manifest["labels"][0]["split"], "train")


if __name__ == "__main__":
    unittest.main()
