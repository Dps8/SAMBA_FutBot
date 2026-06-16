import tempfile
import unittest
from pathlib import Path

from samba_futbot.io_utils import read_json


class IoUtilsTest(unittest.TestCase):
    def test_read_json_accepts_utf8_bom_files_from_windows_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bom.json"
            path.write_text('{"ok": true}', encoding="utf-8-sig")

            data = read_json(path)

        self.assertEqual(data, {"ok": True})


if __name__ == "__main__":
    unittest.main()
