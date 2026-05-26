import unittest

from samba_futbot.drive import parse_embedded_folder


class DriveParserTest(unittest.TestCase):
    def test_parse_embedded_folder(self):
        html = """
        <div class="flip-entry" id="entry-folder1" tabindex="0" role="link">
          <div class="flip-entry-info">
            <a href="https://drive.google.com/drive/folders/folder1" target="_blank">
              <div class="flip-entry-title">17Abril</div>
            </a>
          </div>
          <div class="flip-entry-last-modified"><div>Apr 20</div></div>
        </div>
        <div class="flip-entry" id="entry-file1" tabindex="0" role="link">
          <div class="flip-entry-info">
            <a href="https://drive.google.com/file/d/file1/view?usp=drive_web" target="_blank">
              <div class="flip-entry-title">video-297_singular_display.mov</div>
            </a>
          </div>
          <div class="flip-entry-last-modified"><div>Apr 20</div></div>
        </div>
        """
        items = parse_embedded_folder(html, "Meta_Glasses")
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0].is_folder)
        self.assertFalse(items[1].is_folder)
        self.assertEqual(items[1].path, "Meta_Glasses/video-297_singular_display.mov")


if __name__ == "__main__":
    unittest.main()
