# Third-party notices

SAMBA FutBot is licensed under Apache-2.0. Dependencies retain their own
licenses and copyrights.

| Project | Role in SAMBA FutBot | License / terms |
|---|---|---|
| Meta SAM 3 | Promptable image/video segmentation and official fine-tuning stack | Upstream SAM 3 repository license and model checkpoint terms apply. |
| PyTorch / TorchVision | CUDA inference and training runtime | BSD-style license. |
| Hugging Face Hub / Transformers / Accelerate | Checkpoint access and model utilities | Apache-2.0. Model terms remain separate. |
| OpenCV | Video I/O, geometry, masks and rendering | Apache-2.0. |
| NumPy | Numerical arrays and geometry | BSD-3-Clause. |
| pandas | Tabular exports and analysis | BSD-3-Clause. |
| Pillow | Submission cards and image utilities | HPND. |
| PyYAML | Configuration files | MIT. |
| tqdm | Progress reporting | MPL-2.0 and MIT. |
| Supervision | Detection containers, annotation helpers and ByteTrack integration | MIT. |
| scikit-learn | Optional clustering and evaluation helpers | BSD-3-Clause. |
| Matplotlib | Optional tactical plots | PSF-based license. |
| FFmpeg / libx264 | Optional H.264 packaging for submission videos | Their upstream build and codec licenses apply; no binary is redistributed in this repository. |
| Remotion | Reproducible professional video timeline and rendering | Remotion License; free-license terms apply to eligible individuals and organizations. |
| React / React DOM | Component runtime used by the Remotion timeline | MIT. |
| esbuild | Local bundling for the Remotion project | MIT. |

The official SAM 3 repository is installed separately from its upstream URL;
weights and access tokens are not committed. Challenge videos are also excluded
from Git. See `requirements-sam3.txt`, `pyproject.toml` and the upstream
projects for exact versions and complete license texts. Remotion and frontend
tool versions are pinned in `video_studio/package.json` and `pnpm-lock.yaml`.
