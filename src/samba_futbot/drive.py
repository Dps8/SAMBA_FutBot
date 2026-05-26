from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable


EMBEDDED_FOLDER_URL = "https://drive.google.com/embeddedfolderview?id={folder_id}#list"
DOWNLOAD_URL = "https://drive.google.com/uc?export=download&id={file_id}"
USER_AGENT = "Mozilla/5.0 (compatible; SAMBA-FutBot/0.1)"


@dataclass(slots=True)
class DriveItem:
    id: str
    name: str
    path: str
    url: str
    kind: str
    last_modified: str | None = None

    @property
    def is_folder(self) -> bool:
        return self.kind == "folder"

    @property
    def extension(self) -> str:
        return Path(self.name).suffix.lower()

    def to_record(self) -> dict:
        return asdict(self)


def _fetch_text(url: str, opener: urllib.request.OpenerDirector | None = None) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    open_fn = opener.open if opener else urllib.request.urlopen
    with open_fn(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def parse_embedded_folder(html_text: str, parent_path: str) -> list[DriveItem]:
    """Parse Google Drive's public embedded folder page.

    This avoids requiring a Google API key for a public challenge folder. The
    parser is intentionally small and covered by tests because Drive markup can
    change.
    """

    pattern = re.compile(
        r'<div class="flip-entry" id="entry-([^"]+)"[\s\S]*?'
        r'<a href="([^"]+)"[\s\S]*?'
        r'<div class="flip-entry-title">([\s\S]*?)</div>[\s\S]*?'
        r'<div class="flip-entry-last-modified"><div>(.*?)</div>',
        re.IGNORECASE,
    )
    items: list[DriveItem] = []
    for item_id, raw_url, raw_name, raw_date in pattern.findall(html_text):
        name = html.unescape(re.sub("<[^>]+>", "", raw_name)).strip()
        url = html.unescape(raw_url)
        kind = "folder" if "/drive/folders/" in url else "file"
        path = f"{parent_path}/{name}" if parent_path else name
        items.append(
            DriveItem(
                id=item_id,
                name=name,
                path=path,
                url=url,
                kind=kind,
                last_modified=html.unescape(raw_date).strip() or None,
            )
        )
    return items


def index_public_folder(root_id: str, root_name: str = "Meta_Glasses") -> list[DriveItem]:
    """Recursively list a public Google Drive folder."""

    seen: set[str] = set()
    queue: deque[tuple[str, str]] = deque([(root_id, root_name)])
    indexed: list[DriveItem] = []

    while queue:
        folder_id, folder_path = queue.popleft()
        if folder_id in seen:
            continue
        seen.add(folder_id)
        text = _fetch_text(EMBEDDED_FOLDER_URL.format(folder_id=folder_id))
        for item in parse_embedded_folder(text, folder_path):
            indexed.append(item)
            if item.is_folder:
                queue.append((item.id, item.path))
    return indexed


def write_manifest(items: Iterable[DriveItem], out_path: str | Path) -> None:
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = [item.to_record() for item in items]
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> list[DriveItem]:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [DriveItem(**record) for record in records]


def find_manifest_item(
    items: Iterable[DriveItem], *, file_id: str | None = None, name: str | None = None
) -> DriveItem:
    matches: list[DriveItem] = []
    for item in items:
        if item.is_folder:
            continue
        if file_id and item.id == file_id:
            matches.append(item)
        elif name and item.name == name:
            matches.append(item)
        elif name and name.lower() in item.path.lower():
            matches.append(item)
    if not matches:
        target = file_id or name
        raise FileNotFoundError(f"No Drive file matched {target!r}")
    if len(matches) > 1 and not file_id:
        options = "\n".join(f"- {item.path} ({item.id})" for item in matches[:10])
        raise ValueError(f"Multiple files matched {name!r}:\n{options}")
    return matches[0]


def manifest_output_path(item: DriveItem, out_dir: str | Path, strip_root: bool = False) -> Path:
    parts = [part for part in item.path.split("/") if part not in {"", ".", ".."}]
    if strip_root and len(parts) > 1:
        parts = parts[1:]
    safe_parts = [_safe_path_part(part) for part in parts]
    return Path(out_dir).joinpath(*safe_parts)


def download_manifest_files(
    items: Iterable[DriveItem],
    out_dir: str | Path,
    *,
    extensions: set[str] | None = None,
    strip_root: bool = False,
    limit: int | None = None,
    force: bool = False,
    progress: Callable[[dict[str, str | int]], None] | None = None,
) -> list[dict[str, str | int]]:
    results: list[dict[str, str | int]] = []
    files = [item for item in items if not item.is_folder]
    if extensions:
        normalized_extensions = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
        files = [item for item in files if item.extension in normalized_extensions]
    if limit is not None:
        files = files[:limit]

    for index, item in enumerate(files, start=1):
        output = manifest_output_path(item, out_dir, strip_root=strip_root)
        if output.exists() and output.stat().st_size > 0 and not force:
            result = {
                "index": index,
                "total": len(files),
                "status": "exists",
                "path": str(output),
                "bytes": output.stat().st_size,
            }
            results.append(result)
            if progress:
                progress(result)
            continue
        if output.exists() and output.stat().st_size == 0:
            output.unlink()
        result = {
            "index": index,
            "total": len(files),
            "status": "downloading",
            "path": str(output),
            "bytes": 0,
        }
        if progress:
            progress(result)
        download_drive_file(item.id, output)
        result = {
            "index": index,
            "total": len(files),
            "status": "downloaded",
            "path": str(output),
            "bytes": output.stat().st_size,
        }
        results.append(result)
        if progress:
            progress(result)
    return results


def _parse_confirm_form(html_text: str) -> tuple[str, dict[str, str]] | None:
    form_match = re.search(
        r'<form[^>]+id="download-form"[^>]+action="([^"]+)"[\s\S]*?</form>',
        html_text,
        flags=re.IGNORECASE,
    )
    if not form_match:
        return None
    form_html = form_match.group(0)
    action = html.unescape(form_match.group(1))
    params = {
        html.unescape(name): html.unescape(value)
        for name, value in re.findall(r'name="([^"]+)" value="([^"]*)"', form_html)
    }
    return action, params


def download_drive_file(file_id: str, out_path: str | Path, max_bytes: int | None = None) -> Path:
    """Download a public Drive file, handling the large-file warning form."""

    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = output.with_name(output.name + ".part")
    if temp_output.exists():
        temp_output.unlink()

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    first_url = DOWNLOAD_URL.format(file_id=file_id)
    first_request = urllib.request.Request(first_url, headers={"User-Agent": USER_AGENT})
    with opener.open(first_request, timeout=60) as response:
        content_type = response.headers.get("Content-Type", "")
        first_chunk = response.read(8192)
        if "text/html" not in content_type.lower():
            with temp_output.open("wb") as fh:
                fh.write(first_chunk)
                _copy_response(response, fh, max_bytes=max_bytes, already=len(first_chunk))
            temp_output.replace(output)
            return output
        html_text = first_chunk.decode("utf-8", "replace") + response.read().decode(
            "utf-8", "replace"
        )

    form = _parse_confirm_form(html_text)
    if not form:
        raise RuntimeError("Drive did not return a downloadable file or confirmation form.")
    action, params = form
    confirm_url = action + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(confirm_url, headers={"User-Agent": USER_AGENT})
    with opener.open(request, timeout=60) as response, temp_output.open("wb") as fh:
        _copy_response(response, fh, max_bytes=max_bytes)
    temp_output.replace(output)
    return output


def _copy_response(response, fh, max_bytes: int | None = None, already: int = 0) -> None:
    written = already
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        if max_bytes is not None and written + len(chunk) > max_bytes:
            chunk = chunk[: max_bytes - written]
        fh.write(chunk)
        written += len(chunk)
        if max_bytes is not None and written >= max_bytes:
            break


def _safe_path_part(value: str) -> str:
    forbidden = '<>:"\\|?*'
    cleaned = "".join("_" if ch in forbidden else ch for ch in value).strip()
    return cleaned or "_"
