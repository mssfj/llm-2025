from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Dict

from huggingface_hub import upload_folder
from tqdm.auto import tqdm

FOLDER_PATH = Path("dataset/openmathinstruct-2_structured")
REPO_ID = "mssfj/openmathinstruct-2_structured-1000"
REPO_TYPE = "dataset"


def _iter_files(folder: Path) -> list[Path]:
    """Return all files inside the folder (recursive)."""
    return [path for path in folder.rglob("*") if path.is_file()]


def _calc_total_bytes(files: list[Path]) -> int:
    return sum(path.stat().st_size for path in files)


def _build_progress_callback(total_bytes: int) -> tuple[tqdm, Callable[[Any], None]]:
    """
    Build a robust progress callback that works with the UploadProgress object.
    It uses attribute lookups instead of typing to stay compatible with older versions.
    """
    progress_bar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="Uploading to Hugging Face",
    )

    def _on_progress(progress: Any) -> None:
        uploaded = None
        for attr in ("uploaded_bytes", "uploaded", "current"):
            value = getattr(progress, attr, None)
            if value is not None:
                uploaded = value
                break

        total = None
        for attr in ("total_bytes", "total", "total_size"):
            value = getattr(progress, attr, None)
            if value is not None:
                total = value
                break

        if total is not None and total > 0:
            progress_bar.total = total

        if uploaded is not None:
            progress_bar.update(uploaded - progress_bar.n)

    return progress_bar, _on_progress


def main() -> None:
    folder = FOLDER_PATH
    files = _iter_files(folder)
    total_bytes = _calc_total_bytes(files)

    kwargs: Dict[str, Any] = {
        "folder_path": str(folder),
        "repo_id": REPO_ID,
        "repo_type": REPO_TYPE,
    }

    if "progress_callback" in inspect.signature(upload_folder).parameters:
        progress_bar, progress_callback = _build_progress_callback(total_bytes)
        try:
            upload_folder(**kwargs, progress_callback=progress_callback)
        finally:
            progress_bar.close()
    else:
        print("This huggingface_hub version does not support progress callbacks.")
        upload_folder(**kwargs)


if __name__ == "__main__":
    main()
