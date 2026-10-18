from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = ROOT / "main.pdf"
ASSET_DIR = ROOT / "assets" / "figures"
MANIFEST = ASSET_DIR / "asset_manifest.json"


def main() -> None:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for PDF asset extraction. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    pdf_path = DEFAULT_PDF
    if not pdf_path.exists():
        raise SystemExit(
            f"Expected thesis PDF at {pdf_path}. Copy the thesis PDF there before extraction."
        )

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    manifest = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        text = page.get_text().lower()
        keywords = [
            "tube bending",
            "mandrel",
            "pressure die",
            "main-axis",
            "secondary-axis",
            "uncertainty",
            "quantile",
            "gaussian process",
        ]
        if not any(keyword in text for keyword in keywords):
            continue
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        filename = f"thesis_page_{page_index + 1:03d}.png"
        output = ASSET_DIR / filename
        pix.save(output)
        manifest.append(
            {
                "asset": str(output.relative_to(ROOT)),
                "source": str(pdf_path.relative_to(ROOT)),
                "page": page_index + 1,
                "status": "rendered_full_page_for_manual_cropping",
            }
        )

    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Extracted {len(manifest)} candidate thesis asset pages.")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
