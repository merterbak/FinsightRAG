from __future__ import annotations
import base64
import os
from io import BytesIO
from typing import Dict, List, Optional, Tuple
from mistralai import Mistral


VALID_DOCUMENT_EXTENSIONS = {".pdf"}
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OCR_MODEL_NAME = "mistral-ocr-2505"


def read_mistral_key(config: Optional[Dict] = None):
    if config:
        value = config.get("mistral_api_key")
        if isinstance(value, str) and value.strip():
            return value.strip()

    env_value = os.getenv("MISTRAL_API_KEY", "").strip()
    if env_value:
        return env_value

    raise RuntimeError("Missing MISTRAL_API_KEY. Add it to your .env file or set it in UI")


def ensure_pdf_name(filename: str):
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        return filename
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    return f"{stem}.pdf"


def upload_pdf_document(client: Mistral, filename: str, content: bytes):
    uploaded = client.files.upload(file={"file_name": filename, "content": content}, purpose="ocr")
    signed = client.files.get_signed_url(file_id=uploaded.id)
    return uploaded.id, signed.url


def document_source_from_bytes(client: Mistral, file_bytes: bytes, filename: str):
    extension = os.path.splitext(filename)[1].lower()

    if extension in VALID_DOCUMENT_EXTENSIONS or file_bytes.startswith(b"%PDF"):
        pdf_name = ensure_pdf_name(filename)
        file_id, signed_url = upload_pdf_document(client, pdf_name, file_bytes)
        return {"type": "document_url", "document_url": signed_url}, file_id, signed_url

    encoded = base64.b64encode(file_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{encoded}"
    return {"type": "image_url", "image_url": data_url}, None, None


def collect_markdown_and_images(ocr_response):
    markdown_chunks: List[str] = []
    images: List[Dict[str, str]] = []
    rendered_markdown = ""

    for page in getattr(ocr_response, "pages", []) or []:
        page_markdown = getattr(page, "markdown", "") or getattr(page, "text", "")
        if page_markdown:
            markdown_chunks.append(page_markdown)
            rendered_markdown += ("\n\n" if rendered_markdown else "") + page_markdown

        for image in getattr(page, "images", []) or []:
            base64_data = getattr(image, "image_base64", None)
            if not base64_data:
                continue

            data_url = base64_data if "," in base64_data else f"data:image/png;base64,{base64_data}"
            image_id = getattr(image, "id", "") or getattr(image, "filename", "") or "ocr_image"

            images.append({
                "id": image_id,
                "data_url": data_url,
                "aliases": [image_id],
            })

            rendered_markdown = rendered_markdown.replace(f"![]({image_id})", f"![]({data_url})")
            rendered_markdown = rendered_markdown.replace(f"({image_id})", f"({data_url})")

    markdown_text = "\n\n".join(markdown_chunks).strip()
    return markdown_text, images


def extract_text_from_bytes(file_bytes: bytes, config: Dict, filename: Optional[str] = None):
    client = Mistral(api_key=read_mistral_key(config))
    guessed_name = filename or "upload.bin"

    document_source, file_id, signed_url = document_source_from_bytes(client, file_bytes, guessed_name)

    ocr_response = client.ocr.process(
        model=OCR_MODEL_NAME,
        document=document_source,
        include_image_base64=True,
    )

    markdown_text, images = collect_markdown_and_images(ocr_response)

    if file_id and signed_url:
        images.insert(0, {
            "id": "mistral_file",
            "file_id": file_id,
            "signed_url": signed_url,
            "data_url": signed_url,
            "aliases": ["mistral_file"],
        })

    return markdown_text, images

