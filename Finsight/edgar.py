from __future__ import annotations
import contextlib
import os
import re
from datetime import datetime
from importlib.util import find_spec
from typing import Any, Dict, List, Optional, Tuple, TypedDict
import requests

SEC_TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TMPL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"
DEFAULT_SEC_USER_AGENT = "FinsightRAG (example@gmail.com)"


class FilingEntry(TypedDict, total=False):
    accession_number: str
    filing_date: str
    primary_document: str
    form: str
    report_date: Optional[str]


class ChunkMetadata(TypedDict, total=False):
    ticker: str
    cik: Optional[str]
    filing_date: str
    doc_url: str
    form: str
    images: List[Dict[str, Any]]


class Chunk(TypedDict):
    text: str
    metadata: ChunkMetadata


def extract_email_from_ua(user_agent: Optional[str]):
    if not user_agent:
        return None
    match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", user_agent)
    if not match:
        return None
    return match.group(0)


def get_sec_headers(config: Dict[str, Any]):
    headers: Dict[str, str] = {
        "Accept": "application/json, text/html, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    configured = config.get("sec_user_agent") or DEFAULT_SEC_USER_AGENT
    headers["User-Agent"] = configured

    email_value = extract_email_from_ua(headers["User-Agent"])
    if email_value:
        headers["From"] = email_value

    return headers


def get_json(url: str, config: Dict[str, Any]):
    timeout_value = int(config.get("sec_timeout", 60) or 60)
    response = requests.get(url, headers=get_sec_headers(config), timeout=timeout_value)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected SEC response structure")
    return payload


def lookup_cik_for_ticker(ticker: str, config: Dict[str, Any]):
    data = get_json(SEC_TICKER_CIK_URL, config)
    target = ticker.upper()
    for entry in data.values():
        entry_ticker = entry.get("ticker", "").upper()
        if entry_ticker == target:
            cik_value = entry.get("cik_str")
            return str(cik_value) if cik_value is not None else None
    return None


def get_latest_10q_entries(cik: str, config: Dict[str, Any], limit: int = 1):
    cik_padded = str(cik).zfill(10)
    payload = get_json(SEC_SUBMISSIONS_URL_TMPL.format(cik_padded=cik_padded), config)
    recent = payload.get("filings", {}).get("recent", {})

    forms = list(recent.get("form", []))
    accession_numbers = list(recent.get("accessionNumber", []))
    filing_dates = list(recent.get("filingDate", []))
    primary_docs = list(recent.get("primaryDocument", []))

    results: List[FilingEntry] = []
    for form_value, accession_value, filing_date_value, primary_doc_value in zip(
        forms, accession_numbers, filing_dates, primary_docs
    ):
        if form_value != "10-Q":
            continue
        entry: FilingEntry = {
            "accession_number": accession_value.replace("-", ""),
            "filing_date": filing_date_value,
            "primary_document": primary_doc_value,
            "form": form_value,
        }
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def get_recent_filings_by_forms(cik: str, config: Dict[str, Any], forms: List[str], limit: int = 10,):
    cik_padded = str(cik).zfill(10)
    payload = get_json(SEC_SUBMISSIONS_URL_TMPL.format(cik_padded=cik_padded), config)
    recent = payload.get("filings", {}).get("recent", {})

    all_forms = list(recent.get("form", []))
    accession_numbers = list(recent.get("accessionNumber", []))
    filing_dates = list(recent.get("filingDate", []))
    primary_docs = list(recent.get("primaryDocument", []))

    allowed_forms = {item.upper() for item in forms}
    results: List[FilingEntry] = []

    for form_value, accession_value, filing_date_value, primary_doc_value in zip(
        all_forms, accession_numbers, filing_dates, primary_docs
    ):
        if allowed_forms and form_value.upper() not in allowed_forms:
            continue
        entry: FilingEntry = {
            "accession_number": accession_value.replace("-", ""),
            "filing_date": filing_date_value,
            "primary_document": primary_doc_value,
            "form": form_value,
        }
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def get_filings_by_quarter(cik: str, config: Dict[str, Any], year: Optional[int] = None, quarter: Optional[int] = None, limit: int = 1,):
    cik_padded = str(cik).zfill(10)
    payload = get_json(SEC_SUBMISSIONS_URL_TMPL.format(cik_padded=cik_padded), config)
    recent = payload.get("filings", {}).get("recent", {})

    forms = list(recent.get("form", []))
    accession_numbers = list(recent.get("accessionNumber", []))
    filing_dates = list(recent.get("filingDate", []))
    primary_docs = list(recent.get("primaryDocument", []))
    report_dates = list(recent.get("reportDate", []))
    while len(report_dates) < len(forms):
        report_dates.append(None)

    results: List[FilingEntry] = []
    for form_value, accession_value, filing_date_value, primary_doc_value, report_date_value in zip(
        forms, accession_numbers, filing_dates, primary_docs, report_dates
    ):
        allowed_forms = {"10-K", "10-K/A", "10-Q", "10-Q/A"}
        if quarter in (1, 2, 3):
            allowed_forms = {"10-Q", "10-Q/A"}
        if quarter == 4:
            allowed_forms = {"10-K", "10-K/A"}

        if form_value not in allowed_forms:
            continue

        parsed_date = parse_date_safe(report_date_value) or parse_date_safe(filing_date_value)
        if parsed_date is None:
            continue
        if year is not None and parsed_date.year != year:
            continue
        if quarter is not None and ((parsed_date.month - 1) // 3 + 1) != quarter:
            continue

        entry: FilingEntry = {
            "accession_number": accession_value.replace("-", ""),
            "filing_date": filing_date_value,
            "primary_document": primary_doc_value,
            "form": form_value,
        }
        results.append(entry)
        if len(results) >= limit:
            break
    return results


def build_primary_document_url(cik: str, accession_number: str, primary_document: str):
    return f"{SEC_ARCHIVES_BASE}/{int(cik)}/{accession_number}/{primary_document}"


def extract_date_from_url(url: str):
    if not url:
        return None

    filename = url.rsplit("/", 1)[-1] if "/" in url else url
    match = re.search(r"(\d{8})", filename)
    if not match:
        return None

    raw = match.group(1)
    year = raw[0:4]
    month = raw[4:6]
    day = raw[6:8]
    return f"{year}-{month}-{day}"


def fetch_primary_document(url: str, config: Dict[str, Any]):
    timeout_value = int(config.get("sec_timeout", 60) or 60)
    response = requests.get(url, headers=get_sec_headers(config), timeout=timeout_value)
    response.raise_for_status()
    return response.content, response.headers.get("Content-Type")


def convert_html_url_to_pdf_bytes(html_url: str, config: Dict[str, Any]):
    html_bytes, _ = fetch_primary_document(html_url, config)
    if not html_bytes:
        return None

    html_text = html_bytes.decode("utf-8", errors="ignore")
    return render_html_content_to_pdf_playwright(html_text, html_url, config)


def process_filings(entries: List[FilingEntry], ticker: str, cik: str, config: Dict[str, Any], ocr_func, progress_callback,):
    chunks: List[Chunk] = []
    total_entries = len(entries)

    index = 0
    while index < total_entries:
        entry = entries[index]
        result = download_and_extract_entry(
            entry,
            index + 1,
            total_entries,
            cik,
            config,
            ocr_func,
            progress_callback,
        )

        if result:
            text_value, doc_url, form_type, filing_date = result
            new_chunks = chunk_filing_text(text_value, ticker, cik, filing_date, doc_url, form_type)
            chunks.extend(new_chunks)
            progress_callback(f"Indexed {len(new_chunks)} chunks from {doc_url}")

        index += 1

    return chunks


def process_uploaded_files(uploaded_files: List[Any], ticker: str, config: Dict[str, Any], ocr_func, progress_callback,):
    chunks: List[Chunk] = []
    total_files = len(uploaded_files)

    index = 0
    while index < total_files:
        file_obj = uploaded_files[index]
        result = extract_uploaded_file(
            file_obj,
            index + 1,
            total_files,
            config,
            ocr_func,
            progress_callback,
        )

        if result:
            source_name, text_value = result
            new_chunks = chunk_filing_text(text_value, ticker, cik=None, filing_date="uploaded", doc_url=source_name, form="UPLOAD")
            chunks.extend(new_chunks)
            progress_callback(f"Indexed {len(new_chunks)} chunks from {source_name}")

        index += 1

    return chunks


def extract_text_safely(data: bytes, url: str, ocr_func, config: Dict[str, Any]):
    filename = resolve_filename_hint(url)
    if data.startswith(b"%PDF") and not filename.lower().endswith(".pdf"):
        if "." in filename:
            base_name = filename.rsplit(".", 1)[0]
        else:
            base_name = filename
        filename = f"{base_name}.pdf"

    result = ocr_func(data, config, filename=filename)

    if isinstance(result, tuple):
        if not result:
            return None
        return result[0]

    return result


def parse_date_safe(value: Optional[str]):
    if not value or len(value) != 10 or value[4] != "-" or value[7] != "-":
        return None

    year_part = value[:4]
    month_part = value[5:7]
    day_part = value[8:10]

    if not (year_part.isdigit() and month_part.isdigit() and day_part.isdigit()):
        return None

    year = int(year_part)
    month = int(month_part)
    day = int(day_part)

    if month < 1 or month > 12:
        return None

    thirty_one_months = {1, 3, 5, 7, 8, 10, 12}
    thirty_months = {4, 6, 9, 11}

    if month in thirty_one_months:
        max_day = 31
    elif month in thirty_months:
        max_day = 30
    else:
        is_leap_year = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        max_day = 29 if is_leap_year else 28

    if day < 1 or day > max_day:
        return None

    return datetime(year, month, day)


def ensure_base_tag(html_text: str, base_url: Optional[str]):
    if base_url is None or base_url == "":
        return html_text

    base_tag = f'<base href="{base_url}">'
    if "<head" in html_text.lower():
        start_index = html_text.lower().find("<head")
        end_index = html_text.find(">", start_index)
        if end_index != -1:
            return html_text[: end_index + 1] + base_tag + html_text[end_index + 1 :]
    return "<head>" + base_tag + "</head>" + html_text


def render_html_content_to_pdf_playwright(html_text: str, base_url: Optional[str], config: Dict[str, Any]):
    if find_spec("playwright.sync_api") is None:
        raise RuntimeError("Playwright import failed: module not found")

    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        headers = get_sec_headers(config)
        user_agent_value = headers.get("User-Agent")
        browser = playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"],
        )

        with contextlib.ExitStack() as stack:
            stack.callback(browser.close)
            context = browser.new_context(
                user_agent=user_agent_value or None,
                extra_http_headers=headers,
                device_scale_factor=2,
            )
            stack.callback(context.close)

            def handle_route(route):
                resource_type = route.request.resource_type
                allowed_types = {
                    "document",
                    "stylesheet",
                    "image",
                    "media",
                    "font",
                    "script",
                    "xhr",
                    "fetch",
                    "eventsource",
                }
                if resource_type in allowed_types:
                    route.continue_()
                else:
                    route.abort()

            context.route("**/*", handle_route)
            page = context.new_page()

            content_text = ensure_base_tag(html_text, base_url)
            page.set_content(content_text, wait_until="load")
            with contextlib.suppress(Exception):
                page.wait_for_load_state("networkidle", timeout=60000)
            with contextlib.suppress(Exception):
                page.wait_for_timeout(2000)
            pdf_bytes = page.pdf(format="Letter", print_background=True, prefer_css_page_size=True)
            if pdf_bytes:
                return pdf_bytes
            return None


def is_html_document(content_type: Optional[str], url: str):
    if content_type and "html" in content_type.lower():
        return True

    if url.lower().endswith((".htm", ".html")):
        return True

    return False


def normalize_filing_date(entry: FilingEntry, doc_url: str):
    date_from_url = extract_date_from_url(doc_url)
    if date_from_url:
        return date_from_url

    filing_date_value = entry.get("filing_date")
    if filing_date_value is None:
        return ""

    return filing_date_value


def resolve_filename_hint(url: str):
    if url.startswith("uploaded://"):
        suffix = url.split("uploaded://", 1)[1]
        if suffix:
            return suffix
        return "document.bin"

    if "/" in url:
        return url.rsplit("/", 1)[-1] or "document.bin"

    if url:
        return url

    return "document.bin"


def download_and_extract_entry(entry: FilingEntry, index: int, total: int, cik: str, config: Dict[str, Any], ocr_func, progress_callback,):
    form_type = entry.get("form", "10-Q")
    filing_date = entry.get("filing_date", "Unknown")
    progress_callback(f"[{index}/{total}] Downloading {form_type} {filing_date}...")

    accession_number = entry.get("accession_number", "")
    primary_document = entry.get("primary_document", "")
    doc_url = build_primary_document_url(cik, accession_number, primary_document)

    data, content_type = fetch_primary_document(doc_url, config)

    if is_html_document(content_type, doc_url):
        progress_callback(f"[{index}/{total}] Converting HTML to PDF...")
        pdf_bytes = convert_html_url_to_pdf_bytes(doc_url, config)
        if not pdf_bytes:
            progress_callback(f"HTML→PDF conversion failed for {doc_url}")
            return None
        data = pdf_bytes
    else:
        progress_callback(f"[{index}/{total}] Running OCR...")

    text_value = extract_text_safely(data, doc_url, ocr_func, config)
    if not text_value:
        progress_callback(f"No text extracted for {doc_url}")
        return None

    normalized_date = normalize_filing_date(entry, doc_url)
    return text_value, doc_url, form_type, normalized_date


def extract_uploaded_file(file_obj, index: int, total: int, config: Dict[str, Any], ocr_func, progress_callback):
    name_value = getattr(file_obj, "name", None)
    if not name_value:
        name_value = f"upload_{index}"

    source_name = f"uploaded://{name_value.lower()}"
    progress_callback(f"[{index}/{total}] Processing {name_value}...")

    data = file_obj.read()
    if not data:
        progress_callback(f"No data found in {name_value}")
        return None

    if name_value.lower().endswith((".htm", ".html")):
        progress_callback(f"[{index}/{total}] Converting uploaded HTML to PDF...")
        pdf_bytes = render_html_content_to_pdf_playwright(data.decode("utf-8", errors="ignore"), base_url=None, config=config)
        if not pdf_bytes:
            progress_callback(f"HTML→PDF conversion failed for {source_name}")
            return None
        data = pdf_bytes
    else:
        progress_callback(f"[{index}/{total}] Running OCR on uploaded bytes...")

    text_value = extract_text_safely(data, source_name, ocr_func, config)
    if not text_value:
        progress_callback(f"OCR produced no text for {source_name}")
        return None

    return source_name, text_value


def chunk_filing_text(text: str, ticker: str, cik: Optional[str], filing_date: str, doc_url: str, form: str):
    metadata: ChunkMetadata = {}
    metadata["ticker"] = ticker.upper()
    metadata["filing_date"] = filing_date
    metadata["doc_url"] = doc_url
    metadata["form"] = form
    if cik is not None:
        metadata["cik"] = cik
    return [{"text": text, "metadata": metadata}]
