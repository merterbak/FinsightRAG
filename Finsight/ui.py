from __future__ import annotations
import base64
import os
import re
import time
from datetime import date
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
from PIL import Image
from chat import stream_chat_with_mistral
from edgar import (
    DEFAULT_SEC_USER_AGENT,
    build_primary_document_url,
    convert_html_url_to_pdf_bytes,
    extract_date_from_url,
    fetch_primary_document,
    get_filings_by_quarter,
    get_recent_filings_by_forms,
    is_html_document,
    lookup_cik_for_ticker,
)
from ocr import extract_text_from_bytes
from rag import query_rag_index
from report import markdown_to_pdf_bytes


def fix_preview_dates(entries: List[Dict], cik: str):
    fixed: List[Dict] = []
    for entry in entries:
        doc_url = build_primary_document_url(cik, entry.get("accession_number", ""), entry.get("primary_document", ""))
        correct_date = extract_date_from_url(doc_url) or entry.get("filing_date")
        updated = entry.copy()
        updated["filing_date"] = correct_date
        fixed.append(updated)
    return fixed


def render_sidebar_config(config: Dict):
    with st.sidebar:
        st.subheader("Configuration")

        st.caption(
            "SEC requires a contact email in the User-Agent for EDGAR requests."
        )

        sec_value = st.text_input(
            "SEC User-Agent",
            value=config.get("sec_user_agent") or DEFAULT_SEC_USER_AGENT,
            key="sidebar_sec_user_agent",
            help="SEC recommends including a contact email.",
        )

        mistral_env_available = bool(os.getenv("MISTRAL_API_KEY"))
        mistral_placeholder = "" if mistral_env_available else config.get("mistral_api_key", "")
        mistral_value = None
        if not mistral_env_available:
            mistral_value = st.text_input(
                "Mistral API Key",
                value=mistral_placeholder,
                type="password",
                key="sidebar_mistral_api_key",
                help="basically stored in session only when not provided via .env.",
            )

        if st.button("Apply Config"):
            st.session_state.setdefault("config", {})["sec_user_agent"] = sec_value
            if mistral_value:
                st.session_state["config"]["mistral_api_key"] = mistral_value
            st.success("Configuration updated")


def handle_fetch_preview(ticker: str, form_types: List[str], year: Optional[int], quarter: Optional[str], progress):
    if not ticker or not form_types:
        return

    config = st.session_state.get("config", {})
    cik = lookup_cik_for_ticker(ticker, config)
    st.session_state["last_ticker"] = ticker
    if not cik:
        st.session_state["filings_preview"] = []
        progress.error(f"Unable to resolve ticker {ticker} to a CIK.")
        return

    quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    has_specific_quarter = isinstance(year, int) and quarter in quarter_map

    if has_specific_quarter:
        entries = get_filings_by_quarter(
            cik,
            config,
            year=int(year),
            quarter=quarter_map.get(quarter),
            limit=20,
        )
    else:
        entries = get_recent_filings_by_forms(cik, config, forms=form_types, limit=20)

    fixed_entries = fix_preview_dates(entries, cik)
    st.session_state["filings_preview"] = fixed_entries
    progress.success(f"Found {len(fixed_entries)} filings for {ticker}")


def render_filings_selection(entries: List[Dict]):
    if not entries:
        return []

    st.markdown("### Select Filings to Index")
    selected_idx: List[int] = []
    columns = st.columns(3)
    for idx, entry in enumerate(entries):
        with columns[idx % 3]:
            doc = str(entry.get("primary_document", ""))
            if len(doc) > 30:
                doc = doc[:27] + "..."
            form = entry.get("form", "")
            filing_date = entry.get("filing_date", "")
            label = f"**{form}** • {filing_date}\n`{doc}`"
            if st.checkbox(label, key=f"sel_{idx}"):
                selected_idx.append(idx)

    col1, col2, _ = st.columns([1, 1, 1])
    if st.button("Select All", key="select_all_btn"):
        for idx in range(len(entries)):
            st.session_state[f"sel_{idx}"] = True
        selected_idx = list(range(len(entries)))
    if st.button("Select None", key="select_none_btn"):
        for idx in range(len(entries)):
            st.session_state[f"sel_{idx}"] = False
        selected_idx = []

    return selected_idx


def render_fetch_tab():
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker_value = st.text_input(
            "Ticker Symbol",
            value=(st.session_state.get("last_ticker") or "AAPL"),
            key="ticker_input",
        )
        ticker = ticker_value.upper()
    with col2:
        st.markdown("**Examples:** AAPL, TSLA, MSFT, GOOGL, AMZN")

    st.markdown("### Selection Criteria")
    col_form, col_year, col_quarter = st.columns([1, 1, 1])
    with col_form:
        form_types = st.multiselect(
            "Form Types",
            ["10-K", "10-Q", "8-K"],
            default=["10-K", "10-Q"],
            key="form_types",
        )
    with col_year:
        current_year = date.today().year
        year_options: List[Any] = ["Any"] + list(range(current_year, 2019, -1))
        selected_year = st.selectbox("Year", options=year_options, index=0, key="year_select")
        year_value = int(selected_year) if isinstance(selected_year, int) else None
    with col_quarter:
        quarter_choice = st.selectbox(
            "Quarter",
            options=["Any", "Q1", "Q2", "Q3", "Q4"],
            index=0,
            key="quarter_select",
        )
        quarter_value = quarter_choice if quarter_choice != "Any" else None

    col_preview, col_process = st.columns([1, 1])
    with col_preview:
        preview = st.button("Preview Filings", key="preview_btn", use_container_width=True)
    with col_process:
        go = st.button("Process & Index", key="fetch_btn", use_container_width=True, type="primary")

    progress = st.empty()
    if preview:
        handle_fetch_preview(ticker, form_types, year_value, quarter_value, progress)

    entries = st.session_state.get("filings_preview") or []
    selected_idx = render_filings_selection(entries)

    st.markdown("### Upload Local Files (Optional)")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "htm", "html"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        st.success(f"Uploaded {len(uploaded_files)} file(s)")

    indexed_documents = st.session_state.get("indexed_documents") or []
    if indexed_documents:
        unique_docs = {getattr(doc, "metadata", {}).get("doc_url") for doc in indexed_documents}
        unique_docs.discard("")
        if unique_docs:
            st.info(
                f"Indexed {len(indexed_documents)} chunks across {len(unique_docs)} filings in the current session."
            )

    return ticker, form_types, year_value, quarter_value, go, uploaded_files, selected_idx, progress


def render_chat_tab():
    col_input, col_controls = st.columns([2, 1])
    with col_input:
        user_input = st.chat_input(
            "Ask about revenue, margins, financial metrics etc.",
            key="chat_input",
        )
    with col_controls:
        k = st.slider("Context Chunks", min_value=2, max_value=10, value=5, key="chat_topk")
        if st.button("Clear Chat", use_container_width=True, key="clear_chat_btn"):
            st.session_state["chat_history"] = []
            st.rerun()
        with st.expander("Sources for Latest Question", expanded=False):
            render_sources_panel()

    if not user_input:
        render_chat_history()
        return

    if st.session_state.get("rag_index") is None:
        st.warning("Please fetch and index filings first.")
        return

    handle_chat_input(user_input, k)


def render_report_tab():
    st.subheader("Generate Report")
    st.caption("Create a simplified structured markdown report from filings.")

    indexed_docs = st.session_state.get("indexed_documents") or []
    doc_options: Dict[str, Dict[str, Any]] = {}
    for doc in indexed_docs:
        metadata = getattr(doc, "metadata", {}) or {}
        doc_url = metadata.get("doc_url")
        if not doc_url or doc_url in doc_options:
            continue
        doc_options[doc_url] = metadata

    sorted_options = sorted(
        doc_options.items(),
        key=lambda item: item[1].get("filing_date", "") + item[1].get("form", "") + item[1].get("ticker", ""),
    )
    doc_urls = [item[0] for item in sorted_options]
    doc_labels = [
        f"{metadata.get('form', 'Form')} • {metadata.get('filing_date', 'Unknown date')} • {metadata.get('ticker', '?')}"
        for _, metadata in sorted_options
    ]

    with st.form("report_form"):
        default_ticker = st.session_state.get("last_ticker", "AAPL").upper()
        default_title = f"{default_ticker} 10-Q Summary Report"
        report_title = st.text_input("Report title", value=default_title)
        report_date = st.date_input("Report date", value=date.today())
        include_sections = st.multiselect(
            "Sections",
            [
                "Executive Summary",
                "Key Financials",
                "Management Discussion",
                "Liquidity & Capital Resources",
                "Capital Allocation",
                "Risks",
                "Outlook",
                "Sources",
            ],
            default=["Executive Summary", "Key Financials", "Risks", "Outlook", "Sources"],
        )
        target_pages = st.slider("Target PDF length (pages)", 2, 8, 4)
        max_context = st.slider("Max context chunks", 5, 40, 28)

        if doc_urls:
            selected_docs = st.multiselect(
                "Include filings",
                options=doc_urls,
                default=doc_urls,
                format_func=lambda url: doc_labels[doc_urls.index(url)],
            )
        else:
            selected_docs = []

        submit_report = st.form_submit_button("Generate Report")

    if submit_report:
        if st.session_state.get("rag_index") is None:
            st.warning("Please fetch and index filings first.")
        else:
            generate_report(
                report_title,
                report_date,
                include_sections,
                target_pages,
                max_context,
                selected_docs,
            )


def handle_chat_input(user_input: str, k: int):
    retrieval = query_rag_index(st.session_state["rag_index"], user_input, top_k=int(k))
    context_blocks = [chunk.get("text", "") for chunk, _ in retrieval]
    context_text = "\n\n".join(context_blocks)

    conversation: List[Dict[str, str]] = []
    conversation.append({"role": "system", "content": "You are a helpful assistant for financial filings."})
    history = st.session_state.get("chat_history", [])[-6:]
    for item in history:
        conversation.append({"role": item.get("role", "user"), "content": item.get("content", "")})
    conversation.append({"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_input}"})

    st.session_state.setdefault("chat_history", [])
    st.session_state["chat_history"].append({"role": "user", "content": user_input, "sources": retrieval})

    with st.chat_message("user"):
        st.write(user_input)

    answer_text = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        last_update = 0.0
        for token in stream_chat_with_mistral(conversation, st.session_state.get("config", {})):
            if not token:
                continue
            answer_text += token
            now = time.monotonic()
            if (now - last_update) >= 0.03 or token.endswith(("\n", " ", ".")):
                placeholder.markdown(answer_text + "▌")
                last_update = now
        placeholder.markdown(answer_text)

    st.session_state["chat_history"].append({"role": "assistant", "content": answer_text})
    st.rerun()


def render_sources_panel():
    history = st.session_state.get("chat_history", [])
    last_sources = None
    for item in reversed(history):
        if item.get("role") == "user" and item.get("sources"):
            last_sources = item.get("sources")
            break

    if not last_sources:
        st.caption("No sources yet.")
        return

    lines: List[str] = []
    for idx, (chunk, score) in enumerate(last_sources, start=1):
        metadata = chunk.get("metadata", {})
        url = metadata.get("doc_url", "")
        filing_date = metadata.get("filing_date", "")
        snippet = chunk.get("text", "")
        if len(snippet) > 800:
            snippet = snippet[:800] + "..."

        mistral_line = ""
        for img in metadata.get("images", []) or []:
            signed = img.get("signed_url")
            if signed:
                mistral_line = f"\nMistral file: {signed}"
                break

        lines.append(f"**{idx}.** {snippet}\n\n{url}{mistral_line}\n{filing_date} · sim={score:.3f}\n")

    st.markdown("\n".join(lines))


def generate_report(title: str, report_date: date, sections: List[str], target_pages: int, max_context: int, selected_doc_urls: Optional[List[str]] = None,):
    broad_query = (
        "financial performance trends management discussion liquidity cash flow risks outlook guidance metrics "
        "customers segments revenue expenses capital expenditures debt cash flow dividends buybacks investments headcount"
    )
    retrieval = query_rag_index(
        st.session_state["rag_index"],
        broad_query,
        top_k=int(max_context),
        allowed_doc_urls=selected_doc_urls,
    )

    context_blocks = [chunk.get("text", "") for chunk, _ in retrieval]
    context_text = "\n\n".join(context_blocks)
    sections_text = ", ".join(sections)

    system_message = (
        "Create a clear, well structured markdown report strictly using the given context. "
        f"Target approximately {target_pages} pages with detailed sections, tables, scenario analysis, and Q/Q and Y/Y comparisons when possible. "
        "Provide: 1) Executive Summary with 5-8 bullets (strategy, highlights, risks, catalysts), 2) KPIs and key financials with numbers (revenue, margins, income, EPS, segments), 3) Management discussion, 4) Liquidity and capital resources (cash, debt, credit lines, maturities, FCF), 5) Capital allocation (capex, R&D, buybacks, dividends, acquisitions), 6) Risks and mitigation, 7) Outlook with scenarios and catalysts, 8) Sources. "
        "Cite sources inline as footnotes like [^1] and include a Sources section mapping footnotes to URLs and filing dates. "
        "Call out data gaps explicitly and avoid hallucination."
    )

    user_message = (
        f"Report title: {title}\n"
        f"Date: {report_date}\n"
        f"Sections to include: {sections_text}\n\n"
        f"Context (verbatim excerpts from filings):\n{context_text}"
    )

    messages = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]
    parts: List[str] = []
    for token in stream_chat_with_mistral(
        messages,
        st.session_state.get("config", {}),
        model="mistral-large-latest",
        temperature=0.2,
    ):
        if token:
            parts.append(token)
    report_md = "".join(parts)

    st.markdown("**Report Preview**")
    st.markdown(report_md)
    with st.expander("Markdown source"):
        st.code(report_md, language="markdown")
    st.download_button(
        label="Download Markdown",
        data=report_md.encode("utf-8"),
        file_name=f"{title.replace(' ', '_')}.md",
        mime="text/markdown",
    )

    pdf_bytes = markdown_to_pdf_bytes(report_md, title=title)
    st.download_button(
        label="Download PDF",
        data=pdf_bytes,
        file_name=f"{title.replace(' ', '_')}.pdf",
        mime="application/pdf",
    )


def render_ocr_tab():
    st.subheader("SEC Report OCR")
    st.caption("Select a recent SEC filing and extract text and images via OCR.")

    config = st.session_state.get("config", {})

    col_ticker, col_form = st.columns([1, 1])
    with col_ticker:
        ticker_value = st.text_input(
            "Ticker Symbol",
            value=(st.session_state.get("last_ticker") or "AAPL"),
            key="ocr_ticker",
        )
        ticker = ticker_value.upper()
    with col_form:
        form_types = st.multiselect(
            "Form Types",
            ["10-K", "10-Q"],
            default=["10-Q", "10-K"],
            key="ocr_forms",
        )

    col_preview, col_run = st.columns([1, 1])
    with col_preview:
        preview = st.button("Preview SEC Filings", key="ocr_preview_btn", use_container_width=True)
    with col_run:
        run = st.button("Run OCR", key="ocr_run_btn", type="primary", use_container_width=True)

    if preview:
        preview_ocr_filings(ticker, form_types, config)

    entries = st.session_state.get("ocr_filings_preview") or []
    selected_idx = select_ocr_filing(entries)

    if run:
        execute_ocr_run(entries, selected_idx, config)


def preview_ocr_filings(ticker: str, form_types: List[str], config: Dict):
    if not ticker or not form_types:
        return

    cik = lookup_cik_for_ticker(ticker, config)
    if not cik:
        st.session_state["ocr_filings_preview"] = []
        st.error(f"Unable to resolve ticker {ticker} to a CIK.")
        return

    entries = get_recent_filings_by_forms(cik, config, forms=form_types, limit=20)
    fixed_entries = fix_preview_dates(entries, cik)
    st.session_state["ocr_filings_preview"] = fixed_entries
    st.session_state["ocr_cik"] = cik
    st.success(f"Found {len(fixed_entries)} filings for {ticker}")


def select_ocr_filing(entries: List[Dict]):
    if not entries:
        return None

    st.markdown("### Select a Filing")
    labels = []
    for entry in entries:
        doc = str(entry.get("primary_document", ""))
        if len(doc) > 34:
            doc = doc[:31] + "..."
        labels.append(f"{entry.get('form', '')} • {entry.get('filing_date', '')} • `{doc}`")

    selected = st.radio(
        "Select a filing",
        options=list(range(len(entries))),
        format_func=lambda i: labels[i],
        key="ocr_sel_idx",
    )
    return selected


def execute_ocr_run(entries: List[Dict], selected_idx: Optional[int], config: Dict):
    cik = st.session_state.get("ocr_cik")
    if not entries or selected_idx is None or cik is None:
        st.warning("Please preview and select a filing first.")
        return

    entry = entries[int(selected_idx)]
    doc_url = build_primary_document_url(cik, entry.get("accession_number", ""), entry.get("primary_document", ""))

    progress = st.empty()
    progress.info("Fetching filing...")
    data, content_type = fetch_primary_document(doc_url, config)
    if is_html_document(content_type, doc_url):
        progress.info("Converting HTM file to PDF for OCR compatibility")
        pdf_bytes = convert_html_url_to_pdf_bytes(doc_url, config)
        if not pdf_bytes:
            progress.error("HTML→PDF conversion failed.")
            return
        data = pdf_bytes

    progress.info("Running Mistral OCR...")
    filename = doc_url.rsplit("/", 1)[-1]
    if data and isinstance(data, (bytes, bytearray)) and data[:4] == b"%PDF" and not filename.endswith(".pdf"):
        if "." in filename:
            filename = filename.rsplit(".", 1)[0] + ".pdf"
        else:
            filename = filename + ".pdf"

    markdown_text, images = extract_text_from_bytes(data, config, filename=filename)
    progress.empty()

    display_ocr_results(doc_url, markdown_text, images)


def display_ocr_results(doc_url: str, markdown_text: str, images: List[Dict]):
    rendered_md = markdown_text or ""
    for image in images or []:
        image_id = image.get("id", "")
        data_url = image.get("data_url", "")
        if not image_id or not data_url:
            continue
        pattern_md = rf"!\[[^\]]*\]\(\s*{re.escape(image_id)}\s*\)"
        rendered_md = re.sub(pattern_md, f"![image]({data_url})", rendered_md)
        rendered_md = rendered_md.replace(f"({image_id})", f"({data_url})")
        pattern_html = rf'<img([^>]+)src="\s*{re.escape(image_id)}\s*"([^\"]*)>'
        replacement_html = f'<img\\1src="{data_url}"\\2>'
        rendered_md = re.sub(pattern_html, replacement_html, rendered_md, flags=re.IGNORECASE)

    st.markdown(f"**Source:** {doc_url}")
    col_text, col_images = st.columns([2, 1])
    with col_text:
        st.markdown("### Rendered Markdown")
        if rendered_md:
            st.markdown(rendered_md)
        else:
            st.caption("No text extracted.")
        with st.expander("Extracted Markdown (raw)"):
            st.text(rendered_md)

    with col_images:
        st.markdown("### Extracted Images")
        shown = 0
        for image in images or []:
            data_url = image.get("data_url", "")
            if not data_url or not data_url.startswith("data:image"):
                continue

            encoded_part = data_url.split(",", 1)[1] if "," in data_url else data_url
            image_bytes = base64.b64decode(encoded_part)
            pil_image = Image.open(BytesIO(image_bytes))
            st.image(pil_image, width="stretch")

            st.markdown(
                f"<img src=\"{data_url}\" style=\"max-width:100%; width:100%; height:auto;\" />",
                unsafe_allow_html=True,
            )
            shown += 1

        if shown == 0:
            st.caption("No images extracted.")
        else:
            st.caption(f"Images extracted: {shown}")


def render_chat_history():
    history = st.session_state.get("chat_history") or []
    if not history:
        return

    idx = len(history) - 1
    while idx >= 0:
        current = history[idx]
        previous = history[idx - 1] if idx - 1 >= 0 else None
        if previous and previous.get("role") == "user" and current.get("role") == "assistant":
            with st.chat_message("user"):
                st.write(previous.get("content", ""))
            with st.chat_message("assistant"):
                st.write(current.get("content", ""))
            idx -= 2
            continue

        role = current.get("role", "user")
        with st.chat_message(role):
            st.write(current.get("content", ""))
        idx -= 1
