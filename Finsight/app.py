from __future__ import annotations
from typing import List, Dict
import os
import requests
import streamlit as st
from dotenv import load_dotenv
from edgar import (
    get_filings_by_quarter,
    get_latest_10q_entries,
    get_recent_filings_by_forms,
    lookup_cik_for_ticker,
    process_filings,
    process_uploaded_files,
)
from ocr import extract_text_from_bytes
from rag import build_rag_index, query_rag_index
from ui import (
    render_chat_tab,
    render_fetch_tab,
    render_ocr_tab,
    render_report_tab,
    render_sidebar_config,
)


def initialize_app():
    st.set_page_config(page_title="Finsight RAG", layout="wide", initial_sidebar_state="expanded")
    project_root = os.path.dirname(__file__)
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path if os.path.exists(env_path) else None, override=False)
    ensure_session_state_keys()


def ensure_session_state_keys():
    config = st.session_state.get("config")
    if not isinstance(config, dict):
        config = {}

    project_root = os.path.dirname(__file__)
    env_path = os.path.join(project_root, ".env")
    config["env_file_found"] = os.path.exists(env_path)
    config["mistral_env_available"] = bool(os.getenv("MISTRAL_API_KEY"))

    config.setdefault("sec_user_agent", os.getenv("SEC_USER_AGENT", ""))

    defaults = {
        "config": config,
        "rag_index": None,
        "last_ticker": None,
        "source_chunks": [],
        "chat_history": [],
        "filings_preview": [],
        "indexed_documents": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    st.session_state["config"] = config


def determine_filings_to_index(cik: str, config: Dict, form_types, year, quarter, selected_idx):
    preview_entries = st.session_state.get("filings_preview") or []
    if preview_entries:
        if selected_idx:
            chosen: List[Dict] = []
            for index in selected_idx:
                if 0 <= index < len(preview_entries):
                    chosen.append(preview_entries[index])
            if chosen:
                return chosen
        return preview_entries

    quarter_map = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
    has_specific_quarter = isinstance(year, int) and quarter in quarter_map

    if has_specific_quarter:
        return get_filings_by_quarter(
            cik,
            config,
            year=year,
            quarter=quarter_map[quarter],
            limit=20,
        )

    entries = get_recent_filings_by_forms(cik, config, forms=form_types, limit=20)
    return entries


def handle_fetch_and_index(ticker, form_types, year, quarter, go, uploaded_files, selected_idx, progress):
    if not go:
        return

    st.session_state["last_ticker"] = ticker
    config = dict(st.session_state.get("config") or {})

    if uploaded_files:
        chunks = process_uploaded_files(uploaded_files, ticker, config, extract_text_from_bytes, progress.info)
    else:
        cik = lookup_cik_for_ticker(ticker, config)
        if not cik:
            progress.error(f"Unable to resolve ticker {ticker} to a CIK.")
            return

        entries = determine_filings_to_index(cik, config, form_types, year, quarter, selected_idx)
        if not entries:
            progress.error("No filings available to index.")
            return

        chunks = process_filings(
            entries,
            ticker,
            cik,
            config,
            extract_text_from_bytes,
            progress.info,
        )

    if not chunks:
        progress.error("No text extracted from the selected filings.")
        return

    rag_index = build_rag_index(chunks, config, progress.info)
    st.session_state["rag_index"] = rag_index
    st.session_state["source_chunks"] = chunks
    st.session_state["indexed_documents"] = rag_index.get("documents", [])

    doc_urls = {chunk.get("metadata", {}).get("doc_url", "") for chunk in chunks}
    doc_urls.discard("")

    unique_filings = len(doc_urls)
    filing_word = "filing" if unique_filings == 1 else "filings"
    chunk_count = len(st.session_state.get("indexed_documents") or [])
    chunk_word = "chunk" if chunk_count == 1 else "chunks"
    progress.success(f"Success! Indexed {chunk_count} {chunk_word} from {unique_filings} {filing_word}.")


def main():
    initialize_app()
    st.title("Finsight RAG")
    st.caption("Fetch SEC filings, chat with your financial data and generate reports")
    render_sidebar_config(st.session_state.get("config") or {})

    tab_fetch, tab_chat, tab_report, tab_ocr = st.tabs(
        ["📁 Fetch & Index", "💬 Chat", "📊 Generate Report", "🖼️ OCR Preview"]
    )

    with tab_fetch:
        st.markdown("### Fetch and Process SEC Filings")
        (
            ticker,
            form_types,
            year,
            quarter,
            go,
            uploaded_files,
            selected_idx,
            progress,
        ) = render_fetch_tab()
        handle_fetch_and_index(ticker, form_types, year, quarter, go, uploaded_files, selected_idx, progress)

    with tab_chat:
        if st.session_state.get("rag_index") is None:
            st.warning("No Index Available, Please fetch and index filings first to start chatting.")
        else:
            render_chat_tab()

    with tab_report:
        if st.session_state.get("rag_index") is None:
            st.warning("No Index Available, Please fetch and index filings first to generate reports.")
        else:
            render_report_tab()

    with tab_ocr:
        render_ocr_tab()


if __name__ == "__main__":
    main()
