from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

#lightweight RAG
def build_documents(raw_chunks: List[Dict]):
    documents: List[Document] = []
    for chunk in raw_chunks:
        text = chunk.get("text", "")
        metadata = dict(chunk.get("metadata", {}) or {})
        documents.append(Document(page_content=text, metadata=metadata))
    return documents


def build_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])


def build_vector_store(documents: List[Document]):
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    return FAISS.from_documents(documents, embeddings)


def build_retrievers(split_documents: List[Document]):
    vector_store = build_vector_store(split_documents)
    faiss = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 40, "lambda_mult": 0.7},
    )

    bm25 = BM25Retriever.from_documents(split_documents)
    bm25.k = 10

    ensemble = EnsembleRetriever(retrievers=[faiss, bm25], weights=[0.6, 0.4])

    llm = ChatMistralAI(model="mistral-medium-2508", temperature=0.0)
    multi_query = MultiQueryRetriever.from_llm(retriever=ensemble, llm=llm)

    return {
        "faiss": faiss,
        "bm25": bm25,
        "ensemble": ensemble,
        "multi_query": multi_query,
    }


def build_rag_index(chunks: List[Dict], config: Dict, progress_callback):
    progress_callback("Preparing documents for indexing...")
    documents = build_documents(chunks)

    progress_callback("Splitting documents into chunks...")
    splitter = build_splitter()
    split_documents = splitter.split_documents(documents)

    progress_callback("Building retrievers...")
    retrievers = build_retrievers(split_documents)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return {
        "faiss_retriever": retrievers["faiss"],
        "bm25_retriever": retrievers["bm25"],
        "ensemble_retriever": retrievers["ensemble"],
        "retriever": retrievers["multi_query"],
        "cross_encoder": cross_encoder,
        "source_chunks": chunks,
        "documents": split_documents,
    }


def merge_chat_history(query: str, chat_history: Optional[Iterable[Dict[str, str]]]):
    if not chat_history:
        return query

    user_lines = [item.get("content", "") for item in chat_history if item.get("role") == "user"]
    history_text = "\n".join(filter(None, user_lines))
    return f"{history_text}\nCurrent question: {query}" if history_text else query


def collect_results(ranked: List[Tuple[Document, float]], source_chunks: List[Dict], allowed_urls: Optional[Iterable[str]], top_k: int):
    allowed = {url for url in allowed_urls or [] if url}
    results: List[Tuple[Dict, float]] = []

    for document, score in ranked:
        if len(results) >= top_k:
            break

        matched_chunk = next(
            (
                chunk
                for chunk in source_chunks
                if chunk.get("text") == document.page_content
                and all(chunk.get("metadata", {}).get(key) == value for key, value in document.metadata.items())
            ),
            None,
        )

        payload = matched_chunk or {"text": document.page_content, "metadata": document.metadata}

        if allowed:
            doc_url = payload.get("metadata", {}).get("doc_url")
            if doc_url and doc_url not in allowed:
                continue

        results.append((payload, float(score)))

    return results


def query_rag_index(rag_index: Dict,query: str,top_k: int = 5,chat_history: Optional[Iterable[Dict[str, str]]] = None,use_multi_query: bool = False,allowed_doc_urls: Optional[Iterable[str]] = None,):
    if not query:
        return []

    retriever = rag_index["retriever"] if use_multi_query else rag_index["ensemble_retriever"]
    cross_encoder = rag_index["cross_encoder"]
    faiss_retriever = rag_index["faiss_retriever"]
    bm25_retriever = rag_index["bm25_retriever"]

    faiss_retriever.search_kwargs.update({"k": top_k, "fetch_k": max(4 * top_k, 20)})
    bm25_retriever.k = max(3 * top_k, 15)

    retriever_input = merge_chat_history(query, chat_history)
    candidates = retriever.invoke(retriever_input)
    if not candidates:
        return []

    scored = zip(candidates, cross_encoder.predict([[query, doc.page_content] for doc in candidates]))
    ranked = sorted(scored, key=lambda item: float(item[1]), reverse=True)

    return collect_results(ranked, rag_index["source_chunks"], allowed_doc_urls, top_k)


