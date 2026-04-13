import numpy as np
import pandas as pd
from pathlib import Path
from typing import Iterable, List

from config import DATA_DIR

SUPPORTED_SOURCE_EXTENSIONS = {
    ".csv",
    ".docx",
    ".json",
    ".md",
    ".pdf",
    ".txt",
    ".xls",
    ".xlsx",
}


def _normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _make_document(doc_id: str, title: str, text: str, source_path: Path, **metadata) -> dict:
    document = {
        "id": doc_id,
        "title": title,
        "text": text.strip(),
        "source_path": str(source_path),
        "source_type": source_path.suffix.lower().lstrip("."),
    }
    document.update({key: value for key, value in metadata.items() if value not in (None, "")})
    return document


def _load_csv_documents(path: Path) -> List[dict]:
    df = pd.read_csv(path)
    columns = {column.lower(): column for column in df.columns}
    has_text_column = "text" in columns
    documents = []

    if has_text_column:
        id_column = columns.get("id")
        title_column = columns.get("title")
        text_column = columns["text"]
        for index, row in df.iterrows():
            text = _normalize_text(row.get(text_column))
            if not text:
                continue
            doc_id = _normalize_text(row.get(id_column)) or f"{path.stem}-{index}"
            title = _normalize_text(row.get(title_column)) or path.stem
            documents.append(_make_document(doc_id, title, text, path, row=index))
        return documents

    for index, row in df.fillna("").iterrows():
        row_lines = [f"{column}: {_normalize_text(value)}" for column, value in row.items() if _normalize_text(value)]
        text = "\n".join(row_lines).strip()
        if not text:
            continue
        doc_id = f"{path.stem}-row-{index}"
        title = f"{path.stem} row {index}"
        documents.append(_make_document(doc_id, title, text, path, row=index))

    return documents


def _load_excel_documents(path: Path) -> List[dict]:
    sheets = pd.read_excel(path, sheet_name=None)
    documents = []

    for sheet_name, df in sheets.items():
        for index, row in df.fillna("").iterrows():
            row_lines = [f"{column}: {_normalize_text(value)}" for column, value in row.items() if _normalize_text(value)]
            text = "\n".join(row_lines).strip()
            if not text:
                continue
            doc_id = f"{path.stem}-{sheet_name}-{index}"
            title = f"{path.stem} - {sheet_name} - row {index}"
            documents.append(
                _make_document(
                    doc_id,
                    title,
                    text,
                    path,
                    sheet=sheet_name,
                    row=index,
                )
            )

    return documents


def _load_text_document(path: Path) -> List[dict]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [_make_document(path.stem, path.stem, text, path)]


def _load_json_documents(path: Path) -> List[dict]:
    df = pd.read_json(path)
    if isinstance(df, pd.DataFrame):
        columns = {column.lower(): column for column in df.columns}
        if "text" in columns:
            id_column = columns.get("id")
            title_column = columns.get("title")
            text_column = columns["text"]
            documents = []
            for index, row in df.iterrows():
                text = _normalize_text(row.get(text_column))
                if not text:
                    continue
                doc_id = _normalize_text(row.get(id_column)) or f"{path.stem}-{index}"
                title = _normalize_text(row.get(title_column)) or path.stem
                documents.append(_make_document(doc_id, title, text, path, row=index))
            return documents

        documents = []
        for index, row in df.fillna("").iterrows():
            row_lines = [f"{column}: {_normalize_text(value)}" for column, value in row.items() if _normalize_text(value)]
            text = "\n".join(row_lines).strip()
            if not text:
                continue
            documents.append(
                _make_document(
                    f"{path.stem}-row-{index}",
                    f"{path.stem} row {index}",
                    text,
                    path,
                    row=index,
                )
            )
        return documents
    raise ValueError(f"Unsupported JSON structure in {path}. Expected a JSON object array.")


def _load_pdf_documents(path: Path) -> List[dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF ingestion requires `pypdf`. Install dependencies from requirements.txt.") from exc

    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        documents.append(
            _make_document(
                f"{path.stem}-page-{page_number}",
                f"{path.stem} - page {page_number}",
                text,
                path,
                page=page_number,
            )
        )
    return documents


def _load_docx_documents(path: Path) -> List[dict]:
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError("Word ingestion requires `python-docx`. Install dependencies from requirements.txt.") from exc

    document = Document(str(path))
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    text = "\n".join(paragraphs).strip()
    if not text:
        return []
    return [_make_document(path.stem, path.stem, text, path)]


def _iter_source_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SOURCE_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {path.suffix}. Supported: {sorted(SUPPORTED_SOURCE_EXTENSIONS)}"
            )
        return [path]

    if not path.exists():
        raise FileNotFoundError(f"Document source not found: {path}")

    if not path.is_dir():
        raise ValueError(f"Unsupported source path: {path}")

    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SOURCE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No supported document files found in {path}. Supported: {sorted(SUPPORTED_SOURCE_EXTENSIONS)}"
        )
    return files


def _load_documents_from_file(path: Path) -> List[dict]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _load_csv_documents(path)
    if suffix in {".xlsx", ".xls"}:
        return _load_excel_documents(path)
    if suffix == ".pdf":
        return _load_pdf_documents(path)
    if suffix == ".docx":
        return _load_docx_documents(path)
    if suffix in {".txt", ".md"}:
        return _load_text_document(path)
    if suffix == ".json":
        return _load_json_documents(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def load_documents(source_path: Path = None) -> List[dict]:
    documents = []
    path = source_path or DATA_DIR
    for file_path in _iter_source_files(path):
        documents.extend(_load_documents_from_file(file_path))

    if not documents:
        raise ValueError(f"No ingestible content found in {path}.")

    return documents


def to_serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value
