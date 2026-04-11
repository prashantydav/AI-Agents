import argparse
from pathlib import Path

from app.retriever import SemanticSearchEngine
from app.utils import load_documents


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic search embeddings and Chroma collections.")
    parser.add_argument("--strategy", choices=["fixed", "recursive", "semantic"], default="semantic", help="Chunking strategy to ingest.")
    parser.add_argument("--source", type=Path, help="Optional CSV file with id,title,text columns.")
    parser.add_argument("--force", action="store_true", help="Force rebuild embedding cache and store.")
    args = parser.parse_args()

    documents = load_documents(args.source) if args.source else load_documents()
    engine = SemanticSearchEngine(documents)
    store = engine.get_store(args.strategy)
    store.prepare(force=args.force)

    print(f"Ingested {len(store.chunks)} chunks for strategy '{args.strategy}'.")


if __name__ == "__main__":
    main()
