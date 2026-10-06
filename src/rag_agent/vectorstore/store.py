from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from loguru import logger

from rag_agent.agent.state import (
    ChunkMetadata,
    DocumentChunk,
    IngestionResult,
    RetrievedChunk,
)
from rag_agent.config import get_settings


class VectorStoreManager:
    """Manage ChromaDB storage and retrieval."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._collection = self._initialise()

    @staticmethod
    def generate_chunk_id(source: str, text: str) -> str:
        """Generate a deterministic ID for a document chunk."""
        import hashlib

        content = f"{source}:{text}".encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def _initialise(self):
        """Initialize the persistent ChromaDB collection."""

        try:
            chroma_path = Path(self.settings.chroma_db_path)
            chroma_path.mkdir(parents=True, exist_ok=True)

            client = chromadb.PersistentClient(
                path=str(chroma_path)
            )

            collection = client.get_or_create_collection(
                name=self.settings.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                f"ChromaDB initialised: "
                f"collection='{self.settings.chroma_collection_name}', "
                f"items={collection.count()}"
            )

            return collection

        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise ChromaDB: {exc}"
            ) from exc

    def check_duplicate(self, chunk_id: str) -> bool:
        """Check whether a chunk already exists."""

        try:
            result = self._collection.get(
                ids=[chunk_id]
            )

            ids = result.get("ids", [])

            return chunk_id in ids

        except Exception:
            return False

    def ingest(
        self,
        chunks: list[DocumentChunk],
    ) -> IngestionResult:
        """Add document chunks to ChromaDB."""

        result = IngestionResult()

        if not chunks:
            return result

        embeddings = self._get_embeddings()

        batch_size = 100

        for start in range(0, len(chunks), batch_size):

            batch = chunks[
                start : start + batch_size
            ]

            ids = []
            documents = []
            metadatas = []
            vectors = []

            for chunk in batch:

                try:

                    if self.check_duplicate(
                        chunk.chunk_id
                    ):
                        result.skipped += 1
                        continue

                    vector = embeddings.embed_documents(
                        [chunk.chunk_text]
                    )[0]

                    ids.append(chunk.chunk_id)
                    documents.append(chunk.chunk_text)
                    metadatas.append(
                        chunk.metadata.to_dict()
                    )
                    vectors.append(vector)

                except Exception as exc:
                    result.errors.append(
                        f"{chunk.chunk_id}: {exc}"
                    )

            if ids:

                try:

                    self._collection.upsert(
                        ids=ids,
                        documents=documents,
                        metadatas=metadatas,
                        embeddings=vectors,
                    )

                    result.ingested += len(ids)

                    for chunk_id in ids:
                        if chunk_id not in result.document_ids:
                            result.document_ids.append(
                                chunk_id
                            )

                except Exception as exc:

                    result.errors.append(
                        f"Batch ingestion error: {exc}"
                    )

        logger.info(
            f"Ingestion complete: "
            f"ingested={result.ingested}, "
            f"skipped={result.skipped}, "
            f"errors={len(result.errors)}"
        )

        return result

    def _get_embeddings(self):
        """Create the configured embedding model."""

        from rag_agent.config import EmbeddingFactory

        return EmbeddingFactory(
            self.settings
        ).create()

    def query(
        self,
        query_text: str,
        topic_filter: str | None = None,
        difficulty_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks from ChromaDB."""

        if not query_text.strip():
            return []

        embeddings = self._get_embeddings()

        query_embedding = embeddings.embed_query(
            query_text
        )

        where_conditions = []

        if topic_filter:
            where_conditions.append(
                {
                    "topic": topic_filter.lower()
                }
            )

        if difficulty_filter:
            where_conditions.append(
                {
                    "difficulty": difficulty_filter.lower()
                }
            )

        where: dict[str, Any] | None = None

        if len(where_conditions) == 1:
            where = where_conditions[0]

        elif len(where_conditions) > 1:
            where = {
                "$and": where_conditions
            }

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": self.settings.retrieval_k,
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if where:
            query_kwargs["where"] = where

        results = self._collection.query(
            **query_kwargs
        )

        retrieved: list[RetrievedChunk] = []

        ids = results.get("ids", [[]])[0]
        documents = results.get(
            "documents",
            [[]],
        )[0]
        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]
        distances = results.get(
            "distances",
            [[]],
        )[0]

        for chunk_id, document, metadata_dict, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        ):

            score = 1.0 - float(distance)

            if (
                score
                < self.settings.similarity_threshold
            ):
                continue

            metadata = ChunkMetadata.from_dict(
                metadata_dict
            )

            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=metadata,
                    score=score,
                )
            )

        retrieved.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        return retrieved

    def list_documents(self) -> list[dict[str, Any]]:
        """Return a summary of documents in the collection."""

        results = self._collection.get(
            include=["metadatas"]
        )

        documents: dict[str, int] = {}

        for metadata in results.get(
            "metadatas",
            [],
        ):

            source = metadata.get(
                "source",
                "unknown",
            )

            documents[source] = (
                documents.get(source, 0) + 1
            )

        return [
            {
                "source": source,
                "chunks": count,
            }
            for source, count in sorted(
                documents.items()
            )
        ]

    def get_document_chunks(
        self,
        source: str,
    ) -> list[DocumentChunk]:
        """Return all chunks belonging to a source document."""

        results = self._collection.get(
            where={
                "source": source
            },
            include=[
                "documents",
                "metadatas",
            ],
        )

        chunks = []

        ids = results.get(
            "ids",
            [],
        )

        documents = results.get(
            "documents",
            [],
        )

        metadatas = results.get(
            "metadatas",
            [],
        )

        for chunk_id, document, metadata_dict in zip(
            ids,
            documents,
            metadatas,
        ):

            metadata = ChunkMetadata.from_dict(
                metadata_dict
            )

            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    chunk_text=document,
                    metadata=metadata,
                )
            )

        return chunks

    def get_collection_stats(self) -> dict[str, Any]:
        """Return collection statistics."""

        results = self._collection.get(
            include=["metadatas"]
        )

        metadatas = results.get(
            "metadatas",
            [],
        )

        topics = set()
        sources = set()
        bonus_topics_present = False

        bonus_topics = {
            "som",
            "boltzmannmachine",
            "gan",
        }

        for metadata in metadatas:

            topic = str(
                metadata.get(
                    "topic",
                    "",
                )
            ).lower()

            source = metadata.get(
                "source",
                "",
            )

            if topic:
                topics.add(topic)

            if source:
                sources.add(source)

            if topic in bonus_topics:
                bonus_topics_present = True

        return {
            "total_chunks": len(metadatas),
            "topics": sorted(topics),
            "sources": sorted(sources),
            "bonus_topics_present": bonus_topics_present,
        }

    def delete_document(
        self,
        source: str,
    ) -> int:
        """Delete all chunks from a source document."""

        results = self._collection.get(
            where={
                "source": source
            }
        )

        ids = results.get(
            "ids",
            [],
        )

        if not ids:
            return 0

        self._collection.delete(
            ids=ids
        )

        logger.info(
            f"Deleted {len(ids)} chunks "
            f"from '{source}'"
        )

        return len(ids)