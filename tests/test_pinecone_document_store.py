import os
from inspect import getmembers, isclass, isfunction
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock
import numpy as np
import pytest

import src.pinecone_haystack
from pinecone_haystack.document_store import PineconeDocumentStore
from pinecone_haystack.errors import (
    PineconeDocumentStoreError,
    PineconeDocumentStoreFilterError,
)
from haystack.preview.dataclasses import Document
from haystack.preview.testing.document_store import DocumentStoreBaseTests
from tests import pinecone_mock
import pinecone

class TestPineconeDocumentStore:
    @pytest.fixture
    def ds(self, monkeypatch, request) -> PineconeDocumentStore:
        """
        This fixture provides an empty document store and takes care of cleaning up after each test
        """
    
        for fname, function in getmembers(pinecone_mock, isfunction):
            monkeypatch.setattr(f"pinecone.{fname}", function, raising=False)
        for cname, class_ in getmembers(pinecone_mock, isclass):
            monkeypatch.setattr(f"pinecone.{cname}", class_, raising=False)

        return PineconeDocumentStore(
            api_key=os.environ.get("PINECONE_API_KEY") or "pinecone-test-key",
            embedding_dim=768,
            embedding_field="embedding",
            index="haystack_tests",
            similarity="cosine",
            recreate_index=True,
        )

    @pytest.fixture
    def doc_store_with_docs(self, ds: PineconeDocumentStore, documents: List[Document]) -> PineconeDocumentStore:
        """
        This fixture provides a pre-populated document store and takes care of cleaning up after each test
        """
        ds.write_documents(documents)
        return ds

    @pytest.fixture
    def mocked_ds(self):
        class DSMock(PineconeDocumentStore):
            pass

        pinecone.init = MagicMock()
        DSMock._create_index = MagicMock()
        mocked_ds = DSMock(api_key="MOCK")

        return mocked_ds

    def docs_all_formats(self) -> List[Union[Document, Dict[str, Any]]]:
        return [
            # Document object
            Document(
                text="Lloyds to cut 945 jobs as part of 3-year restructuring plan, Last month we added to our $GILD position and started a new one in $BWLD We see slow, steady, unspectacular growth going forward near term. Lloyds Banking Group's share price lifts amid reports bank is poised to axe hundreds of UK jobs",
                metadata={
                    "target": "Lloyds",
                    "sentiment_score": -0.532,
                    "format": "headline",
                },
            ),
            Document(
                text="FTSE 100 drops 2.5 pct on Glencore, metals price fears. Glencore sees Tripoli-based NOC as sole legal seller of Libyan oil. Glencore Studies Possible IPO for Agricultural Trading Business. Glencore chief blames rivals' overproduction for share price fall.",
                metadata={
                    "target": "Glencore",
                    "sentiment_score": 0.037,
                    "format": "headline",
                },
            ),
            Document(
                text="Shell's $70 Billion BG Deal Meets Shareholder Skepticism. Shell and BG Shareholders to Vote on Deal at End of January. EU drops Shell, BP, Statoil from ethanol benchmark investigation. Shell challenges Exxon dominance with 47 billion-pound bid for BG",
                metadata={
                    "target": "Shell",
                    "sentiment_score": -0.345,
                    "format": "headline",
                },
            ),
            Document(
                text="$TSLA lots of green on the 5 min, watch the hourly $259.33 possible resistance currently @ $257.00.Tesla is recalling 2,700 Model X cars.Hard to find new buyers of $TSLA at 250. Shorts continue to pile in.",
                metadata={
                    "target": "TSLA",
                    "sentiment_score": 0.318,
                    "format": "post",
                },
            ),
            Document(
                text="HSBC appoints business leaders to board. HSBC Says Unit to Book $585 Million Charge on Settlement. HSBC Hit by Fresh Details of Tax Evasion Claims. HSBC Hit by Fresh Details of Tax Evasion Claims. Goldman Sachs, Barclays, HSBC downplay Brexit threat.",
                metadata={
                    "target": "HSBC",
                    "sentiment_score": 0.154,
                    "format": "post",
                },
            ),
            # Without meta
            Document(
                text="Aspen to Buy Anaesthetics From AstraZeneca for $520 Million. AstraZeneca wins FDA approval for key new lung cancer pill. AstraZeneca boosts respiratory unit with $575 mln Takeda deal. AstraZeneca Acquires ZS Pharma in $2.7 Billion Deal."
            ),
            Document(
                text="Anheuser-Busch InBev Increases Offer for Rival SABMiller. Australia clears AB Inbev's $100 billion SABMiller buyout plan.Australia clears AB Inbev's $100 billion SABMiller buyout plan."
            ),
            Document(
                text="The Coca-Cola Company and Coca-Cola FEMSA to Acquire AdeS Soy-Based Beverage Business From Unilever."
            ),
        ]
    
    @pytest.mark.integration
    def test_ne_filters(self, ds, documents):
        ds.write_documents(documents)

        result = ds.get_all_documents(filters={"format": {"$ne": "headline"}})
        assert len(result) == 2    

    @pytest.mark.integration
    def test_filter_documents_with_filters(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Test that the `filter_documents()` function can filter the documents retrieved using filters.
        """
        eq_docs = doc_store_with_docs.filter_documents(filters={"type": {"$eq": "article"}})
        documents = document_store.filter_documents(filters=filters)
        normal_docs = doc_store_with_docs.filter_documents(filters={"type": {"$eq": "article"}})
        assert eq_docs == normal_docs

    @pytest.mark.integration
    def test_filter_documents_with_batch_size(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Test that the `write_documents()` function can retrieve the documents in batches.
        """

        batch_size = 10
        documents = document_store.write_documents(batch_size=batch_size)

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            assert len(batch) == len(batch_size)

    @pytest.mark.integration
    def test_filter_documents_ids_extended_filter_ne(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.filter_documents(filters={"target": {"$ne": "Glencore"}})
        assert all(d.meta.get("metadata", None) != "Glencore" for d in retrieved_docs)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_nin(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.filter_documents(filters={"format": {"$nin": ["target", "post"]}})
        assert {"target", "post"}.isdisjoint({d.metadata.get("metadata", None) for d in retrieved_docs})

    @pytest.mark.integration
    def test_filter_documents_extended_filter_gt(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.filter_documents(filters={"sentiment_score": {"$gt": 3.0}})
        assert all(d.metadata["sentiment_score"] > 3.0 for d in retrieved_docs)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_gte(self, doc_store_with_docs: PineconeDocumentStore):
        retrieved_docs = doc_store_with_docs.filter_documents(filters={"sentiment_score": {"$gte": 3.0}})
        assert all(d.metadata["sentiment_score"] >= 3.0 for d in retrieved_docs)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_compound_and_other_field_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "sentiment_score": {"$lte": 0.2, "$gte": 0.3},
            "target": ["Shell", "Glencore", "HSBC", "Lloyds", "TSLA"],
        }

        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]te' operation must be a float or int.",
        ):
            doc_store_with_docs.filter_documents(filters=filters_simplified)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_compound_and_or_explicit(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters = {
            "$and": {
                "sentiment_score": {"$lte": 0.2, "$gte": 0.3},
                "target": {
                    "name": {"$in": ["HSBC", "Lloyds"]},
                    "sentiment_score": {"$lte": 5.0},
                },
            }
        }

        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]te' operation must be a float or int.",
        ):
            doc_store_with_docs.filter_documents(filters=filters)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_and_or_simplified(self, doc_store_with_docs: PineconeDocumentStore):
        filters_simplified = {
            "sentiment_score": {"$lte": 0.2, "$gte": 0.3},
            "$or": {"format": ["headline", "post"], "sentiment_score": {"0.318"}},
        }

        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]te' operation must be a float or int.",
        ):
            doc_store_with_docs.filter_documents(filters=filters_simplified)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_and_or_and_not_explicit(self, doc_store_with_docs: PineconeDocumentStore):
        filters = {
            "$and": {
                "sentiment_score": {"$gte": 0.037},
                "$or": {
                    "target": {"$in": ["LLyods", "Glencore", "HSBC", "TSLA", "Shell"]},
                    "$and": {"format": {"$in": ["headline", "post"]}},
                },
            }
        }
        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]te' operation must be a float or int.",
        ):
            doc_store_with_docs.filter_documents(filters=filters)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_and_or_and_not_simplified(
        self, doc_store_with_docs: PineconeDocumentStore
    ):
        filters_simplified = {
            "sentiment_score": {"$lte": "0.037"},
            "$or": {
                "target": ["LLyods", "Glencore"],
                "$and": {"format": {"$lte": "headline"}, "$not": {"format": "post"}},
            },
        }
        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]te' operation must be a float or int.",
        ):
            doc_store_with_docs.filter_documents(filters=filters_simplified)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_compound_nested_not(self, doc_store_with_docs: PineconeDocumentStore):
        # Test nested logical operations within "$not".
        filters = {
            "$not": {
                "$or": {
                    "$and": {"target": {"Lloyds"}},
                    "$not": {"format": {"healdine"}},
                }
            }
        }
        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]t' operation must be a float or int.",
        ):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_filter_documents_extended_filter_compound_same_level_not(self, doc_store_with_docs: PineconeDocumentStore):
        # Test same logical operator twice on the same level.
        filters = {
            "$or": [
                {
                    "$and": {
                        "target": ["LLyods", "Glencore", "TSLA", "Shell"],
                        "format": {"$in": ["post"]},
                    }
                },
                {
                    "$and": {
                        "target": ["LLyods", "Glencore", "HSBC", "TSLA", "Shell"],
                        "format": {"$in": ["headline"]},
                    }
                },
            ]
        }

        with pytest.raises(
            FilterError,
            match=r"Comparison value for '\$[l|g]te' operation must be a float or int.",
        ):
            doc_store_with_docs.get_all_documents(filters=filters)

    @pytest.mark.integration
    def test_multilayer_dict(self, doc_store_with_docs: PineconeDocumentStore):

        multilayer_meta = {
            "$and": {"target": {"Glencore": {"format": "headline", "format": "post"}}},
            "metadata": "multilayer-test",
        }
        doc = Document(
            text="Multilayered dict",
            meta=multilayer_meta,
            embedding=np.random.rand(768).astype(np.float32),
        )

        doc_store_with_docs.write_documents([doc])
        retrieved_docs = doc_store_with_docs.filter_documents(filters={"metadata": {"$eq": "multilayer-test"}})

        assert len(retrieved_docs) == 1
        assert retrieved_docs[0].meta == multilayer_meta

    def test_get_embedding_count(self, doc_store_with_docs: PineconeDocumentStore):
        """
        We expect 1 doc with an embeddings because all documents in already written in doc_store_with_docs contain no
        embeddings.
        """
        doc = Document(
            text="Doc with embedding",
            embedding=np.random.rand(768).astype(np.float32),
        )
        doc_store_with_docs.write_documents([doc])
        assert doc_store_with_docs.get_embedding_count() == 1

        # Test the count of document in the index

    @pytest.mark.integration
    def test_get_document_count_after_write_doc_with_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        with an embedding is written to the document store.
        """
        # there are 5 docs in doc_store_with_docs (all without embeddings)
        initial_document_count = 5

        # we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and also initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # document with embedding is written to doc_store_with_docs
        doc = Document(
            text="Doc with embedding",
            embedding=np.random.rand(768).astype(np.float32),
        )
        doc_store_with_docs.write_documents([doc])

        # so we expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # but we expect initial_document_count documents without embeddings to be unchanged
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count

    @pytest.mark.integration
    def test_get_document_count_after_write_doc_without_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        without an embedding is written to the document store.
        """
        initial_document_count = 5

        # we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and we also expect initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # document without embedding is written to doc_store_with_docs
        doc = Document(text="Doc without embedding")
        doc_store_with_docs.write_documents([doc])

        # we now expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # And we also expect initial_document_count + 1 documents without embeddings, because the document we just
        # wrote has no embeddings
        assert (
            doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count + 1
        )

    @pytest.mark.integration
    def test_get_document_count_after_delete_doc_with_embedding(self, doc_store_with_docs: PineconeDocumentStore):
        """
        Tests that get_document_count() returns the correct number of documents in the document store after a document
        with an embedding is deleted from the document store.
        """
        initial_document_count = 5

        # we expect initial_document_count documents without embeddings in doc_store_with_docs
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count
        # and also initial_document_count documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count

        # two documents with embedding are written to doc_store_with_docs
        doc_1 = Document(
            text="Doc with embedding 1",
            embedding=np.random.rand(768).astype(np.float32),
        )
        doc_2 = Document(
            text="Doc with embedding 2",
            embedding=np.random.rand(768).astype(np.float32),
        )
        doc_store_with_docs.write_documents([doc_1, doc_2])

        # total number is initial_document_count + 2
        assert doc_store_with_docs.get_document_count() == initial_document_count + 2

        # remove one of the documents with embedding
        all_embedding_docs = doc_store_with_docs.get_all_documents(type_metadata=DOCUMENT_WITH_EMBEDDING)

        doc_store_with_docs.delete_documents(ids=[all_embedding_docs[0].id])

        # since we deleted one doc, we expect initial_document_count + 1 documents in total
        assert doc_store_with_docs.get_document_count() == initial_document_count + 1

        # and we expect initial_document_count documents without embeddings
        assert doc_store_with_docs.get_document_count(only_documents_without_embedding=True) == initial_document_count

    #@pytest.mark.unit
    def test_split_overlap_meta(self, mocked_ds):
        """
        Tests that we can upload Docs with a _split_overlap_meta field to Pinecone as a JSON string
        and that the field is parsed correctly as dictionary when retrieved.
        """
        doc = Document(content="test", meta={"_split_overlap": [{"doc_id": "test_id", "range": (0, 10)}]}, id="test_id")
        # Test writing as JSON string
        mocked_ds.write_documents([doc])
        call_args = mocked_ds.pinecone_indexes["document"].upsert.call_args.kwargs
        assert list(call_args["vectors"])[0][2] == {
            "doc_type": "no-vector",
            "content": "test",
            "content_type": "text",
            "_split_overlap": '[{"doc_id": "test_id", "range": [0, 10]}]',
        }
        # Test retrieving as dict
        mocked_ds._get_all_document_ids = MagicMock(return_value=["test_id"])
        mocked_ds.pinecone_indexes["document"].fetch.return_value = {
            "vectors": {
                "test_id": {
                    "metadata": {
                        "_split_overlap": '[{"doc_id": "test_id", "range": [0, 10]}]',
                        "content": "test",
                        "content_type": "text",
                    }
                }
            }
        }
        retrieved_docs = mocked_ds.get_all_documents()
        assert retrieved_docs[0].meta["_split_overlap"] == [{"doc_id": "test_id", "range": [0, 10]}]
