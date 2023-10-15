import os
from inspect import getmembers, isclass, isfunction
import pinecone
from typing import Any, Dict, List, Union
from unittest.mock import MagicMock
from unittest.mock import patch
import numpy as np
import pytest
from tests import pinecone_mock
from pinecone_haystack.document_store import PineconeDocumentStore
from pinecone_haystack.retriever import PineconeRetriever
from haystack.preview import (
    component,
    Document,
    default_to_dict,
    default_from_dict,
    DeserializationError,
)

from haystack.preview.document import Document

class TestPineconeRetriever:
    @pytest.mark.integration
    def test_init(self):
        document_store = PineconeDocumentStore("pinecone-test-key")
        retriever = PineconeRetriever(document_store=document_store)
        assert retriever.document_store == document_store
        assert retriever.filters == None
        assert retriever.top_k == 10
        assert retriever.scale_score == True
        assert retriever.return_embedding == False
        
    @pytest.mark.integration
    def test_run(self):
        document_store = PineconeDocumentStore("pinecone-test-key")
        with patch.object(self.document_store, "query") as mock_query:
            mock_query.return_value = Document(
                content="$TSLA lots of green on the 5 min, watch the hourly $259.33 possible resistance currently @ $257.00.Tesla is recalling 2,700 Model X cars.Hard to find new buyers of $TSLA at 250. Shorts continue to pile in.",
                metadata4={
                    "target": "TSLA",
                    "sentiment_score": 0.318,
                    "format": "post",
                })

            results = self.retriever.run(["How many cars is TSLA recalling?"])
        
            assert len(results["documents"]) == 1
            assert results["documents"][0][0].content == "$TSLA lots of green on the 5 min, watch the hourly $259.33 possible resistance currently @ $257.00.Tesla is recalling 2,700 Model X cars.Hard to find new buyers of $TSLA at 250. Shorts continue to pile in."
        
    @pytest.mark.integration
    def test_to_dict(self):
        document_store = PineconeDocumentStore("pinecone-test-key")
        retriever = PineconeRetriever(document_store=document_store)
        doc_dict = retriever.to_dict()
        assert doc_dict == {
            "init_parameters": {
                "document_store": "test_document_store",
                "filters": None,
                "top_k": 10,
                "use_async_client": False,
                "max_retries": 3,
                "timeout": 120,
            }
        }
        
    @pytest.mark.integration
    def test_from_dict(self):
        """
        Test deserialization of this component from a dictionary, using default initialization parameters.
        """
        retriever_component_dict = {
            "type": "PineconeRetriever",
            "init_parameters": {
                "document_store": "test_document_store",
                "type":"PineconeRetriever",
                "filters": None,
                "top_k": 10,
                "scale_score": True,
                "return_embedding": False,
            }
        }
        retriever = PineconeRetriever.from_dict(retriever_component_dict)

        assert retriever.document_store == "test_document_store"
        assert retriever.filters is None
        assert retriever.top_k == 10
        assert retriever.scale_score is True
        assert retriever.return_embedding is False

