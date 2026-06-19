import unittest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from backend.crag import grade_documents_node, rewrite_query_node
from backend.srag import check_hallucination_srag, check_answer_srag

class TestGraders(unittest.TestCase):
    
    @patch('backend.crag.llm')
    def test_grade_documents_relevant(self, mock_llm):
        # Mock LLM to return "yes" (relevant)
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "yes"
        mock_llm.invoke.return_value = mock_response
        
        state = {
            "messages": [],
            "documents": [Document(page_content="This is relevant content.")],
            "query": "relevant query",
            "web_search_needed": False
        }
        
        result = grade_documents_node(state)
        self.assertFalse(result["web_search_needed"])
        self.assertEqual(len(result["documents"]), 1)

    @patch('backend.crag.llm')
    def test_grade_documents_irrelevant(self, mock_llm):
        # Mock LLM to return "no" (irrelevant)
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "no"
        mock_llm.invoke.return_value = mock_response
        
        state = {
            "messages": [],
            "documents": [Document(page_content="This is spam content.")],
            "query": "specific query",
            "web_search_needed": False
        }
        
        result = grade_documents_node(state)
        self.assertTrue(result["web_search_needed"])
        self.assertEqual(len(result["documents"]), 0)

    @patch('backend.crag.llm')
    def test_rewrite_query(self, mock_llm):
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "optimized search phrase"
        mock_llm.invoke.return_value = mock_response
        
        state = {"query": "bad query term"}
        result = rewrite_query_node(state)
        self.assertEqual(result["query"], "optimized search phrase")

    @patch('backend.srag.llm')
    def test_check_hallucination_grounded(self, mock_llm):
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "yes"
        mock_llm.invoke.return_value = mock_response
        
        docs = [Document(page_content="Grounded facts")]
        result = check_hallucination_srag(docs, "Grounded facts statement")
        self.assertEqual(result, "yes")

    @patch('backend.srag.llm')
    def test_check_hallucination_ungrounded(self, mock_llm):
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "no"
        mock_llm.invoke.return_value = mock_response
        
        docs = [Document(page_content="Grounded facts")]
        result = check_hallucination_srag(docs, "Completely made up statement")
        self.assertEqual(result, "no")

    @patch('backend.srag.llm')
    def test_check_answer_useful(self, mock_llm):
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "yes"
        mock_llm.invoke.return_value = mock_response
        
        result = check_answer_srag("What is 2+2?", "The answer is 4")
        self.assertEqual(result, "yes")
        
    @patch('backend.srag.llm')
    def test_check_answer_useless(self, mock_llm):
        mock_response = MagicMock(spec=AIMessage)
        mock_response.content = "no"
        mock_llm.invoke.return_value = mock_response
        
        result = check_answer_srag("What is 2+2?", "Today is Friday")
        self.assertEqual(result, "no")
