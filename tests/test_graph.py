import unittest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from backend.app import chatbot

class TestGraphIntegration(unittest.TestCase):

    @patch('backend.crag.llm')
    @patch('backend.srag.llm')
    @patch('backend.app.llm_with_tools')
    @patch('backend.app.thread_has_document')
    def test_crag_web_search_fallback_trigger(self, mock_has_doc, mock_llm_tools, mock_srag_llm, mock_crag_llm):
        """
        Verify that if documents are graded irrelevant (CRAG),
        the graph cycles through rewrite_query and web_search nodes.
        """
        mock_has_doc.return_value = True
        
        # 1. Mock CRAG relevance grader to say "no" (irrelevant)
        mock_crag_response = MagicMock(spec=AIMessage)
        mock_crag_response.content = "no"
        mock_crag_llm.invoke.return_value = mock_crag_response
        
        # 2. Mock generation LLM to return standard text
        mock_gen_response = MagicMock(spec=AIMessage)
        mock_gen_response.content = "Grounded response from web context"
        mock_srag_llm.invoke.return_value = mock_gen_response
        
        # 3. Patch web search tool invoke inside web_search_node
        with patch('backend.crag.web_search') as mock_search:
            mock_search.invoke.return_value = "Search result snippets about ChatSphere."
            
            # Setup initial state config
            config = {"configurable": {"thread_id": "integration_test_crag_thread"}}
            initial_state = {
                "messages": [HumanMessage(content="What is ChatSphere built on?")],
                "documents": [Document(page_content="Unrelated text about apples.")],
                "query": "What is ChatSphere built on?",
                "web_search_needed": False,
                "loop_count": 0
            }
            
            # Invoke Graph
            res = chatbot.invoke(initial_state, config=config)
            
            # Verify web search was invoked because the document was irrelevant
            mock_search.invoke.assert_called_once()
            self.assertEqual(res["messages"][-1].content, "Grounded response from web context")

    @patch('backend.crag.llm')
    @patch('backend.srag.llm')
    @patch('backend.app.thread_has_document')
    def test_self_rag_loop_cap_guard(self, mock_has_doc, mock_srag_llm, mock_crag_llm):
        """
        Verify that if generated output is hallucinated, the graph loops,
        but safely terminates at loop_count == 3 to prevent infinite loops.
        """
        mock_has_doc.return_value = True
        
        # 1. Mock document grading to be relevant (no web search fallback needed)
        mock_crag_response = MagicMock(spec=AIMessage)
        mock_crag_response.content = "yes"
        mock_crag_llm.invoke.return_value = mock_crag_response
        
        # 2. Mock srag evaluations: Always return "no" (hallucinated) to force loops
        mock_hallucination_response = MagicMock(spec=AIMessage)
        mock_hallucination_response.content = "no" # not grounded
        mock_srag_llm.invoke.return_value = mock_hallucination_response
        
        config = {"configurable": {"thread_id": "integration_test_loop_thread"}}
        initial_state = {
            "messages": [HumanMessage(content="Query")],
            "documents": [Document(page_content="Valid document text")],
            "query": "Query",
            "web_search_needed": False,
            "loop_count": 0
        }
        
        # Run graph
        res = chatbot.invoke(initial_state, config=config)
        
        # Verify loop count capped at 3 and safely returned
        self.assertEqual(res["loop_count"], 3)
