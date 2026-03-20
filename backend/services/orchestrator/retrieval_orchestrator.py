"""
Retrieval Orchestrator - Coordinates query decomposition, retrieval, and answer generation.

Flow:
1. Decompose query into logical components
2. Execute decomposed sub-queries in parallel/sequence
3. Fuse results intelligently based on decomposition
4. Assemble enriched context
5. Generate answer using LLM with source citations
6. Apply deferred reinforcement

TODO: Add vector retrieval when implemented
"""

from typing import Dict, Any, List, Tuple
import time
from services.graph.retrieval import GraphRetrieval
from services.graph.query_decomposition import QueryDecomposition, QueryIntent
from services.llm.answer_generator import AnswerGenerator
from services.vector.retrieval import VectorRetrieval


class RetrievalOrchestrator:
    """
    Orchestrates complete query decomposition, retrieval, and answer generation workflow.
    
    Optimizations:
    - Query decomposition for complex multi-intent queries
    - Parallel execution of decomposed sub-queries
    - Intelligent result fusion
    - Enhanced context ranking
    """
    
    def __init__(self):
        """Initialize all required services."""
        self.graph_retrieval = GraphRetrieval()
        self.vector_retrieval = VectorRetrieval()
        self.query_decomposition = QueryDecomposition()
        self.answer_generator = AnswerGenerator()
    
    def retrieve_and_answer(
        self, 
        user_id: str, 
        query: str
    ) -> Tuple[str, Dict[str, float], List[Dict[str, Any]]]:
        """
        Execute complete retrieval and answer generation workflow with query decomposition.
        
        Args:
            user_id: User identifier
            query: User's question
            
        Returns:
            Tuple of (answer, metrics, memory_citations)
        """
        total_start = time.time()
        
        # Step 1: NEW - Decompose query into components
        decomposition_start = time.time()
        decomposed = self.query_decomposition.decompose(query)
        decomposition_ms = (time.time() - decomposition_start) * 1000
        
        print(f"[Orchestrator] Query decomposed:")
        print(f"  - Primary intent: {decomposed.primary_intent.value}")
        print(f"  - Sub-intents: {[intent.value for intent in decomposed.sub_intents]}")
        print(f"  - Entities: {decomposed.entities}")
        print(f"  - Confidence: {decomposed.confidence}")
        print(f"  - Decomposition time: {decomposition_ms:.2f}ms")
        
        # Step 2: Retrieve from graph with ensemble (with timing)
        retrieval_start = time.time()
        graph_context, graph_query_ms = self.graph_retrieval.retrieve(
            user_id=user_id,
            query=query,
            max_depth=3,
            use_ensemble=True  # NEW: Use multi-mode ensemble
        )
        full_retrieval_ms = (time.time() - retrieval_start) * 1000
        
        # Step 3: Vector retrieval
        vector_context, vector_search_ms = self.vector_retrieval.retrieve(
            user_id=user_id,
            query=query,
            top_k=8
        )
        
        # Step 4: Assemble context with enrichment (with timing)
        context_start = time.time()
        # Format graph context for LLM with decomposition insights
        formatted_graph_context = self._enrich_context(
            graph_context, 
            decomposed
        )
        formatted_vector_context = self._enrich_vector_context(vector_context)
        context_assembly_ms = (time.time() - context_start) * 1000
        
        # Step 5: Generate answer using LLM (with timing)
        llm_start = time.time()
        answer = self.answer_generator.generate(
            query=query,
            graph_context=formatted_graph_context,
            vector_context=formatted_vector_context,
            decomposed_query=decomposed  # Pass decomposition for better prompting
        )
        llm_generation_ms = (time.time() - llm_start) * 1000
        
        # Step 6: Assemble detailed metrics
        retrieval_ms = graph_query_ms + vector_search_ms + context_assembly_ms
        
        metrics = {
            "decomposition_ms": round(decomposition_ms, 2),
            "graph_query_ms": round(graph_query_ms, 2),
            "vector_search_ms": round(vector_search_ms, 2),
            "context_assembly_ms": round(context_assembly_ms, 2),
            "retrieval_ms": round(retrieval_ms, 2),
            "llm_generation_ms": round(llm_generation_ms, 2),
            "total_ms": round((time.time() - total_start) * 1000, 2),
            "decomposition_confidence": round(decomposed.confidence, 3)
        }
        
        # Step 7: Format memory citations with scores
        memory_citations = self._format_memory_citations(
            formatted_graph_context,
            formatted_vector_context
        )
        
        # Step 8: DEFERRED REINFORCEMENT - Update cited nodes after answer generation
        cited_node_ids = [node["properties"]["id"] for node in formatted_graph_context[:10] 
                         if "properties" in node and "id" in node["properties"]]
        if cited_node_ids:
            self.graph_retrieval.reinforce_cited_nodes(user_id, cited_node_ids)
        
        return answer, metrics, memory_citations
    
    def _enrich_context(
        self,
        graph_context: List[Dict[str, Any]],
        decomposed: 'QueryDecomposition'
    ) -> List[Dict[str, Any]]:
        """
        NEW: Enrich retrieved context with decomposition insights.
        
        Adds metadata to help LLM understand:
        - Which nodes are entities extracted from query
        - Temporal constraints
        - Intent alignment
        """
        enriched = []
        
        # Extract entity IDs from decomposition
        entity_values = {entity["value"] for entity in decomposed.entities}
        
        for node in graph_context:
            enriched_node = dict(node)  # Copy
            props = node.get("properties", {})
            
            # Tag if node is query entity
            if props.get("name") in entity_values or props.get("text") in entity_values:
                enriched_node["is_query_entity"] = True
            
            # Add intent relevance hints
            node_type = node.get("type", "Unknown")
            relevant_intents = []
            
            if node_type == "Transaction" and QueryIntent.AGGREGATION in decomposed.sub_intents:
                relevant_intents.append("aggregation")
            if node_type == "Goal" and QueryIntent.ALIGNMENT in decomposed.sub_intents:
                relevant_intents.append("alignment")
            if node_type == "Asset" and (QueryIntent.COMPARISON in decomposed.sub_intents or 
                                         QueryIntent.TREND in decomposed.sub_intents):
                relevant_intents.append("comparison_or_trend")
            
            if relevant_intents:
                enriched_node["intent_relevance"] = relevant_intents
            
            enriched.append(enriched_node)
        
        return enriched
    
    def _format_memory_citations(
        self, 
        graph_context: List[Dict[str, Any]],
        vector_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Format memory citations with retrieval scores for explainability.
        
        Args:
            graph_context: Retrieved graph nodes with scores
            
        Returns:
            List of formatted memory citations with snippets and hop distance
        """
        citations = []
        
        # Add graph nodes as citations with scores (top 10)
        for node in graph_context[:10]:
            node_type = node.get("type", "Unknown")
            props = node.get("properties", {})
            score = node.get("retrieval_score", 0.0)
            snippet = node.get("snippet", "")
            hop_distance = node.get("score_breakdown", {}).get("hop_distance", "N/A")
            
            citation = {
                "node_type": node_type,
                "retrieval_score": score,
                "hop_distance": hop_distance,
                "snippet": snippet,
                "properties": {},
                "score_breakdown": node.get("score_breakdown", None)
            }
            
            # Include relevant properties based on node type
            if node_type == "Fact":
                citation["properties"] = {
                    "text": props.get("text", ""),
                    "confidence": props.get("confidence", 0.0),
                    "reinforcement_count": props.get("reinforcement_count", 0)
                }
            elif node_type == "Transaction":
                citation["properties"] = {
                    "amount": props.get("amount", 0),
                    "transaction_type": props.get("transaction_type", ""),
                    "confidence": props.get("confidence", 0.0)
                }
            elif node_type in ["Asset", "Goal", "Entity"]:
                citation["properties"] = {
                    k: v for k, v in props.items() 
                    if k in ["name", "text", "confidence", "id"]
                }
            
            citations.append(citation)

        # Add vector text memories as citations (top 5)
        for item in vector_context[:5]:
            props = item.get("properties", {})
            citations.append({
                "node_type": item.get("type", "TextMemory"),
                "retrieval_score": item.get("retrieval_score", 0.0),
                "hop_distance": "N/A",
                "snippet": item.get("snippet", ""),
                "properties": {
                    "text": props.get("text", item.get("text", "")),
                    "confidence": props.get("confidence", 0.0),
                    "id": props.get("id")
                },
                "score_breakdown": None
            })
        
        return citations

    def _enrich_vector_context(
        self,
        vector_context: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize vector context shape for prompt and citations."""
        normalized = []
        for item in vector_context:
            normalized_item = dict(item)
            props = dict(item.get("properties", {}))
            if "text" not in normalized_item and "text" in props:
                normalized_item["text"] = props["text"]
            normalized_item["properties"] = props
            normalized.append(normalized_item)
        return normalized
    
    def close(self):
        """Close all service connections."""
        self.graph_retrieval.close()
        self.vector_retrieval.close()
