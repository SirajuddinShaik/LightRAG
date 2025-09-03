"""
FastAPI backend for LightRAG querying pipeline with visual hopping
"""
import json
import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
from dataclasses import dataclass
import google.generativeai as genai
import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LightRAG Query Pipeline")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedding model
embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')

# Gemini Provider Class (same as test_hierarchical_grouping_gemini.py)
class GeminiProvider:
    def __init__(self, api_key: str, model_id: str, temperature: float = 0.1, response_mime_type: str = None, max_tokens: int = 8192, system_prompt: str = None):
        if not api_key:
            raise ValueError("Gemini API key is required.")
        
        genai.configure(api_key=api_key)
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        if response_mime_type:
            generation_config.response_mime_type = response_mime_type
            
        self.model = genai.GenerativeModel(
            model_name=model_id,
            generation_config=generation_config,
            system_instruction=system_prompt
        )
        # Semaphore to limit concurrency to 30 requests at a time
        self.semaphore = asyncio.Semaphore(30)

    async def generate_content(self, prompt):
        async with self.semaphore:
            retries = 1000
            for i in range(retries):
                try:
                    response = await self.model.generate_content_async(prompt)
                    return response.text
                except Exception as e:
                    print(f"Error during Gemini generation: {e}. Retrying in 30 seconds...")
                    await asyncio.sleep(30)
            
            print(f"Failed to generate content from Gemini for model {self.model.model_name} after {retries} retries.")
            return None

# Global Gemini provider instance
_gemini_provider = None

def get_gemini_provider():
    global _gemini_provider
    if _gemini_provider is None:
        print("Initializing Gemini Provider...")
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        _gemini_provider = GeminiProvider(
            api_key=gemini_api_key,
            model_id="models/gemini-2.5-pro",  # Using latest model
            temperature=0.1,
            max_tokens=15000,
            system_prompt=None
        )
        print("Gemini Provider initialized successfully!")
    return _gemini_provider

@dataclass
class Node:
    id: str
    entity_type: str
    description: str
    embedding: Optional[np.ndarray] = None

@dataclass
class Edge:
    source: str
    target: str
    description: str
    keywords: str
    weight: float
    embedding: Optional[np.ndarray] = None

class GraphData:
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.graph = nx.Graph()  # Undirected graph for bidirectional traversal
        self.directed_graph = nx.DiGraph()  # Directed graph for unidirectional traversal
        self.chunks_data = {}
        self.entities_data = {}
        
    def load_from_files(self, folder_path: str):
        """Load graph data from LightRAG folder"""
        folder = Path(folder_path)
        
        # Load GraphML
        graphml_path = folder / "graph_chunk_entity_relation.graphml"
        if graphml_path.exists():
            self._load_graphml(str(graphml_path))
        
        # Load vector databases
        vdb_chunks_path = folder / "vdb_chunks.json"
        if vdb_chunks_path.exists():
            with open(vdb_chunks_path, 'r') as f:
                self.chunks_data = json.load(f)
        
        vdb_entities_path = folder / "vdb_entities.json"
        if vdb_entities_path.exists():
            with open(vdb_entities_path, 'r') as f:
                self.entities_data = json.load(f)
                
        # Compute embeddings
        self._compute_embeddings()
    
    def _load_graphml(self, path: str):
        """Load GraphML file and extract nodes/edges"""
        tree = ET.parse(path)
        root = tree.getroot()
        
        # Parse namespace
        ns = {'graphml': 'http://graphml.graphdrawing.org/xmlns'}
        
        # Load nodes
        for node in root.findall('.//graphml:node', ns):
            node_id = node.get('id')
            entity_type = ""
            description = ""
            
            for data in node.findall('graphml:data', ns):
                key = data.get('key')
                if key == 'd1':  # entity_type
                    entity_type = data.text or ""
                elif key == 'd2':  # description
                    description = data.text or ""
            
            self.nodes[node_id] = Node(
                id=node_id,
                entity_type=entity_type,
                description=description
            )
        
        # Load edges
        for edge in root.findall('.//graphml:edge', ns):
            source = edge.get('source')
            target = edge.get('target')
            description = ""
            keywords = ""
            weight = 1.0
            
            for data in edge.findall('graphml:data', ns):
                key = data.get('key')
                if key == 'd7':  # description
                    description = data.text or ""
                elif key == 'd8':  # keywords
                    keywords = data.text or ""
                elif key == 'd6':  # weight
                    try:
                        weight = float(data.text or 1.0)
                    except:
                        weight = 1.0
            
            edge_obj = Edge(
                source=source,
                target=target,
                description=description,
                keywords=keywords,
                weight=weight
            )
            
            self.edges.append(edge_obj)
            # Add to both undirected and directed graphs
            self.graph.add_edge(source, target, weight=weight, edge_obj=edge_obj)
            self.directed_graph.add_edge(source, target, weight=weight, edge_obj=edge_obj)
    
    def _compute_embeddings(self, batch_size: int = 10):
        """Compute embeddings for nodes and edges with batching for better performance"""
        # Node embeddings with batching
        node_list = list(self.nodes.values())
        node_texts = [f"{node.entity_type}: {node.description}" for node in node_list]
        
        # Process nodes in batches
        for i in range(0, len(node_texts), batch_size):
            batch_texts = node_texts[i:i + batch_size]
            batch_nodes = node_list[i:i + batch_size]
            
            # Compute embeddings for the batch
            batch_embeddings = embedding_model.encode(batch_texts)
            
            # Assign embeddings to nodes
            for node, embedding in zip(batch_nodes, batch_embeddings):
                node.embedding = embedding
        
        # Edge embeddings with batching
        edge_texts = [f"{edge.keywords}: {edge.description}" for edge in self.edges]
        
        # Process edges in batches
        for i in range(0, len(edge_texts), batch_size):
            batch_texts = edge_texts[i:i + batch_size]
            batch_edges = self.edges[i:i + batch_size]
            
            # Compute embeddings for the batch
            batch_embeddings = embedding_model.encode(batch_texts)
            
            # Assign embeddings to edges
            for edge, embedding in zip(batch_edges, batch_embeddings):
                edge.embedding = embedding

# Global graph data
graph_data = GraphData()

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    folder_path: str
    threshold: float = 0.5
    top_k: int = 5
    max_nodes: int = 100

class HopRequest(BaseModel):
    current_nodes: List[str]
    visited_nodes: List[str]
    context_window: List[str]
    query_embedding: List[float]
    threshold: float = 0.5
    hop_number: int
    max_nodes: int = 100
    direction: str = "bidirectional"  # "bidirectional" or "unidirectional"

class QueryResponse(BaseModel):
    query: str
    top_nodes: List[Dict[str, Any]]
    hop_data: Dict[str, Any]
    context_window: List[str]

class HopResponse(BaseModel):
    next_nodes: List[Dict[str, Any]]
    new_context: List[str]
    hop_complete: bool
    cycle_detected: bool
    similarity_scores: Dict[str, float]
    hop_edges: List[Dict[str, Any]]

class AnswerRequest(BaseModel):
    query: str
    context_window: List[str]

class AnswerResponse(BaseModel):
    answer: str
    sources_used: List[str]

class RewriteRequest(BaseModel):
    query: str
    edge_threshold: int = 30

class RewriteResponse(BaseModel):
    original_query: str
    rewritten_query: str
    high_degree_nodes: List[Dict[str, Any]]
    enhancement_reasoning: str

def extract_keywords(query: str) -> List[str]:
    """Extract keywords from query using simple approach"""
    # Simple keyword extraction - can be enhanced with NLP libraries
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'what', 'how', 'when', 'where', 'who', 'why', 'which', 'that', 'this', 'can', 'could', 'would', 'should', 'will', 'do', 'does', 'did', 'have', 'has', 'had'}
    words = query.lower().split()
    keywords = [word.strip('.,!?;:') for word in words if word.lower() not in stop_words and len(word) > 2]
    return keywords

def find_keyword_matching_nodes(query: str, keywords: List[str], top_k: int = 5) -> List[Tuple[str, float, str]]:
    """Find nodes that match query keywords"""
    keyword_matches = []
    query_lower = query.lower()
    
    for node_id, node in graph_data.nodes.items():
        # Create searchable text from node
        searchable_text = f"{node.id.lower()} {node.entity_type.lower()} {node.description.lower()}"
        
        # Calculate keyword match score
        keyword_score = 0
        matched_keywords = []
        
        # Direct query substring matching (highest weight)
        if query_lower in searchable_text:
            keyword_score += 2.0
            matched_keywords.append("full_query_match")
        
        # Individual keyword matching
        for keyword in keywords:
            if keyword.lower() in searchable_text:
                keyword_score += 1.0
                matched_keywords.append(keyword)
        
        # Entity name partial matching
        if any(keyword.lower() in node.id.lower() for keyword in keywords):
            keyword_score += 1.5
            matched_keywords.append("entity_name_match")
        
        # Normalize score by text length to avoid bias toward longer descriptions
        text_length_factor = min(len(searchable_text) / 200, 2.0)  # Cap at 2x
        normalized_score = keyword_score / text_length_factor if text_length_factor > 0 else keyword_score
        
        if keyword_score > 0:
            keyword_matches.append((node_id, normalized_score, f"Keywords: {', '.join(matched_keywords)}"))
    
    # Sort by keyword score and return top-k
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
    return keyword_matches[:top_k]

def find_embedding_similar_nodes(query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
    """Find top-k most similar nodes to query using embeddings"""
    query_embedding = embedding_model.encode(query)
    
    similarities = []
    for node_id, node in graph_data.nodes.items():
        if node.embedding is not None:
            similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
            similarities.append((node_id, similarity, "Embedding similarity"))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def find_top_nodes_combined(query: str, keyword_count: int = 5, embedding_count: int = 5) -> List[Tuple[str, float, str]]:
    """Find top nodes using both keyword matching and embedding similarity"""
    keywords = extract_keywords(query)
    
    # Get keyword-based matches
    keyword_nodes = find_keyword_matching_nodes(query, keywords, keyword_count)
    
    # Get embedding-based matches
    embedding_nodes = find_embedding_similar_nodes(query, embedding_count)
    
    # Combine results with different scoring strategies
    combined_results = {}
    
    # Add keyword matches (with higher weight for keyword relevance)
    for node_id, keyword_score, match_type in keyword_nodes:
        if node_id not in combined_results:
            combined_results[node_id] = {
                'keyword_score': keyword_score,
                'embedding_score': 0.0,
                'match_types': [match_type],
                'combined_score': keyword_score * 1.2  # Boost keyword matches
            }
        else:
            combined_results[node_id]['keyword_score'] = max(combined_results[node_id]['keyword_score'], keyword_score)
            combined_results[node_id]['match_types'].append(match_type)
    
    # Add embedding matches
    for node_id, embedding_score, match_type in embedding_nodes:
        if node_id not in combined_results:
            combined_results[node_id] = {
                'keyword_score': 0.0,
                'embedding_score': embedding_score,
                'match_types': [match_type],
                'combined_score': embedding_score
            }
        else:
            combined_results[node_id]['embedding_score'] = max(combined_results[node_id]['embedding_score'], embedding_score)
            combined_results[node_id]['match_types'].append(match_type)
    
    # Calculate final combined scores
    for node_id, scores in combined_results.items():
        # Hybrid scoring: keyword + embedding with bonus for both
        keyword_component = scores['keyword_score'] * 0.6
        embedding_component = scores['embedding_score'] * 0.4
        
        # Bonus if node matches both keyword and embedding criteria
        both_match_bonus = 0.3 if scores['keyword_score'] > 0 and scores['embedding_score'] > 0 else 0
        
        scores['combined_score'] = keyword_component + embedding_component + both_match_bonus
    
    # Sort by combined score and format results
    sorted_results = sorted(
        combined_results.items(), 
        key=lambda x: x[1]['combined_score'], 
        reverse=True
    )
    
    # Format as list of tuples with detailed match information
    final_results = []
    for node_id, scores in sorted_results:
        match_details = f"Keyword: {scores['keyword_score']:.3f}, Embedding: {scores['embedding_score']:.3f}, Combined: {scores['combined_score']:.3f}"
        final_results.append((node_id, scores['combined_score'], match_details))
    
    return final_results

def compute_edge_similarity(query_embedding: np.ndarray, edge: Edge) -> float:
    """Compute similarity between query and edge"""
    if edge.embedding is None:
        return 0.0
    return cosine_similarity([query_embedding], [edge.embedding])[0][0]

def get_adjacent_nodes_with_similarity(current_nodes: List[str], query_embedding: np.ndarray, threshold: float, direction: str = "bidirectional") -> Tuple[List[Tuple[str, float, str]], List[Dict[str, Any]]]:
    """Get adjacent nodes with similarity scores above threshold and their connecting edges
    
    Args:
        current_nodes: List of current node IDs to traverse from
        query_embedding: Query embedding for similarity computation
        threshold: Minimum similarity threshold
        direction: "bidirectional" or "unidirectional" traversal mode
    """
    adjacent_with_scores = []
    hop_edges = []
    
    # Choose the appropriate graph based on direction
    selected_graph = graph_data.graph if direction == "bidirectional" else graph_data.directed_graph
    
    for node_id in current_nodes:
        if node_id in selected_graph:
            if direction == "bidirectional":
                # For bidirectional, use neighbors() which gets all connected nodes
                neighbors = list(selected_graph.neighbors(node_id))
            else:
                # For unidirectional, use successors() which gets only outgoing connections
                neighbors = list(selected_graph.successors(node_id))
            
            for neighbor in neighbors:
                # Get edge data - use appropriate graph structure
                if direction == "bidirectional":
                    edge_data = selected_graph[node_id][neighbor]
                else:
                    edge_data = selected_graph[node_id][neighbor]
                
                edge_obj = edge_data.get('edge_obj')
                
                if edge_obj:
                    edge_similarity = compute_edge_similarity(query_embedding, edge_obj)
                    
                    # Check node similarity too
                    neighbor_node = graph_data.nodes.get(neighbor)
                    if neighbor_node and neighbor_node.embedding is not None:
                        node_similarity = cosine_similarity([query_embedding], [neighbor_node.embedding])[0][0]
                        combined_similarity = (edge_similarity + node_similarity) / 2
                        
                        if combined_similarity >= threshold:
                            adjacent_with_scores.append((neighbor, combined_similarity, edge_obj.description))
                            
                            # Add edge information for visualization
                            hop_edges.append({
                                "source": node_id,
                                "target": neighbor,
                                "description": edge_obj.description,
                                "keywords": edge_obj.keywords,
                                "weight": edge_obj.weight,
                                "similarity": combined_similarity,
                                "direction": direction
                            })
    
    # Remove duplicates and sort by similarity
    unique_adjacent = list(set(adjacent_with_scores))
    unique_adjacent.sort(key=lambda x: x[1], reverse=True)
    
    return unique_adjacent, hop_edges

@app.post("/initialize", response_model=dict)
async def initialize_graph(request: dict):
    """Initialize graph data from folder"""
    folder_path = request.get("folder_path")
    if not folder_path:
        raise HTTPException(status_code=400, detail="folder_path is required")
    
    try:
        graph_data.load_from_files(folder_path)
        return {
            "status": "success",
            "nodes_count": len(graph_data.nodes),
            "edges_count": len(graph_data.edges),
            "message": "Graph data loaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading graph data: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process initial query and return top nodes using both keywords and embeddings"""
    try:
        # Extract keywords for debugging/display
        keywords = extract_keywords(request.query)
        
        # Calculate how many nodes to get from each method
        keyword_count = max(1, request.top_k // 2)  # At least 1, half of total
        embedding_count = max(1, request.top_k // 2)  # At least 1, half of total
        
        # Find top nodes using combined approach
        top_nodes_with_scores = find_top_nodes_combined(
            request.query, 
            keyword_count=keyword_count, 
            embedding_count=embedding_count
        )
        
        # Limit to requested top_k
        top_nodes_with_scores = top_nodes_with_scores[:request.top_k]
        
        # Prepare response data
        top_nodes = []
        for node_id, score, match_details in top_nodes_with_scores:
            node = graph_data.nodes[node_id]
            top_nodes.append({
                "id": node_id,
                "entity_type": node.entity_type,
                "description": node.description,
                "similarity_score": float(score),
                "match_details": match_details,
                "match_type": "combined"
            })
        
        # Initialize context window with top node descriptions
        context_window = []
        for node in top_nodes:
            context_text = f"Entity: {node['id']} ({node['entity_type']})\nDescription: {node['description']}\nMatch: {node['match_details']}"
            context_window.append(context_text)
        
        hop_data = {
            "current_hop": 0,
            "visited_nodes": [node["id"] for node in top_nodes],
            "query_embedding": embedding_model.encode(request.query).tolist(),
            "extracted_keywords": keywords
        }
        
        return QueryResponse(
            query=request.query,
            top_nodes=top_nodes,
            hop_data=hop_data,
            context_window=context_window
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/hop", response_model=HopResponse)
async def perform_hop(request: HopRequest):
    """Perform one hop and return next nodes"""
    try:
        query_embedding = np.array(request.query_embedding)
        
        # Get adjacent nodes with similarity scores and their connecting edges
        adjacent_nodes, hop_edges = get_adjacent_nodes_with_similarity(
            request.current_nodes, 
            query_embedding, 
            request.threshold,
            request.direction
        )
        
        # Filter out already visited nodes (cycle detection)
        new_nodes = []
        similarity_scores = {}
        cycle_detected = False
        filtered_hop_edges = []
        
        for node_id, similarity, edge_desc in adjacent_nodes:
            if node_id in request.visited_nodes:
                cycle_detected = True
                continue
            
            node = graph_data.nodes.get(node_id)
            if node:
                new_nodes.append({
                    "id": node_id,
                    "entity_type": node.entity_type,
                    "description": node.description,
                    "similarity_score": float(similarity),
                    "edge_description": edge_desc
                })
                similarity_scores[node_id] = float(similarity)
        
        # Filter hop edges to only include edges to new nodes
        new_node_ids = {node["id"] for node in new_nodes}
        filtered_hop_edges = [edge for edge in hop_edges if edge["target"] in new_node_ids]
        
        # Add new context
        new_context = []
        for node in new_nodes:
            context_text = f"Node: {node['description']}"
            if context_text not in request.context_window:
                new_context.append(context_text)
        
        # Check if we've reached the max nodes limit
        total_nodes_after_hop = len(request.visited_nodes) + len(new_nodes)
        hop_complete = len(new_nodes) == 0 or total_nodes_after_hop >= request.max_nodes
        
        return HopResponse(
            next_nodes=new_nodes,
            new_context=new_context,
            hop_complete=hop_complete,
            cycle_detected=cycle_detected,
            similarity_scores=similarity_scores,
            hop_edges=filtered_hop_edges
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing hop: {str(e)}")

@app.post("/answer", response_model=AnswerResponse)
async def generate_answer(request: AnswerRequest):
    """Generate final answer using Gemini"""
    try:
        # Prepare context for Gemini
        context = "\n".join(request.context_window)
        
        prompt = f"""
        Based on the following context information, please answer the user's query comprehensively:
        
        Query: {request.query}
        
        Context:
        {context}
        
        Please provide a detailed and accurate answer based on the context provided. If the context doesn't contain enough information to fully answer the query, please indicate what information is missing.
        """
        
        provider = get_gemini_provider()
        response = await provider.generate_content(prompt)
        
        return AnswerResponse(
            answer=response if response else "I apologize, but I couldn't generate a response at this time.",
            sources_used=request.context_window[:5]  # Return first 5 sources used
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@app.post("/rewrite-query", response_model=RewriteResponse)
async def rewrite_query(request: RewriteRequest):
    """Rewrite query using high-degree nodes for better context"""
    try:
        # Find high-degree nodes (nodes with many connections)
        node_degrees = dict(graph_data.graph.degree())
        high_degree_nodes = [
            (node_id, degree) for node_id, degree in node_degrees.items() 
            if degree >= request.edge_threshold
        ]
        
        # Sort by degree and take top nodes
        high_degree_nodes.sort(key=lambda x: x[1], reverse=True)
        top_high_degree = high_degree_nodes[:10]  # Top 10 high-degree nodes
        
        # Prepare node information for rewriting
        node_info = []
        for node_id, degree in top_high_degree:
            node = graph_data.nodes.get(node_id)
            if node:
                node_info.append({
                    "id": node_id,
                    "entity_type": node.entity_type,
                    "description": node.description[:200] + "..." if len(node.description) > 200 else node.description,
                    "degree": degree
                })
        
        # Create context for LLM to rewrite query
        nodes_context = "\n".join([
            f"- {node['id']} ({node['entity_type']}): {node['description']} [Connected to {node['degree']} other entities]"
            for node in node_info
        ])
        
        rewrite_prompt = f"""
        You are a knowledge graph query enhancement agent. You work with graph data where nodes represent entities and edges represent relationships between entities.
        
        Original User Query: "{request.query}"
        
        You have access to a knowledge graph with the following highly connected nodes (nodes with {request.edge_threshold}+ edges):
        {nodes_context}
        
        Context: These nodes have many connections ({request.edge_threshold}+ edges each), making them central entities in the knowledge graph. They are likely important concepts that can provide better search context.
        
        Task: Enhance the user's query by incorporating relevant entity names from the high-degree nodes above. The enhanced query should:
        1. Preserve the original search intent and meaning
        2. Include specific entity names that are relevant to the query
        3. Add contextual details that will improve search precision
        4. Remain natural and readable
        5. Focus on entities with the highest edge counts when possible
        
        Guidelines:
        - If the query is vague (like "what is this about?"), incorporate multiple relevant high-degree entities
        - If the query is already specific, only add relevant complementary entities
        - Maintain the question format if the original was a question
        - Use entity names exactly as they appear in the node list
        
        Provide only the enhanced query text, nothing else.
        """
        
        provider = get_gemini_provider()
        rewritten_query = await provider.generate_content(rewrite_prompt)
        
        if not rewritten_query:
            rewritten_query = request.query  # Fallback to original
        
        # Create reasoning
        reasoning = f"Enhanced query using {len(node_info)} high-degree entities (threshold: {request.edge_threshold}+ connections). These entities are central to the knowledge graph and may provide better search context."
        
        return RewriteResponse(
            original_query=request.query,
            rewritten_query=rewritten_query.strip(),
            high_degree_nodes=node_info,
            enhancement_reasoning=reasoning
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rewriting query: {str(e)}")

@app.get("/graph-stats")
async def get_graph_stats():
    """Get graph statistics"""
    node_degrees = dict(graph_data.graph.degree()) if graph_data.graph else {}
    return {
        "nodes_count": len(graph_data.nodes),
        "edges_count": len(graph_data.edges),
        "node_types": list(set(node.entity_type for node in graph_data.nodes.values())),
        "avg_node_degree": sum(node_degrees.values()) / len(graph_data.nodes) if graph_data.nodes else 0,
        "high_degree_nodes_30": len([d for d in node_degrees.values() if d >= 30]),
        "high_degree_nodes_40": len([d for d in node_degrees.values() if d >= 40]),
        "max_degree": max(node_degrees.values()) if node_degrees else 0
    }

@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML file at root route"""
    try:
        frontend_path = Path(__file__).parent / "query_pipeline_frontend.html"
        if frontend_path.exists():
            return FileResponse(str(frontend_path), media_type="text/html")
        else:
            raise HTTPException(status_code=404, detail="Frontend file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving frontend: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
