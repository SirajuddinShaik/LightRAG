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
from datetime import datetime

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

# Global graph data and persistence
graph_data = GraphData()
_last_folder_path = None
_persistence_file = "query_pipeline_state.json"

def save_persistence_state():
    """Save current state to persistence file"""
    global _last_folder_path
    try:
        state = {
            "last_folder_path": _last_folder_path,
            "nodes_count": len(graph_data.nodes),
            "edges_count": len(graph_data.edges),
            "timestamp": str(datetime.now())
        }
        with open(_persistence_file, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"State persisted: {state}")
    except Exception as e:
        print(f"Failed to save persistence state: {e}")

def load_persistence_state():
    """Load and restore state from persistence file"""
    global _last_folder_path
    try:
        if Path(_persistence_file).exists():
            with open(_persistence_file, 'r') as f:
                state = json.load(f)
            
            _last_folder_path = state.get("last_folder_path")
            print(f"Found persistence state: {state}")
            
            # Try to auto-load from last used path
            if _last_folder_path and Path(_last_folder_path).exists():
                print(f"Auto-loading graph data from: {_last_folder_path}")
                graph_data.load_from_files(_last_folder_path)
                print(f"Auto-loaded: {len(graph_data.nodes)} nodes, {len(graph_data.edges)} edges")
                return True
    except Exception as e:
        print(f"Failed to load persistence state: {e}")
    return False

# Auto-load on startup
auto_loaded = load_persistence_state()
if auto_loaded:
    print("✅ System initialized with persisted data")
else:
    print("🔄 System starting fresh - no persisted data found")

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
    global _last_folder_path
    folder_path = request.get("folder_path")
    if not folder_path:
        raise HTTPException(status_code=400, detail="folder_path is required")
    
    try:
        graph_data.load_from_files(folder_path)
        
        # Save persistence state
        _last_folder_path = folder_path
        save_persistence_state()
        
        return {
            "status": "success",
            "nodes_count": len(graph_data.nodes),
            "edges_count": len(graph_data.edges),
            "message": "Graph data loaded successfully",
            "persisted": True
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

@app.get("/status")
async def get_system_status():
    """Get current system status including data loading state"""
    try:
        has_data = len(graph_data.nodes) > 0 and len(graph_data.edges) > 0
        
        return {
            "status": "ready" if has_data else "uninitialized",
            "has_data": has_data,
            "nodes_count": len(graph_data.nodes),
            "edges_count": len(graph_data.edges),
            "chunks_count": len(graph_data.chunks_data),
            "entities_count": len(graph_data.entities_data),
            "last_folder_path": _last_folder_path,
            "message": "Graph data loaded and ready" if has_data else "No graph data loaded"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking status: {str(e)}")

@app.delete("/delete")
async def clear_all_data():
    """Clear all loaded graph data"""
    try:
        # Clear all data structures
        graph_data.nodes.clear()
        graph_data.edges.clear()
        graph_data.graph.clear()
        graph_data.directed_graph.clear()
        graph_data.chunks_data.clear()
        graph_data.entities_data.clear()
        
        return {
            "status": "success",
            "message": "All graph data cleared successfully",
            "nodes_count": len(graph_data.nodes),
            "edges_count": len(graph_data.edges)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing data: {str(e)}")

@app.get("/show")
async def show_whole_graph():
    """Display the whole graph using premium Cytoscape.js visualization with uniform nodes and edge details"""
    try:
        if len(graph_data.nodes) == 0:
            raise HTTPException(status_code=400, detail="No graph data loaded. Please initialize first.")
        
        # Premium color scheme for entity types
        node_colors = {
            'organization': '#e74c3c',    # Red - Companies, partners, vendors
            'person': '#3498db',          # Blue - Employees, executives, contractors
            'team': '#2ecc71',            # Green - Departments, squads, groups
            'project': '#f39c12',         # Orange - Internal initiatives, OKRs
            'document': '#9b59b6',        # Purple - Files, reports, wikis, notes
            'product': '#1abc9c',         # Turquoise - Products, services, features
            'event': '#e67e22',           # Dark Orange - Meetings, launches, training
            'task': '#34495e',            # Dark Blue-Gray - Tickets, issues, action items
            'location': '#95a5a6',        # Gray - Office, HQ, remote site
            'technology': '#f1c40f',      # Yellow - Tools, tech stack, SaaS apps
            'customer': '#8e44ad',        # Dark Purple - Client, account, partner org
            'default': '#7f8c8d'          # Default Gray
        }
        
        # Prepare data for Cytoscape.js
        elements = []
        
        # Add nodes with uniform size
        uniform_node_size = 60  # Fixed size for all nodes
        for node_id, node in graph_data.nodes.items():
            entity_type = node.entity_type.lower() if node.entity_type else 'default'
            description = node.description or 'No description available'
            
            # Calculate degree for display purposes
            degree = len([e for e in graph_data.edges if e.source == node_id or e.target == node_id])
            
            elements.append({
                'data': {
                    'id': node_id,
                    'label': node_id,
                    'entity_type': entity_type,
                    'description': description,
                    'degree': degree,
                    'size': uniform_node_size,
                    'color': node_colors.get(entity_type, node_colors['default'])
                }
            })
        
        # Add edges with detailed information
        for edge in graph_data.edges:
            elements.append({
                'data': {
                    'id': f"{edge.source}-{edge.target}",
                    'source': edge.source,
                    'target': edge.target,
                    'label': edge.keywords or '',
                    'description': edge.description or 'No description available',
                    'keywords': edge.keywords or 'No keywords',
                    'weight': edge.weight,
                    'edge_type': 'relationship'
                }
            })
        
        # Create premium HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightRAG Knowledge Graph - Premium View</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            overflow: hidden;
        }}
        
        .container {{
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}
        
        .header {{
            background: rgba(255, 255, 255, 0.98);
            padding: 20px 30px;
            backdrop-filter: blur(20px);
            box-shadow: 0 4px 30px rgba(0,0,0,0.1);
            z-index: 1000;
            border-bottom: 1px solid rgba(255,255,255,0.3);
        }}
        
        .title {{
            font-size: 28px;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 15px;
            text-align: center;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
            justify-content: center;
        }}
        
        .control-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .search-box {{
            padding: 12px 20px;
            border: 2px solid #e1e8ed;
            border-radius: 30px;
            font-size: 14px;
            width: 280px;
            background: white;
            transition: all 0.3s ease;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
            transform: translateY(-1px);
        }}
        
        .btn {{
            padding: 12px 24px;
            border: none;
            border-radius: 25px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .btn-primary {{
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        }}
        
        .btn-secondary {{
            background: linear-gradient(45deg, #f8f9fa, #e9ecef);
            color: #495057;
        }}
        
        .btn-secondary:hover {{
            background: linear-gradient(45deg, #e9ecef, #dee2e6);
            transform: translateY(-1px);
        }}
        
        .stats {{
            background: linear-gradient(45deg, #e74c3c, #c0392b);
            color: white;
            padding: 12px 20px;
            border-radius: 25px;
            font-weight: 600;
            font-size: 14px;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
        }}
        
        .legend {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            max-width: 800px;
        }}
        
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 13px;
            font-weight: 500;
            padding: 8px 12px;
            background: rgba(255,255,255,0.7);
            border-radius: 15px;
            transition: all 0.3s ease;
        }}
        
        .legend-item:hover {{
            background: rgba(255,255,255,0.9);
            transform: translateY(-1px);
        }}
        
        .legend-color {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}
        
        .graph-container {{
            flex: 1;
            position: relative;
            background: white;
            margin: 15px;
            border-radius: 20px;
            box-shadow: 0 15px 50px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        #cy {{
            width: 100%;
            height: 100%;
        }}
        
        .info-panel {{
            position: absolute;
            top: 25px;
            right: 25px;
            background: rgba(255, 255, 255, 0.98);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
            max-width: 400px;
            max-height: 500px;
            overflow-y: auto;
            backdrop-filter: blur(20px);
            display: none;
            z-index: 1000;
            border: 1px solid rgba(255,255,255,0.3);
        }}
        
        .info-panel h3 {{
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 20px;
            font-weight: 700;
        }}
        
        .info-panel p {{
            color: #7f8c8d;
            line-height: 1.6;
            margin-bottom: 10px;
        }}
        
        .info-panel .close-btn {{
            position: absolute;
            top: 15px;
            right: 20px;
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            color: #95a5a6;
            transition: all 0.3s ease;
        }}
        
        .info-panel .close-btn:hover {{
            color: #e74c3c;
            transform: scale(1.1);
        }}
        
        .layout-selector {{
            background: white;
            border: 2px solid #e1e8ed;
            border-radius: 25px;
            padding: 10px 16px;
            font-size: 14px;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .floating-help {{
            position: absolute;
            bottom: 25px;
            left: 25px;
            background: rgba(44, 62, 80, 0.95);
            color: white;
            padding: 20px;
            border-radius: 15px;
            font-size: 13px;
            max-width: 320px;
            backdrop-filter: blur(20px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        
        .floating-help h4 {{
            margin-bottom: 12px;
            color: #3498db;
            font-size: 16px;
            font-weight: 600;
        }}
        
        .floating-help ul {{
            list-style: none;
            margin: 0;
            padding: 0;
        }}
        
        .floating-help li {{
            margin-bottom: 6px;
            padding-left: 20px;
            position: relative;
            line-height: 1.4;
        }}
        
        .floating-help li:before {{
            content: "✨";
            position: absolute;
            left: 0;
        }}
        
        .entity-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="title">
                <i class="fas fa-project-diagram"></i> LightRAG Knowledge Graph
            </div>
            <div class="controls">
                <div class="control-group">
                    <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search nodes or edges..." />
                </div>
                
                <div class="control-group">
                    <select class="layout-selector" id="layoutSelector">
                        <option value="cose">🎯 Physics Layout (CoSE)</option>
                        <option value="fcose">⚡ Fast CoSE</option>
                        <option value="circle">⭕ Circle Layout</option>
                        <option value="grid">📊 Grid Layout</option>
                        <option value="concentric">🎪 Concentric Layout</option>
                        <option value="breadthfirst">🌳 Breadth First</option>
                    </select>
                </div>
                
                <button class="btn btn-primary" id="fitBtn">
                    <i class="fas fa-expand-arrows-alt"></i> Fit View
                </button>
                <button class="btn btn-secondary" id="resetBtn">
                    <i class="fas fa-redo"></i> Reset
                </button>
                <button class="btn btn-secondary" id="exportBtn">
                    <i class="fas fa-download"></i> Export PNG
                </button>
                
                <div class="stats">
                    <i class="fas fa-chart-bar"></i> {len(graph_data.nodes)} Nodes | {len(graph_data.edges)} Edges
                </div>
            </div>
            
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['organization']}"></div>
                    <span><i class="fas fa-building"></i> Organizations</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['person']}"></div>
                    <span><i class="fas fa-user"></i> People</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['team']}"></div>
                    <span><i class="fas fa-users"></i> Teams</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['project']}"></div>
                    <span><i class="fas fa-project-diagram"></i> Projects</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['document']}"></div>
                    <span><i class="fas fa-file-alt"></i> Documents</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['product']}"></div>
                    <span><i class="fas fa-box"></i> Products</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['event']}"></div>
                    <span><i class="fas fa-calendar"></i> Events</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['task']}"></div>
                    <span><i class="fas fa-tasks"></i> Tasks</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['location']}"></div>
                    <span><i class="fas fa-map-marker-alt"></i> Locations</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['technology']}"></div>
                    <span><i class="fas fa-cogs"></i> Technology</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: {node_colors['customer']}"></div>
                    <span><i class="fas fa-handshake"></i> Customers</span>
                </div>
            </div>
        </div>
        
        <div class="graph-container">
            <div id="cy"></div>
            
            <div class="info-panel" id="infoPanel">
                <button class="close-btn" id="closeInfo">&times;</button>
                <div id="infoContent"></div>
            </div>
            
            <div class="floating-help">
                <h4><i class="fas fa-lightbulb"></i> Navigation Guide</h4>
                <ul>
                    <li>Click nodes for details</li>
                    <li>Click edges for relationship info</li>
                    <li>Drag to pan around</li>
                    <li>Mouse wheel to zoom</li>
                    <li>Right-click to highlight neighbors</li>
                    <li>Double-click background to deselect</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Graph data
        const elements = {json.dumps(elements, indent=2)};
        
        // Initialize Cytoscape with premium settings
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            
            elements: elements,
            
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'background-color': 'data(color)',
                        'label': 'data(label)',
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'font-size': '12px',
                        'font-weight': 'bold',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'color': '#2c3e50',
                        'text-outline-width': 2,
                        'text-outline-color': '#ffffff',
                        'border-width': 3,
                        'border-color': '#ffffff',
                        'transition-property': 'background-color, border-color, width, height, border-width',
                        'transition-duration': '0.3s',
                        'box-shadow': '0 4px 15px rgba(0,0,0,0.2)'
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-color': '#e74c3c',
                        'border-width': 5,
                        'box-shadow': '0 8px 25px rgba(231, 76, 60, 0.4)'
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'width': 3,
                        'line-color': '#95a5a6',
                        'target-arrow-color': '#95a5a6',
                        'target-arrow-shape': 'triangle',
                        'curve-style': 'bezier',
                        'opacity': 0.8,
                        'transition-property': 'line-color, width, opacity, target-arrow-color',
                        'transition-duration': '0.3s',
                        'arrow-scale': 1.2
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                        'line-color': '#e74c3c',
                        'target-arrow-color': '#e74c3c',
                        'width': 5,
                        'opacity': 1
                    }}
                }},
                {{
                    selector: '.highlighted',
                    style: {{
                        'background-color': '#f39c12',
                        'line-color': '#f39c12',
                        'target-arrow-color': '#f39c12',
                        'border-color': '#f39c12',
                        'transition-property': 'background-color, line-color, target-arrow-color, border-color',
                        'transition-duration': '0.5s'
                    }}
                }},
                {{
                    selector: '.faded',
                    style: {{
                        'opacity': 0.3,
                        'text-opacity': 0.3
                    }}
                }}
            ],
            
            layout: {{
                name: 'cose',
                animate: true,
                animationDuration: 1500,
                fit: true,
                padding: 60,
                nodeRepulsion: function( node ){{ return 4096; }},
                nodeOverlap: 8,
                idealEdgeLength: function( edge ){{ return 64; }},
                edgeElasticity: function( edge ){{ return 64; }},
                nestingFactor: 1.2,
                gravity: 1,
                numIter: 1500,
                initialTemp: 1000,
                coolingFactor: 0.99,
                minTemp: 1.0
            }},
            
            wheelSensitivity: 0.3,
            minZoom: 0.1,
            maxZoom: 8
        }});
        
        // Event handlers
        const infoPanel = document.getElementById('infoPanel');
        const infoContent = document.getElementById('infoContent');
        const searchBox = document.getElementById('searchBox');
        const layoutSelector = document.getElementById('layoutSelector');
        
        // Node click handler
        cy.on('tap', 'node', function(evt) {{
            const node = evt.target;
            const data = node.data();
            
            const entityColor = data.color;
            const entityType = data.entity_type;
            
            infoContent.innerHTML = `
                <h3><i class="fas fa-circle" style="color: ${{entityColor}}"></i> ${{data.label}}</h3>
                <p><span class="entity-badge" style="background-color: ${{entityColor}}; color: white;">${{entityType}}</span></p>
                <p><strong><i class="fas fa-link"></i> Connections:</strong> ${{data.degree}}</p>
                <p><strong><i class="fas fa-info-circle"></i> Description:</strong></p>
                <p style="font-style: italic; max-height: 250px; overflow-y: auto; padding: 10px; background: #f8f9fa; border-radius: 8px;">${{data.description}}</p>
            `;
            
            infoPanel.style.display = 'block';
        }});
        
        // Edge click handler - NEW FEATURE
        cy.on('tap', 'edge', function(evt) {{
            const edge = evt.target;
            const data = edge.data();
            
            infoContent.innerHTML = `
                <h3><i class="fas fa-arrows-alt-h"></i> Relationship Details</h3>
                <p><strong><i class="fas fa-play"></i> From:</strong> ${{data.source}}</p>
                <p><strong><i class="fas fa-stop"></i> To:</strong> ${{data.target}}</p>
                <p><strong><i class="fas fa-tags"></i> Keywords:</strong> ${{data.keywords}}</p>
                <p><strong><i class="fas fa-weight-hanging"></i> Weight:</strong> ${{data.weight}}</p>
                <p><strong><i class="fas fa-info-circle"></i> Description:</strong></p>
                <p style="font-style: italic; max-height: 250px; overflow-y: auto; padding: 10px; background: #f8f9fa; border-radius: 8px;">${{data.description}}</p>
            `;
            
            infoPanel.style.display = 'block';
        }});
        
        // Background click handler
        cy.on('tap', function(evt) {{
            if (evt.target === cy) {{
                infoPanel.style.display = 'none';
                cy.elements().removeClass('highlighted faded');
            }}
        }});
        
        // Right-click to highlight neighbors
        cy.on('cxttap', 'node', function(evt) {{
            const node = evt.target;
            const neighbors = node.neighborhood().add(node);
            
            cy.elements().addClass('faded');
            neighbors.removeClass('faded').addClass('highlighted');
        }});
        
        // Close info panel
        document.getElementById('closeInfo').onclick = function() {{
            infoPanel.style.display = 'none';
        }};
        
        // Enhanced search functionality
        let searchTimeout;
        searchBox.addEventListener('input', function() {{
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {{
                const searchTerm = this.value.toLowerCase();
                
                if (searchTerm === '') {{
                    cy.elements().removeClass('highlighted faded');
                    return;
                }}
                
                const matchingNodes = cy.nodes().filter(function(node) {{
                    const data = node.data();
                    return data.label.toLowerCase().includes(searchTerm) ||
                           data.description.toLowerCase().includes(searchTerm) ||
                           data.entity_type.toLowerCase().includes(searchTerm);
                }});
                
                const matchingEdges = cy.edges().filter(function(edge) {{
                    const data = edge.data();
                    return data.keywords.toLowerCase().includes(searchTerm) ||
                           data.description.toLowerCase().includes(searchTerm);
                }});
                
                const allMatches = matchingNodes.union(matchingEdges);
                
                if (allMatches.length > 0) {{
                    cy.elements().addClass('faded');
                    allMatches.removeClass('faded').addClass('highlighted');
                    
                    // Fit to matching elements
                    cy.fit(allMatches, 100);
                }}
            }}, 300);
        }});
        
        // Enhanced layout selector
        layoutSelector.addEventListener('change', function() {{
            const layoutName = this.value;
            const layoutOptions = {{
                cose: {{
                    name: 'cose',
                    animate: true,
                    animationDuration: 1500,
                    fit: true,
                    padding: 60,
                    nodeRepulsion: function( node ){{ return 4096; }},
                    nodeOverlap: 8,
                    idealEdgeLength: function( edge ){{ return 64; }},
                    edgeElasticity: function( edge ){{ return 64; }},
                    nestingFactor: 1.2,
                    gravity: 1,
                    numIter: 1500
                }},
                fcose: {{
                    name: 'fcose',
                    animate: true,
                    fit: true,
                    padding: 60,
                    nodeDimensionsIncludeLabels: true,
                    uniformNodeDimensions: false,
                    packComponents: true,
                    stepSize: 40,
                    samplingType: true,
                    sampleSize: 25,
                    nodeSeparation: 75,
                    piTol: 0.0000001,
                    nodeRepulsion: 4500,
                    idealEdgeLength: 50,
                    edgeElasticity: 0.45,
                    nestingFactor: 0.1,
                    gravity: 0.25,
                    numIter: 2500
                }},
                circle: {{
                    name: 'circle',
                    animate: true,
                    fit: true,
                    padding: 60,
                    avoidOverlap: true,
                    radius: 200
                }},
                grid: {{
                    name: 'grid',
                    animate: true,
                    fit: true,
                    padding: 60,
                    avoidOverlap: true,
                    rows: Math.ceil(Math.sqrt(cy.nodes().length))
                }},
                concentric: {{
                    name: 'concentric',
                    animate: true,
                    fit: true,
                    padding: 60,
                    avoidOverlap: true,
                    concentric: function( node ){{
                        return node.degree();
                    }},
                    levelWidth: function( nodes ){{
                        return 3;
                    }},
                    minNodeSpacing: 50
                }},
                breadthfirst: {{
                    name: 'breadthfirst',
                    animate: true,
                    fit: true,
                    padding: 60,
                    directed: false,
                    roots: cy.nodes().first(),
                    spacingFactor: 2,
                    avoidOverlap: true
                }}
            }};
            
            cy.layout(layoutOptions[layoutName]).run();
        }});
        
        // Control buttons
        document.getElementById('fitBtn').onclick = function() {{
            cy.fit(undefined, 60);
        }};
        
        document.getElementById('resetBtn').onclick = function() {{
            cy.elements().removeClass('highlighted faded');
            searchBox.value = '';
            infoPanel.style.display = 'none';
            cy.fit(undefined, 60);
        }};
        
        document.getElementById('exportBtn').onclick = function() {{
            const png = cy.png({{
                output: 'blob',
                bg: 'white',
                full: true,
                scale: 3
            }});
            
            const link = document.createElement('a');
            link.download = 'lightrag_knowledge_graph.png';
            link.href = URL.createObjectURL(png);
            link.click();
        }};
        
        // Initial fit
        cy.ready(function() {{
            cy.fit(undefined, 60);
        }});
        
        // Responsive handling
        window.addEventListener('resize', function() {{
            cy.resize();
            cy.fit(undefined, 60);
        }});
    </script>
</body>
</html>
        """
        
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating graph visualization: {{str(e)}}")

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
