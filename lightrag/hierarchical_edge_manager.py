"""
Hierarchical Edge Management Module for LightRAG

This module implements:
1. Edge limiting using hierarchy when the number of edges exceeds a limit
2. Smart routing with similarity threshold checking
3. Hierarchical splitting when edge limits are exceeded
"""

from __future__ import annotations
import asyncio
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
import json
import time

from .base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage
from .utils import logger, compute_mdhash_id
from .constants import GRAPH_FIELD_SEP
from .prompt import PROMPTS


class HierarchicalEdgeManager:
    """Manages hierarchical edge limiting and smart routing for knowledge graphs."""
    
    def __init__(
        self,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
        global_config: dict,
        edge_limit: int = 50,
        llm_response_cache: BaseKVStorage = None
    ):
        self.knowledge_graph_inst = knowledge_graph_inst
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
        self.global_config = global_config
        self.edge_limit = edge_limit
        self.llm_response_cache = llm_response_cache
        
        # Circuit breaker to prevent excessive hierarchical splitting
        self._recent_splits = {}  # node_id -> timestamp
        self._split_cooldown = 300  # 5 minutes cooldown between splits for same node
        
    async def check_edge_limit(self, node_id: str) -> bool:
        """Check if a node has exceeded the outbound edge limit."""
        try:
            # Check outbound edges only for hierarchical splitting
            out_edges = await self.knowledge_graph_inst.node_out_edges(node_id)
            logger.debug(f"Node {node_id} has {len(out_edges)} outbound edges (limit: {self.edge_limit})")
            return len(out_edges) >= self.edge_limit
        except Exception as e:
            logger.error(f"Error checking edge limit for node {node_id}: {e}")
            return False
    
    async def get_node_with_subcategories(self, node_id: str) -> Dict[str, Any]:
        """Get node data including subcategories for similarity checking."""
        node_data = await self.knowledge_graph_inst.get_node(node_id)
        if not node_data:
            return None
            
        # Add subcategories if not present
        if "subcategories" not in node_data:
            subcategories = await self._generate_subcategories(node_id, node_data)
            # Convert list data to JSON string for GraphML compatibility
            node_data["subcategories"] = json.dumps(subcategories) if isinstance(subcategories, list) else subcategories
            
        return node_data
    
    async def _generate_subcategories(self, node_id: str, node_data: Dict[str, Any]) -> List[str]:
        """Generate subcategories for a node based on its description and connections."""
        try:
            # Get connected nodes to understand context
            edges = await self.knowledge_graph_inst.get_node_edges(node_id)
            connected_entities = []
            
            if edges:
                for src, tgt in edges[:10]:  # Limit to first 10 for performance
                    other_node = tgt if src == node_id else src
                    other_data = await self.knowledge_graph_inst.get_node(other_node)
                    if other_data:
                        connected_entities.append({
                            "name": other_node,
                            "type": other_data.get("entity_type", "UNKNOWN"),
                            "description": other_data.get("description", "")[:100]  # Truncate
                        })
            
            # Generate subcategories using LLM
            subcategories = await self._llm_generate_subcategories(
                node_id, 
                node_data.get("description", ""),
                node_data.get("entity_type", "UNKNOWN"),
                connected_entities
            )
            
            return subcategories
            
        except Exception as e:
            logger.error(f"Error generating subcategories for {node_id}: {e}")
            return ["general"]  # Fallback
    
    async def _llm_generate_subcategories(
        self, 
        entity_name: str, 
        description: str, 
        entity_type: str,
        connected_entities: List[Dict]
    ) -> List[str]:
        """Use LLM to generate meaningful subcategories for an entity."""
        
        prompt = f"""
Generate 3-5 specific subcategories for the following entity that would help with similarity matching and hierarchical organization.

Entity: {entity_name}
Type: {entity_type}
Description: {description}

Connected Entities: {json.dumps(connected_entities, indent=2)}

Requirements:
1. Subcategories should be specific and meaningful
2. Focus on functional aspects, not just entity type
3. Consider the entity's role in business workflows
4. Each subcategory should be 1-3 words
5. Return as a JSON list of strings

Example output: ["authentication", "security_validation", "user_verification"]

Subcategories:
"""
        
        try:
            use_llm_func = self.global_config["llm_model_func"]
            response = await use_llm_func(prompt)
            
            # Parse JSON response
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                subcategories = json.loads(json_match.group())
                return [cat.lower().strip() for cat in subcategories if isinstance(cat, str)]
            else:
                # Fallback parsing
                lines = response.strip().split('\n')
                subcategories = []
                for line in lines:
                    if line.strip() and not line.startswith('#'):
                        clean_cat = line.strip().strip('"-').lower()
                        if clean_cat:
                            subcategories.append(clean_cat)
                return subcategories[:5]  # Limit to 5
                
        except Exception as e:
            logger.error(f"Error in LLM subcategory generation: {e}")
            
        # Fallback based on entity type and description
        fallback_categories = []
        if entity_type.lower() in ["person", "user"]:
            fallback_categories = ["user_management", "authentication"]
        elif entity_type.lower() in ["technology", "system"]:
            fallback_categories = ["system_operations", "infrastructure"]
        elif entity_type.lower() in ["organization", "company"]:
            fallback_categories = ["business_operations", "organizational"]
        else:
            fallback_categories = ["general", "functional"]
            
        return fallback_categories
    
    async def find_best_parent_node(
        self, 
        new_entity_name: str, 
        new_entity_data: Dict[str, Any],
        target_node_id: str
    ) -> Tuple[str, float]:
        """
        Find the best parent node using smart routing with similarity threshold.
        
        Returns:
            Tuple of (best_node_id, similarity_score)
        """
        try:
            # Start with the target node
            current_node = target_node_id
            best_node = target_node_id
            best_score = 0.0
            
            # Get target node with subcategories
            target_data = await self.get_node_with_subcategories(target_node_id)
            if not target_data:
                return target_node_id, 0.0
            
            # Calculate similarity with target node
            target_score = await self._calculate_similarity(new_entity_data, target_data)
            best_score = target_score
            
            # Check child nodes recursively
            await self._check_child_nodes_recursively(
                new_entity_data, current_node, best_node, best_score
            )
            
            return best_node, best_score
            
        except Exception as e:
            logger.error(f"Error finding best parent node: {e}")
            return target_node_id, 0.0
    
    async def _check_child_nodes_recursively(
        self, 
        new_entity_data: Dict[str, Any],
        current_node: str,
        best_node: str,
        best_score: float,
        visited: Set[str] = None,
        depth: int = 0,
        max_depth: int = 2  # Reduced from 3 to 2 to prevent deep recursion
    ) -> Tuple[str, float]:
        """Recursively check child nodes for better similarity matches."""
        
        if visited is None:
            visited = set()
            
        # Add safety checks
        if depth >= max_depth or current_node in visited or len(visited) > 20:
            return best_node, best_score
            
        visited.add(current_node)
        
        # Add timeout for this recursive operation
        start_time = time.time()
        timeout_seconds = 30  # 30 seconds timeout for recursion
        
        try:
            # Get all edges from current node
            edges = await self.knowledge_graph_inst.get_node_edges(current_node)
            if not edges:
                return best_node, best_score
            
            # Check each connected node with timeout protection
            for src, tgt in edges:
                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Recursive similarity check timed out at depth {depth}")
                    break
                    
                child_node = tgt if src == current_node else src
                
                if child_node in visited:
                    continue
                    
                # Get child node data with subcategories
                child_data = await self.get_node_with_subcategories(child_node)
                if not child_data:
                    continue
                
                # Calculate similarity
                similarity = await self._calculate_similarity(new_entity_data, child_data)
                
                if similarity > best_score:
                    best_score = similarity
                    best_node = child_node
                    
                    # If this child has higher similarity, check its children too
                    # Always explore children of nodes with better scores
                    # But only if we haven't exceeded time or depth limits
                    if (similarity > best_score * 0.8 and 
                        depth < max_depth - 1 and 
                        time.time() - start_time < timeout_seconds * 0.8):
                        best_node, best_score = await self._check_child_nodes_recursively(
                            new_entity_data, child_node, best_node, best_score, 
                            visited, depth + 1, max_depth
                        )
            
            return best_node, best_score
            
        except Exception as e:
            logger.error(f"Error in recursive child node checking: {e}")
            return best_node, best_score
    
    async def _calculate_similarity(
        self, 
        entity1_data: Dict[str, Any], 
        entity2_data: Dict[str, Any]
    ) -> float:
        """Calculate comprehensive similarity between two entities using all available data and embeddings."""
        
        try:
            # 1. Get subcategories
            cat1 = set()
            cat2 = set()
            
            subcats1 = entity1_data.get("subcategories", [])
            if isinstance(subcats1, str):
                try:
                    import json
                    subcats1 = json.loads(subcats1)
                except:
                    subcats1 = [subcats1] if subcats1 else []
            if isinstance(subcats1, list):
                cat1 = set(cat.lower() for cat in subcats1)
                
            subcats2 = entity2_data.get("subcategories", [])
            if isinstance(subcats2, str):
                try:
                    import json
                    subcats2 = json.loads(subcats2)
                except:
                    subcats2 = [subcats2] if subcats2 else []
            if isinstance(subcats2, list):
                cat2 = set(cat.lower() for cat in subcats2)
            
            # Calculate Jaccard similarity for subcategories
            if cat1 or cat2:
                intersection = len(cat1.intersection(cat2))
                union = len(cat1.union(cat2))
                category_similarity = intersection / union if union > 0 else 0.0
            else:
                category_similarity = 0.0
            
            # 2. Calculate entity type similarity
            type1 = entity1_data.get("entity_type", "").lower()
            type2 = entity2_data.get("entity_type", "").lower()
            type_similarity = 1.0 if type1 == type2 else 0.0
            
            # 3. Calculate description similarity using word overlap
            desc1_words = set(entity1_data.get("description", "").lower().split())
            desc2_words = set(entity2_data.get("description", "").lower().split())
            
            if desc1_words or desc2_words:
                desc_intersection = len(desc1_words.intersection(desc2_words))
                desc_union = len(desc1_words.union(desc2_words))
                desc_similarity = desc_intersection / desc_union if desc_union > 0 else 0.0
            else:
                desc_similarity = 0.0
            
            # 4. Calculate keywords similarity (if available)
            keywords1 = set()
            keywords2 = set()
            
            # Check for keywords field in entity data
            if "keywords" in entity1_data:
                keywords1 = set(entity1_data.get("keywords", "").lower().split(","))
                keywords1 = {k.strip() for k in keywords1 if k.strip()}
                
            if "keywords" in entity2_data:
                keywords2 = set(entity2_data.get("keywords", "").lower().split(","))
                keywords2 = {k.strip() for k in keywords2 if k.strip()}
            
            if keywords1 or keywords2:
                keyword_intersection = len(keywords1.intersection(keywords2))
                keyword_union = len(keywords1.union(keywords2))
                keyword_similarity = keyword_intersection / keyword_union if keyword_union > 0 else 0.0
            else:
                keyword_similarity = 0.0
            
            # 5. Calculate entity name similarity (for related naming patterns)
            name1 = entity1_data.get("entity_id", entity1_data.get("entity_name", "")).lower()
            name2 = entity2_data.get("entity_id", entity2_data.get("entity_name", "")).lower()
            
            name1_words = set(name1.split())
            name2_words = set(name2.split())
            
            if name1_words or name2_words:
                name_intersection = len(name1_words.intersection(name2_words))
                name_union = len(name1_words.union(name2_words))
                name_similarity = name_intersection / name_union if name_union > 0 else 0.0
            else:
                name_similarity = 0.0
            
            # 6. Calculate source/context similarity (if available)
            source1 = entity1_data.get("source_id", "").lower()
            source2 = entity2_data.get("source_id", "").lower()
            
            if source1 and source2:
                source_similarity = 1.0 if source1 == source2 else 0.0
            else:
                source_similarity = 0.0
            
            # 7. Calculate comprehensive text similarity (all text fields combined)
            all_text1 = " ".join([
                entity1_data.get("description", ""),
                entity1_data.get("keywords", ""),
                name1,
                " ".join(cat1) if cat1 else ""
            ]).lower()
            
            all_text2 = " ".join([
                entity2_data.get("description", ""),
                entity2_data.get("keywords", ""),
                name2,
                " ".join(cat2) if cat2 else ""
            ]).lower()
            
            all_words1 = set(all_text1.split())
            all_words2 = set(all_text2.split())
            
            if all_words1 or all_words2:
                text_intersection = len(all_words1.intersection(all_words2))
                text_union = len(all_words1.union(all_words2))
                comprehensive_text_similarity = text_intersection / text_union if text_union > 0 else 0.0
            else:
                comprehensive_text_similarity = 0.0
            
            # 8. Calculate embedding-based semantic similarity
            embedding_similarity = await self._calculate_embedding_similarity(entity1_data, entity2_data)
            
            # Weighted combination of all similarity measures with embedding similarity
            final_similarity = (
                0.20 * category_similarity +          # Subcategories are very important
                0.15 * type_similarity +              # Entity type is important
                0.15 * desc_similarity +              # Description similarity
                0.15 * keyword_similarity +           # Keywords if available
                0.15 * embedding_similarity +         # Semantic embedding similarity
                0.10 * comprehensive_text_similarity + # Overall text similarity
                0.05 * name_similarity +              # Entity name patterns
                0.05 * source_similarity              # Source context
            )
            
            logger.debug(f"Comprehensive similarity breakdown:")
            logger.debug(f"  - Categories: {category_similarity:.3f}")
            logger.debug(f"  - Type: {type_similarity:.3f}")  
            logger.debug(f"  - Description: {desc_similarity:.3f}")
            logger.debug(f"  - Keywords: {keyword_similarity:.3f}")
            logger.debug(f"  - Embeddings: {embedding_similarity:.3f}")
            logger.debug(f"  - Names: {name_similarity:.3f}")
            logger.debug(f"  - Source: {source_similarity:.3f}")
            logger.debug(f"  - Overall text: {comprehensive_text_similarity:.3f}")
            logger.debug(f"  - Final: {final_similarity:.3f}")
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive similarity: {e}")
            return 0.0
    
    async def _calculate_embedding_similarity(
        self,
        entity1_data: Dict[str, Any],
        entity2_data: Dict[str, Any]
    ) -> float:
        """Calculate semantic similarity using embedding functions."""
        try:
            # Get embedding function from global config
            embedding_func = self.global_config.get("embedding_func")
            if not embedding_func:
                logger.debug("No embedding function available, skipping embedding similarity")
                return 0.0
            
            # Prepare text content for both entities
            entity1_content = self._prepare_entity_content_for_embedding(entity1_data)
            entity2_content = self._prepare_entity_content_for_embedding(entity2_data)
            
            if not entity1_content.strip() or not entity2_content.strip():
                logger.debug("Empty content for embedding similarity calculation")
                return 0.0
            
            # Get embeddings for both entities
            embeddings = await embedding_func([entity1_content, entity2_content])
            
            if embeddings is None or len(embeddings) < 2:
                logger.debug("Failed to get embeddings")
                return 0.0
            
            # Calculate cosine similarity between embeddings
            import numpy as np
            
            emb1 = np.array(embeddings[0])
            emb2 = np.array(embeddings[1])
            
            # Normalize vectors
            emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
            emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
            
            # Calculate cosine similarity
            cosine_sim = np.dot(emb1_norm, emb2_norm)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, (cosine_sim + 1.0) / 2.0))
            
            logger.debug(f"Embedding similarity: {similarity:.3f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating embedding similarity: {e}")
            return 0.0
    
    def _prepare_entity_content_for_embedding(self, entity_data: Dict[str, Any]) -> str:
        """Prepare entity content for embedding calculation."""
        try:
            content_parts = []
            
            # Add entity name/id
            name = entity_data.get("entity_id", entity_data.get("entity_name", ""))
            if name:
                content_parts.append(name)
            
            # Add entity type
            entity_type = entity_data.get("entity_type", "")
            if entity_type:
                content_parts.append(entity_type)
            
            # Add description
            description = entity_data.get("description", "")
            if description:
                content_parts.append(description)
            
            # Add keywords
            keywords = entity_data.get("keywords", "")
            if keywords:
                content_parts.append(keywords)
            
            # Add subcategories
            subcats = entity_data.get("subcategories", [])
            if isinstance(subcats, str):
                try:
                    import json
                    subcats = json.loads(subcats)
                except:
                    subcats = [subcats] if subcats else []
            
            if isinstance(subcats, list) and subcats:
                content_parts.append(" ".join(subcats))
            
            # Join all parts with spaces
            return " ".join(content_parts).strip()
            
        except Exception as e:
            logger.error(f"Error preparing entity content for embedding: {e}")
            return ""
    
    async def trigger_hierarchical_splitting(
        self, 
        source_node_id: str,
        new_target_node_id: str,
        new_edge_data: Dict[str, Any],
        out_degree_edges: List[Tuple[str, str]]
    ) -> bool:
        """
        Trigger hierarchical splitting when edge limit is exceeded.
        
        Args:
            source_node_id: The node that exceeded the edge limit
            new_target_node_id: The new node trying to connect
            new_edge_data: Data for the new edge
            
        Returns:
            bool: True if splitting was successful
        """
        try:
            logger.info(f"Triggering hierarchical splitting for node: {source_node_id}")
            
            
            # Circuit breaker: check if we've split this node recently
            current_time = time.time()
            if source_node_id in self._recent_splits:
                last_split_time = self._recent_splits[source_node_id]
                if current_time - last_split_time < self._split_cooldown:
                    logger.warning(f"Node {source_node_id} was split recently, skipping (cooldown: {self._split_cooldown}s)")
                    # Fallback to normal edge creation
                    await self.knowledge_graph_inst.upsert_edge(source_node_id, new_target_node_id, new_edge_data)
                    return True
            
            # Record this split attempt
            self._recent_splits[source_node_id] = current_time
            
            
            
            # Get ONLY outbound edges from the source node
            outbound_edges = await self.knowledge_graph_inst.get_outbound_edges(source_node_id)
            inbound_edges = await self.knowledge_graph_inst.get_inbound_edges(source_node_id)
            # print(outbound_edges)
            if not outbound_edges:
                return False
            
            logger.info(f"Node {source_node_id} has {len(outbound_edges)} outbound edges for hierarchical splitting")
            logger.info(f"Node {source_node_id} has {len(inbound_edges)} inbound edges (not used for splitting)")
            # Collect only outbound connected entities
            connected_entities = []
            edges_to_delete = []
            
            for src, tgt in outbound_edges:
                # src is always source_node_id, tgt is the target node
                target_node_data = await self.knowledge_graph_inst.get_node(tgt)
                edge_data = await self.knowledge_graph_inst.get_edge(src, tgt)
                
                if target_node_data and edge_data:
                    connected_entities.append({
                        "entity_name": tgt,
                        "entity_type": target_node_data.get("entity_type", "UNKNOWN"),
                        "description": target_node_data.get("description", ""),
                        "edge_description": edge_data.get("description", ""),
                        "edge_keywords": edge_data.get("keywords", ""),
                        "weight": edge_data.get("weight", 1.0)
                    })
                    edges_to_delete.append((src, tgt))
            
            # Add the new entity trying to connect
            new_node_data = await self.knowledge_graph_inst.get_node(new_target_node_id)
            if new_node_data:
                connected_entities.append({
                    "entity_name": new_target_node_id,
                    "entity_type": new_node_data.get("entity_type", "UNKNOWN"),
                    "description": new_node_data.get("description", ""),
                    "edge_description": new_edge_data.get("description", ""),
                    "edge_keywords": new_edge_data.get("keywords", ""),
                    "weight": new_edge_data.get("weight", 1.0)
                })
            
            # Get parent node details
            parent_data = await self.knowledge_graph_inst.get_node(source_node_id)
            
            # Use LLM to create hierarchical grouping
            grouping_result = await self._llm_hierarchical_grouping(
                source_node_id,
                parent_data,
                connected_entities
            )
            
            if not grouping_result:
                logger.error("Failed to get hierarchical grouping from LLM")
                return False
            
            # Apply the hierarchical grouping
            success = await self._apply_hierarchical_grouping(
                source_node_id,
                edges_to_delete,
                grouping_result
            )
            
            return success
            
        except Exception as e:
            logger.error(f"Error in hierarchical splitting: {e}")
            return False
    
    async def _llm_hierarchical_grouping(
        self,
        parent_node_id: str,
        parent_data: Dict[str, Any],
        connected_entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM to create hierarchical grouping."""
        
        try:
            # Prepare entities list for the prompt
            entities_list = f"All these Entities are linked From {parent_node_id}:\n"
            for i, entity in enumerate(connected_entities, 1):
                entities_list += f"Entity {i}: {entity['entity_name']} (Type: {entity['entity_type']}) - {entity['description'][:100]}\n"
            
            # Get entity types from global config
            entity_types = self.global_config.get("addon_params", {}).get(
                "entity_types", 
                ["organization", "person", "team", "project", "document", "product", "event", "task", "location", "technology", "customer"]
            )
            
            # Prepare context
            context = {
                "language": self.global_config.get("addon_params", {}).get("language", "English"),
                "entity_types": ", ".join(entity_types),
                "entity_type_analysis": f"Total entities: {len(connected_entities)}",
                "total_entities": len(connected_entities),  # Add the missing total_entities key
                "domain_context": f"Parent node: {parent_node_id} ({parent_data.get('entity_type', 'UNKNOWN')})",
                "parent_node_name": parent_node_id,  # Add the missing parent_node_name key
                "examples": PROMPTS.get("hierarchical_grouping_examples", [""])[0] if PROMPTS.get("hierarchical_grouping_examples") else "",
                "max_groups": min(3, max(2, len(connected_entities) // 10)),  # Dynamic grouping
                "entities_list": entities_list,
                "tuple_delimiter": PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "|"),
                "record_delimiter": PROMPTS.get("DEFAULT_RECORD_DELIMITER", "##"),
                "completion_delimiter": PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "#####")
            }
            
            # Format the prompt
            prompt = PROMPTS["hierarchical_grouping"].format(**context)
            
            # Call LLM
            use_llm_func = self.global_config["llm_model_func"]
            response = await use_llm_func(prompt)
            
            # Parse the response
            parsed_result = await self._parse_hierarchical_response(response, context)
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Error in LLM hierarchical grouping: {e}")
            return None
    
    async def _parse_hierarchical_response(
        self, 
        response: str, 
        context: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Parse the LLM response for hierarchical grouping."""
        
        try:
            import re
            
            new_nodes = []
            new_edges = []
            entity_renames = []
            
            # Updated parsing logic to handle the actual LLM response format
            # Split by ## to get individual records
            records = response.split('##')
            
            for record in records:
                record = record.strip()
                if not record or record == '#####':
                    continue
                    
                # Extract content within parentheses - improved regex
                match = re.search(r'\((.*?)\)', record)
                if not match:
                    continue
                    
                content = match.group(1)
                # Split by | which is the actual tuple delimiter in the response
                parts = [part.strip().strip('"') for part in content.split('|')]
                
                if len(parts) < 2:
                    continue
                
                action = parts[0].strip()
                logger.debug(f"Parsing action: {action} with {len(parts)} parts: {parts}")
                
                if action == "new_node" and len(parts) >= 5:
                    try:
                        confidence = float(parts[4]) if parts[4].replace('.', '').replace('-', '').isdigit() else 0.8
                    except (ValueError, IndexError):
                        confidence = 0.8
                        
                    new_nodes.append({
                        "name": parts[1],
                        "type": parts[2],
                        "description": parts[3],
                        "confidence": confidence
                    })
                    logger.debug(f"Added new node: {parts[1]}")
                    
                elif action == "new_edge" and len(parts) >= 6:
                    try:
                        weight = float(parts[5]) if parts[5].replace('.', '').replace('-', '').isdigit() else 8.0
                    except (ValueError, IndexError):
                        weight = 8.0
                        
                    new_edges.append({
                        "source": parts[1],
                        "target": parts[2],
                        "description": parts[3],
                        "keywords": parts[4],
                        "weight": weight
                    })
                    logger.debug(f"Added new edge: {parts[1]} -> {parts[2]}")
                    
                elif action == "rename_entity" and len(parts) >= 4:
                    entity_renames.append({
                        "old_name": parts[1],
                        "new_name": parts[2],
                        "reason": parts[3]
                    })
                    logger.debug(f"Added entity rename: {parts[1]} -> {parts[2]}")
            
            result = {
                "new_nodes": new_nodes,
                "new_edges": new_edges,
                "entity_renames": entity_renames
            }
            
            logger.info(f"Parsed hierarchical response: {len(new_nodes)} nodes, {len(new_edges)} edges, {len(entity_renames)} renames")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing hierarchical response: {e}")
            return None
    
    async def _apply_hierarchical_grouping(
        self,
        parent_node_id: str,
        edges_to_delete: List[Tuple[str, str]],
        grouping_result: Dict[str, Any]
    ) -> bool:
        """Apply the hierarchical grouping result to the knowledge graph."""
        
        try:
            # First, create new nodes
            for node_info in grouping_result.get("new_nodes", []):
                # Generate subcategories
                subcategories = await self._generate_subcategories_from_description(
                    node_info["description"], 
                    node_info["type"]
                )
                
                # Convert list data to JSON string for GraphML compatibility
                subcategories_json = json.dumps(subcategories) if isinstance(subcategories, list) else subcategories
                
                node_data = {
                    "entity_id": node_info["name"],
                    "entity_type": node_info["type"],
                    "description": node_info["description"],
                    "source_id": f"hierarchical_split_{int(time.time())}",
                    "file_path": "hierarchical_grouping",
                    "created_at": int(time.time()),
                    "subcategories": subcategories_json
                }
                
                await self.knowledge_graph_inst.upsert_node(node_info["name"], node_data)
                
                # Also add to entities vector database for consistency
                await self._upsert_entity_to_vdb(
                    entity_name=node_info["name"],
                    entity_data=node_data
                )
                
                logger.info(f"Created new hierarchical node: {node_info['name']}")
            
            # Apply entity renames with edge preservation
            for rename_info in grouping_result.get("entity_renames", []):
                old_name = rename_info["old_name"]
                new_name = rename_info["new_name"]
                reason = rename_info.get("reason", "Hierarchical grouping rename")
                
                # Validate rename request
                if old_name == new_name:
                    logger.debug(f"Skipping rename: {old_name} -> {new_name} (same name)")
                    continue
                
                # Check if new name already exists
                existing_node = await self.knowledge_graph_inst.get_node(new_name)
                if existing_node:
                    logger.warning(f"Cannot rename {old_name} -> {new_name}: target name already exists")
                    continue
                
                # Use the helper function to rename with edge preservation
                success = await self._rename_entity_with_edge_preservation(
                    old_name, new_name, reason
                )
                
                if not success:
                    logger.warning(f"Failed to rename entity: {old_name} -> {new_name}")
                else:
                    # Update any references in the new_edges list to use the new name
                    for edge_info in grouping_result.get("new_edges", []):
                        if edge_info["source"] == old_name:
                            edge_info["source"] = new_name
                        if edge_info["target"] == old_name:
                            edge_info["target"] = new_name
            
            # Delete old edges
            for src, tgt in edges_to_delete:
                try:
                    # Get edge data before deletion for potential reuse
                    edge_data = await self.knowledge_graph_inst.get_edge(src, tgt)
                    await self.knowledge_graph_inst.remove_edges([(src, tgt)])
                except Exception as e:
                    logger.warning(f"Could not delete edge {src}-{tgt}: {e}")
            
            # Create new edges
            for edge_info in grouping_result.get("new_edges", []):
                source = edge_info["source"]
                target = edge_info["target"]
                
                edge_data = {
                    "description": edge_info["description"],
                    "keywords": edge_info["keywords"],
                    "weight": edge_info["weight"],
                    "source_id": f"hierarchical_split_{int(time.time())}",
                    "file_path": "hierarchical_grouping",
                    "created_at": int(time.time())
                }
                
                await self.knowledge_graph_inst.upsert_edge(source_node_id=source, target_node_id=target, edge_data=edge_data)
                
                # Also add to relationships vector database for consistency
                await self._upsert_relationship_to_vdb(
                    src=source,
                    tgt=target,
                    edge_data=edge_data
                )
                
                logger.info(f"Created new hierarchical edge: {source} -> {target}")
            
            logger.info(f"Successfully applied hierarchical grouping for {parent_node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error applying hierarchical grouping: {e}")
            return False
    
    async def _rename_entity_with_edge_preservation(
        self,
        old_name: str,
        new_name: str,
        reason: str = "Entity renaming"
    ) -> bool:
        """Rename an entity while preserving all its edges.
        
        Args:
            old_name: Original entity name
            new_name: New entity name
            reason: Reason for renaming (for logging)
            
        Returns:
            bool: True if renaming was successful
        """
        try:
            # Get old node data
            old_data = await self.knowledge_graph_inst.get_node(old_name)
            if not old_data:
                logger.warning(f"Cannot rename {old_name}: node not found")
                return False
            
            # Create new node with renamed data
            await self.knowledge_graph_inst.upsert_node(new_name, old_data)
            
            # Also add to entities vector database
            await self._upsert_entity_to_vdb(
                entity_name=new_name,
                entity_data=old_data
            )
            
            # Get all edges connected to the old entity
            edges = await self.knowledge_graph_inst.get_node_edges(old_name)
            if edges:
                # Recreate edges for the new entity
                edges_to_delete = []
                for source, target in edges:
                    edge_data = await self.knowledge_graph_inst.get_edge(source, target)
                    if edge_data:
                        # Create new edge with renamed entity
                        if source == old_name:
                            await self.knowledge_graph_inst.upsert_edge(new_name, target, edge_data)
                            # Also add to relationships vector database
                            await self._upsert_relationship_to_vdb(new_name, target, edge_data)
                        else:
                            await self.knowledge_graph_inst.upsert_edge(source, new_name, edge_data)
                            # Also add to relationships vector database
                            await self._upsert_relationship_to_vdb(source, new_name, edge_data)
                        
                        # Mark old edge for deletion
                        edges_to_delete.append((source, target))
                
                # Delete old edges from both graph and vector database
                if edges_to_delete:
                    await self.knowledge_graph_inst.remove_edges(edges_to_delete)
                    
                    # Also delete from relationships vector database
                    try:
                        from .utils import compute_mdhash_id
                        rel_ids_to_delete = []
                        for src, tgt in edges_to_delete:
                            rel_id = compute_mdhash_id(src + tgt, prefix="rel-")
                            rel_ids_to_delete.append(rel_id)
                        
                        if rel_ids_to_delete:
                            await self.relationships_vdb.delete(rel_ids_to_delete)
                            logger.debug(f"Deleted {len(rel_ids_to_delete)} old relationships from vector database")
                    except Exception as e:
                        logger.warning(f"Could not delete old relationships from vector database: {e}")
                    logger.info(f"Transferred {len(edges_to_delete)} edges from {old_name} to {new_name}")
            
            # Delete old node from both graph and vector database
            await self.knowledge_graph_inst.delete_node(old_name)
            
            # Also delete from entities vector database
            try:
                from .utils import compute_mdhash_id
                old_entity_id = compute_mdhash_id(old_name, prefix="ent-")
                await self.entities_vdb.delete([old_entity_id])
                logger.debug(f"Deleted old entity from vector database: {old_name}")
            except Exception as e:
                logger.warning(f"Could not delete old entity {old_name} from vector database: {e}")
            logger.info(f"Successfully renamed entity: {old_name} -> {new_name} ({reason})")
            return True
            
        except Exception as e:
            logger.error(f"Error renaming entity {old_name} -> {new_name}: {e}")
            return False
    
    async def _upsert_entity_to_vdb(
        self,
        entity_name: str,
        entity_data: Dict[str, Any]
    ) -> None:
        """Add entity to vector database for consistency with knowledge graph."""
        try:
            from .utils import compute_mdhash_id
            
            # Create entity content for vector database
            entity_content = f"{entity_name}\n{entity_data.get('description', '')}"
            
            # Create vector database entry
            entity_vdb_data = {
                "id": compute_mdhash_id(entity_name, prefix="ent-"),
                "content": entity_content,
                "entity_name": entity_name,
                "entity_type": entity_data.get("entity_type", "UNKNOWN"),
                "description": entity_data.get("description", ""),
                "source_id": entity_data.get("source_id", ""),
                "created_at": entity_data.get("created_at", 0)
            }
            
            await self.entities_vdb.upsert({entity_vdb_data["id"]: entity_vdb_data})
            logger.debug(f"Added entity to vector database: {entity_name}")
            
        except Exception as e:
            logger.error(f"Error adding entity {entity_name} to vector database: {e}")
    
    async def _upsert_relationship_to_vdb(
        self,
        src: str,
        tgt: str,
        edge_data: Dict[str, Any]
    ) -> None:
        """Add relationship to vector database for consistency with knowledge graph."""
        try:
            from .utils import compute_mdhash_id
            
            # Create relationship content for vector database
            keywords = edge_data.get("keywords", "")
            description = edge_data.get("description", "")
            rel_content = f"{keywords}\t{src}\n{tgt}\n{description}"
            
            # Create vector database entry
            rel_vdb_data = {
                "id": compute_mdhash_id(src + tgt, prefix="rel-"),
                "content": rel_content,
                "src_id": src,
                "tgt_id": tgt,
                "description": description,
                "keywords": keywords,
                "weight": edge_data.get("weight", 1.0),
                "source_id": edge_data.get("source_id", ""),
                "created_at": edge_data.get("created_at", 0)
            }
            
            await self.relationships_vdb.upsert({rel_vdb_data["id"]: rel_vdb_data})
            logger.debug(f"Added relationship to vector database: {src} -> {tgt}")
            
        except Exception as e:
            logger.error(f"Error adding relationship {src} -> {tgt} to vector database: {e}")
    
    async def _generate_subcategories_from_description(
        self, 
        description: str, 
        entity_type: str
    ) -> List[str]:
        """Generate subcategories from description and entity type."""
        
        # Simple keyword-based subcategory generation
        description_lower = description.lower()
        subcategories = []
        
        # Add entity type as base category
        subcategories.append(entity_type.lower())
        
        # Add functional categories based on keywords
        if any(word in description_lower for word in ["manage", "control", "operate"]):
            subcategories.append("management")
        if any(word in description_lower for word in ["process", "workflow", "pipeline"]):
            subcategories.append("process")
        if any(word in description_lower for word in ["security", "auth", "verify"]):
            subcategories.append("security")
        if any(word in description_lower for word in ["data", "information", "analytics"]):
            subcategories.append("data")
        if any(word in description_lower for word in ["user", "customer", "client"]):
            subcategories.append("user_facing")
        
        return list(set(subcategories))  # Remove duplicates
    
    async def _bidirectional_smart_routing(
        self,
        source_node_id: str,
        target_node_id: str,
        source_node_data: Dict[str, Any],
        target_node_data: Dict[str, Any],
        edge_data: Dict[str, Any]
    ) -> bool:
        """
        Perform bidirectional smart routing as per user requirements:
        
        1 edge = 2 entities(A+B) + relation
        Direction 1: Compare (relation + B) against [A and all its subsets]
        Direction 2: Compare (relation + A) against [B and all subcategories]
        Select the best entity recursively
        """
        try:
            logger.debug(f"Starting bidirectional analysis for {source_node_id} -> {target_node_id}")
            
            # Direction 1: Compare (relation + target) against [source + all its subsets]
            direction1_best = await self._compare_relation_with_entity_subsets(
                edge_data, target_node_data, source_node_id, source_node_data, "direction1"
            )
            
            # Direction 2: Compare (relation + source) against [target + all its subcategories]  
            direction2_best = await self._compare_relation_with_entity_subsets(
                edge_data, source_node_data, target_node_id, target_node_data, "direction2"
            )
            
            # Select the best entity recursively based on both directions
            final_choice = await self._select_best_entity_recursively(
                source_node_id, target_node_id, 
                direction1_best, direction2_best, 
                edge_data
            )
            
            # Execute the final routing decision
            final_source, final_target = final_choice
            await self.knowledge_graph_inst.upsert_edge(final_source, final_target, edge_data)
            
            if final_source != source_node_id or final_target != target_node_id:
                logger.info(f"Bidirectional routing: Rerouted {source_node_id} -> {target_node_id} to {final_source} -> {final_target}")
            else:
                logger.info(f"Bidirectional routing: Using original routing {source_node_id} -> {target_node_id}")
            
            return True
                
        except Exception as e:
            logger.error(f"Error in bidirectional smart routing: {e}")
            # Fallback to original routing
            await self.knowledge_graph_inst.upsert_edge(source_node_id, target_node_id, edge_data)
            return True
    
    async def _find_best_entity_match(
        self,
        entity_name: str,
        edge_data: Dict[str, Any],
        entity_role: str,  # "source" or "target"
        default_match: str
    ) -> str:
        """
        Find the best matching entity for a given edge based on compatibility.
        
        Args:
            entity_name: The entity we're trying to match
            edge_data: The edge data containing keywords and description
            entity_role: Whether this entity is the "source" or "target" of the edge
            default_match: Default entity to return if no better match found
            
        Returns:
            str: The best matching entity name
        """
        try:
            # Get all existing entities
            all_nodes = await self.knowledge_graph_inst.get_all_nodes()
            if not all_nodes:
                return default_match
            
            best_match = default_match
            best_score = 0.0
            
            # Create a virtual entity data for the new entity based on edge information
            virtual_entity_data = {
                "entity_name": entity_name,
                "entity_type": "UNKNOWN",  # Will be inferred from edge
                "description": edge_data.get("description", ""),
                "subcategories": await self._infer_subcategories_from_edge(edge_data, entity_role)
            }
            
            # Check compatibility with existing entities
            for node_id, node_data in all_nodes.items():
                if node_id == entity_name:  # Skip self
                    continue
                    
                # Skip if this node already has too many edges
                if await self.check_edge_limit(node_id):
                    continue
                
                # Calculate compatibility score
                compatibility_score = await self._calculate_entity_compatibility(
                    virtual_entity_data, node_data, edge_data, entity_role
                )
                
                if compatibility_score > best_score:
                    best_score = compatibility_score
                    best_match = node_id
            
            # Only return alternative if it's significantly better
            if best_score > 0.2:  # Minimum threshold for routing
                logger.debug(f"Found better match for {entity_name}: {best_match} (score: {best_score:.3f})")
                return best_match
            else:
                return default_match
                
        except Exception as e:
            logger.error(f"Error finding best entity match: {e}")
            return default_match
    
    async def _calculate_edge_entity_compatibility(
        self,
        edge_data: Dict[str, Any],
        entity_data: Dict[str, Any],
        entity_role: str
    ) -> float:
        """
        Calculate how compatible an edge is with a specific entity.
        
        Args:
            edge_data: Edge information (keywords, description)
            entity_data: Entity information (type, description, subcategories)
            entity_role: "source" or "target" - role of entity in the edge
            
        Returns:
            float: Compatibility score between 0.0 and 1.0
        """
        try:
            # Extract edge keywords and description
            edge_keywords = set(edge_data.get("keywords", "").lower().split())
            edge_description = edge_data.get("description", "").lower()
            
            # Extract entity information
            entity_type = entity_data.get("entity_type", "").lower()
            entity_description = entity_data.get("description", "").lower()
            entity_subcategories = set()
            
            # Handle subcategories (could be JSON string or list)
            subcats = entity_data.get("subcategories", [])
            if isinstance(subcats, str):
                try:
                    import json
                    subcats = json.loads(subcats)
                except:
                    subcats = [subcats] if subcats else []
            
            if isinstance(subcats, list):
                entity_subcategories = set(cat.lower() for cat in subcats)
            
            # Calculate different compatibility aspects
            
            # 1. Keyword overlap with entity description
            entity_words = set(entity_description.split())
            keyword_overlap = len(edge_keywords.intersection(entity_words))
            keyword_score = keyword_overlap / max(len(edge_keywords), 1)
            
            # 2. Subcategory relevance
            edge_words = set(edge_description.split())
            subcategory_overlap = len(entity_subcategories.intersection(edge_words))
            subcategory_score = subcategory_overlap / max(len(entity_subcategories), 1)
            
            # 3. Entity type relevance to edge role
            role_score = self._calculate_role_compatibility(entity_type, edge_data, entity_role)
            
            # 4. Description semantic similarity (simple word overlap)
            desc_words = set(entity_description.split())
            edge_desc_words = set(edge_description.split())
            desc_overlap = len(desc_words.intersection(edge_desc_words))
            desc_score = desc_overlap / max(len(desc_words.union(edge_desc_words)), 1)
            
            # Weighted combination
            final_score = (
                0.3 * keyword_score +
                0.25 * subcategory_score +
                0.25 * role_score +
                0.2 * desc_score
            )
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating edge-entity compatibility: {e}")
            return 0.0
    
    def _calculate_role_compatibility(
        self,
        entity_type: str,
        edge_data: Dict[str, Any],
        entity_role: str
    ) -> float:
        """
        Calculate how well an entity type fits a specific role in an edge.
        """
        edge_keywords = edge_data.get("keywords", "").lower()
        edge_description = edge_data.get("description", "").lower()
        
        # Define role compatibility rules
        if entity_role == "source":
            # Source entities are typically actors, systems, or initiators
            if entity_type in ["person", "user", "system", "service", "organization"]:
                if any(word in edge_keywords + edge_description for word in ["manage", "control", "create", "initiate", "send"]):
                    return 0.8
                return 0.6
            elif entity_type in ["technology", "tool", "application"]:
                if any(word in edge_keywords + edge_description for word in ["process", "execute", "run", "operate"]):
                    return 0.7
                return 0.5
        
        elif entity_role == "target":
            # Target entities are typically objects, recipients, or results
            if entity_type in ["document", "data", "product", "result", "output"]:
                if any(word in edge_keywords + edge_description for word in ["receive", "store", "contain", "produce"]):
                    return 0.8
                return 0.6
            elif entity_type in ["person", "user", "customer"]:
                if any(word in edge_keywords + edge_description for word in ["notify", "inform", "deliver", "provide"]):
                    return 0.7
                return 0.5
        
        return 0.4  # Default moderate compatibility
    
    async def _calculate_entity_compatibility(
        self,
        entity1_data: Dict[str, Any],
        entity2_data: Dict[str, Any],
        edge_data: Dict[str, Any],
        entity_role: str
    ) -> float:
        """
        Calculate overall compatibility between two entities for a specific edge.
        """
        try:
            # Base similarity between entities
            entity_similarity = await self._calculate_similarity(entity1_data, entity2_data)
            
            # Edge compatibility with the existing entity
            edge_compatibility = await self._calculate_edge_entity_compatibility(
                edge_data, entity2_data, entity_role
            )
            
            # Combined score with emphasis on edge compatibility
            combined_score = 0.4 * entity_similarity + 0.6 * edge_compatibility
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Error calculating entity compatibility: {e}")
            return 0.0
    
    async def _infer_subcategories_from_edge(
        self,
        edge_data: Dict[str, Any],
        entity_role: str
    ) -> List[str]:
        """
        Infer likely subcategories for an entity based on edge information.
        """
        try:
            keywords = edge_data.get("keywords", "").lower()
            description = edge_data.get("description", "").lower()
            combined_text = f"{keywords} {description}"
            
            subcategories = []
            
            # Infer based on common patterns
            if any(word in combined_text for word in ["auth", "login", "security", "verify"]):
                subcategories.append("authentication")
            
            if any(word in combined_text for word in ["data", "information", "store", "database"]):
                subcategories.append("data_management")
            
            if any(word in combined_text for word in ["user", "customer", "client", "person"]):
                subcategories.append("user_interaction")
            
            if any(word in combined_text for word in ["process", "workflow", "execute", "run"]):
                subcategories.append("process_management")
            
            if any(word in combined_text for word in ["system", "service", "application", "technology"]):
                subcategories.append("system_component")
            
            # Role-specific subcategories
            if entity_role == "source":
                if any(word in combined_text for word in ["manage", "control", "initiate"]):
                    subcategories.append("management_entity")
            elif entity_role == "target":
                if any(word in combined_text for word in ["receive", "store", "contain"]):
                    subcategories.append("storage_entity")
            
            return subcategories if subcategories else ["general"]
            
        except Exception as e:
            logger.error(f"Error inferring subcategories from edge: {e}")
            return ["general"]
    
    async def _compare_relation_with_entity_subsets(
        self,
        edge_data: Dict[str, Any],
        relation_entity_data: Dict[str, Any],
        comparison_entity_id: str,
        comparison_entity_data: Dict[str, Any],
        direction: str
    ) -> Tuple[str, float]:
        """
        Compare (relation + entity) against [comparison_entity + all its subsets/subcategories]
        
        Args:
            edge_data: The relation/edge information
            relation_entity_data: Data of the entity being combined with relation
            comparison_entity_id: ID of the entity to compare against
            comparison_entity_data: Data of the entity to compare against
            direction: "direction1" or "direction2" for logging
            
        Returns:
            Tuple of (best_entity_id, best_score)
        """
        try:
            logger.debug(f"Comparing relation+entity in {direction}")
            
            # Create combined relation+entity profile
            relation_profile = await self._create_relation_entity_profile(edge_data, relation_entity_data)
            
            # Get all possible entities to compare against (including subsets/subcategories)
            candidate_entities = await self._get_entity_subsets_and_subcategories(
                comparison_entity_id, comparison_entity_data
            )
            
            best_entity = comparison_entity_id
            best_score = 0.0
            
            # Compare against each candidate
            for entity_id, entity_data in candidate_entities.items():
                score = await self._calculate_relation_entity_compatibility(
                    relation_profile, entity_data
                )
                
                logger.debug(f"{direction}: {entity_id} compatibility score: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_entity = entity_id
            
            logger.debug(f"{direction} best match: {best_entity} (score: {best_score:.3f})")
            return best_entity, best_score
            
        except Exception as e:
            logger.error(f"Error in {direction} comparison: {e}")
            return comparison_entity_id, 0.0
    
    async def _create_relation_entity_profile(
        self,
        edge_data: Dict[str, Any],
        entity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a combined profile of relation + entity for comparison"""
        try:
            # Extract edge characteristics
            edge_keywords = set(edge_data.get("keywords", "").lower().split())
            edge_description = edge_data.get("description", "").lower()
            edge_words = set(edge_description.split())
            
            # Extract entity characteristics
            entity_type = entity_data.get("entity_type", "").lower()
            entity_description = entity_data.get("description", "").lower()
            entity_words = set(entity_description.split())
            
            # Get entity subcategories
            entity_subcategories = set()
            subcats = entity_data.get("subcategories", [])
            if isinstance(subcats, str):
                try:
                    import json
                    subcats = json.loads(subcats)
                except:
                    subcats = [subcats] if subcats else []
            
            if isinstance(subcats, list):
                entity_subcategories = set(cat.lower() for cat in subcats)
            
            # Create combined profile
            combined_profile = {
                "keywords": edge_keywords.union(entity_words),
                "description_words": edge_words.union(entity_words),
                "entity_type": entity_type,
                "subcategories": entity_subcategories,
                "combined_description": f"{edge_description} {entity_description}",
                "weight": edge_data.get("weight", 1.0)
            }
            
            return combined_profile
            
        except Exception as e:
            logger.error(f"Error creating relation-entity profile: {e}")
            return {}
    
    async def _get_entity_subsets_and_subcategories(
        self,
        entity_id: str,
        entity_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get all entities that represent subsets/subcategories of the given entity
        This includes the entity itself plus related entities
        """
        try:
            candidates = {entity_id: entity_data}
            
            # Get all nodes to find related entities
            all_nodes = await self.knowledge_graph_inst.get_all_nodes()
            if not all_nodes:
                return candidates
            
            # Extract entity characteristics for comparison safely
            entity_type = entity_data.get("entity_type", "").lower() if entity_data else ""
            entity_subcategories = set()
            
            if entity_data:
                subcats = entity_data.get("subcategories", [])
                if isinstance(subcats, str):
                    try:
                        import json
                        subcats = json.loads(subcats)
                    except:
                        subcats = [subcats] if subcats else []
                
                if isinstance(subcats, list):
                    entity_subcategories = set(cat.lower() for cat in subcats if isinstance(cat, str))
            
            # Handle multiple possible formats from get_all_nodes()
            nodes_iterator = []
            
            if isinstance(all_nodes, dict):
                # all_nodes is a dictionary {node_id: node_data}
                for node_id, node_data in all_nodes.items():
                    if node_id != entity_id and node_data is not None:
                        nodes_iterator.append((node_id, node_data))
                        
            elif isinstance(all_nodes, list):
                # all_nodes is a list of node_ids, need to fetch node data
                for node_id in all_nodes:
                    if isinstance(node_id, str) and node_id != entity_id:
                        try:
                            node_data = await self.knowledge_graph_inst.get_node(node_id)
                            if node_data and isinstance(node_data, dict):
                                nodes_iterator.append((node_id, node_data))
                        except Exception as e:
                            logger.debug(f"Could not fetch node data for {node_id}: {e}")
                            continue
                            
            elif hasattr(all_nodes, '__iter__'):
                # Handle other iterable types
                try:
                    for item in all_nodes:
                        if isinstance(item, dict) and "id" in item:
                            # Handle case where all_nodes contains node objects with 'id' field
                            node_id = item.get("id")
                            if node_id and node_id != entity_id:
                                nodes_iterator.append((node_id, item))
                        elif isinstance(item, str) and item != entity_id:
                            # Handle case where all_nodes is a list of node_ids
                            try:
                                node_data = await self.knowledge_graph_inst.get_node(item)
                                if node_data and isinstance(node_data, dict):
                                    nodes_iterator.append((item, node_data))
                            except Exception as e:
                                logger.debug(f"Could not fetch node data for {item}: {e}")
                                continue
                except Exception as e:
                    logger.debug(f"Error iterating over all_nodes: {e}")
            else:
                logger.warning(f"Unexpected format for all_nodes: {type(all_nodes)}")
                return candidates
            
            # Find related entities (same type or overlapping subcategories)
            for node_id, node_data in nodes_iterator:
                try:
                    if not node_data or not isinstance(node_data, dict):
                        continue
                        
                    # Safely extract node characteristics
                    node_type = node_data.get("entity_type", "").lower()
                    node_subcategories = set()
                    
                    node_subcats = node_data.get("subcategories", [])
                    if isinstance(node_subcats, str):
                        try:
                            import json
                            node_subcats = json.loads(node_subcats)
                        except:
                            node_subcats = [node_subcats] if node_subcats else []
                    
                    if isinstance(node_subcats, list):
                        node_subcategories = set(cat.lower() for cat in node_subcats if isinstance(cat, str))
                    
                    # Include if same type or has overlapping subcategories
                    if (node_type and entity_type and node_type == entity_type) or \
                       (entity_subcategories and node_subcategories and 
                        len(entity_subcategories.intersection(node_subcategories)) > 0):
                        candidates[node_id] = node_data
                        
                except Exception as e:
                    logger.debug(f"Error processing node {node_id}: {e}")
                    continue
            
            logger.debug(f"Found {len(candidates)} candidate entities for {entity_id}")
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting entity subsets: {e}")
            return {entity_id: entity_data}
    
    async def _calculate_relation_entity_compatibility(
        self,
        relation_profile: Dict[str, Any],
        entity_data: Dict[str, Any]
    ) -> float:
        """Calculate compatibility between relation+entity profile and target entity"""
        try:
            # Extract target entity characteristics
            target_type = entity_data.get("entity_type", "").lower()
            target_description = entity_data.get("description", "").lower()
            target_words = set(target_description.split())
            
            target_subcategories = set()
            subcats = entity_data.get("subcategories", [])
            if isinstance(subcats, str):
                try:
                    import json
                    subcats = json.loads(subcats)
                except:
                    subcats = [subcats] if subcats else []
            
            if isinstance(subcats, list):
                target_subcategories = set(cat.lower() for cat in subcats)
            
            # Calculate compatibility scores
            
            # 1. Keyword/description overlap
            profile_words = relation_profile.get("description_words", set())
            word_overlap = len(profile_words.intersection(target_words))
            word_score = word_overlap / max(len(profile_words.union(target_words)), 1)
            
            # 2. Subcategory overlap
            profile_subcats = relation_profile.get("subcategories", set())
            subcat_overlap = len(profile_subcats.intersection(target_subcategories))
            subcat_score = subcat_overlap / max(len(profile_subcats.union(target_subcategories)), 1)
            
            # 3. Entity type compatibility
            profile_type = relation_profile.get("entity_type", "")
            type_score = 1.0 if profile_type == target_type else 0.5
            
            # 4. Semantic similarity based on keywords
            profile_keywords = relation_profile.get("keywords", set())
            keyword_overlap = len(profile_keywords.intersection(target_words))
            keyword_score = keyword_overlap / max(len(profile_keywords), 1)
            
            # Weighted combination
            final_score = (
                0.3 * word_score +
                0.3 * subcat_score +
                0.2 * type_score +
                0.2 * keyword_score
            )
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating relation-entity compatibility: {e}")
            return 0.0
    
    async def _select_best_entity_recursively(
        self,
        original_source: str,
        original_target: str,
        direction1_result: Tuple[str, float],
        direction2_result: Tuple[str, float],
        edge_data: Dict[str, Any]
    ) -> Tuple[str, str]:
        """
        Recursively select the best entity based on both direction results
        
        Direction 1: Compare (relation + target) against [source + subsets] -> finds best source
        Direction 2: Compare (relation + source) against [target + subcategories] -> finds best target
        
        Args:
            original_source: Original source entity
            original_target: Original target entity  
            direction1_result: (best_source_alternative, score) from direction 1
            direction2_result: (best_target_alternative, score) from direction 2
            edge_data: Edge data for additional context
            
        Returns:
            Tuple of (final_source, final_target)
        """
        try:
            direction1_source_alternative, direction1_score = direction1_result
            direction2_target_alternative, direction2_score = direction2_result
            
            logger.debug(f"Recursive selection: D1 source alt({direction1_source_alternative}:{direction1_score:.3f}) vs D2 target alt({direction2_target_alternative}:{direction2_score:.3f})")
            
            # Minimum improvement threshold to justify routing change
            improvement_threshold = 0.1
            
            # Check if either direction found a significantly better match
            if direction1_score > direction2_score + improvement_threshold:
                # Direction 1 found better source alternative
                if direction1_source_alternative != original_source:
                    logger.info(f"Recursive selection: Using D1 source rerouting {original_source} -> {direction1_source_alternative}")
                    return direction1_source_alternative, original_target
                else:
                    logger.debug("Direction 1 best match is original source")
                    
            elif direction2_score > direction1_score + improvement_threshold:
                # Direction 2 found better target alternative  
                if direction2_target_alternative != original_target:
                    logger.info(f"Recursive selection: Using D2 target rerouting {original_target} -> {direction2_target_alternative}")
                    return original_source, direction2_target_alternative
                else:
                    logger.debug("Direction 2 best match is original target")
            
            # If both directions found good improvements, choose the higher scoring one
            elif (direction1_score > 0.3 and direction2_score > 0.3):
                if direction1_score > direction2_score:
                    if direction1_source_alternative != original_source:
                        logger.info(f"Recursive selection: Both good, using D1 source {original_source} -> {direction1_source_alternative}")
                        return direction1_source_alternative, original_target
                else:
                    if direction2_target_alternative != original_target:
                        logger.info(f"Recursive selection: Both good, using D2 target {original_target} -> {direction2_target_alternative}")
                        return original_source, direction2_target_alternative
            
            # Check if we can use both improvements (only if they don't conflict)
            elif (direction1_score > 0.25 and direction2_score > 0.25 and 
                  direction1_source_alternative != original_source and 
                  direction2_target_alternative != original_target and
                  direction1_source_alternative != direction2_target_alternative):  # Avoid self-loops
                logger.info(f"Recursive selection: Using both improvements {direction1_source_alternative} -> {direction2_target_alternative}")
                return direction1_source_alternative, direction2_target_alternative
            
            # No significant improvement found, use original routing
            logger.debug("Recursive selection: No significant improvement, using original routing")
            return original_source, original_target
            
        except Exception as e:
            logger.error(f"Error in recursive entity selection: {e}")
            return original_source, original_target

    async def smart_edge_routing(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: Dict[str, Any]
    ) -> bool:
        """
        Main entry point for bidirectional smart edge routing with hierarchical management.
        
        This method analyzes both source and target entities to find the best routing
        for the edge, considering similarity and hierarchical constraints.
        
        Returns:
            bool: True if edge was successfully routed
        """
        try:
            # Get node existence status
            source_node_exists = await self.knowledge_graph_inst.has_node(source_node_id)
            target_node_exists = await self.knowledge_graph_inst.has_node(target_node_id)
            
            # Get node data for both entities if they exist
            source_node_data = await self.knowledge_graph_inst.get_node(source_node_id) if source_node_exists else None
            target_node_data = await self.knowledge_graph_inst.get_node(target_node_id) if target_node_exists else None
            
            # Check for hierarchical splitting needs on both ends
            source_needs_splitting = source_node_exists and await self.check_edge_limit(source_node_id)
            target_needs_splitting = target_node_exists and await self.check_edge_limit(target_node_id)
            
            # Handle hierarchical splitting if needed
            if source_needs_splitting:
                out_degree_edges = await self.knowledge_graph_inst.node_out_edges(source_node_id)
                logger.info(f"Source node {source_node_id} exceeds outbound edge limit ({len(out_degree_edges)} >= {self.edge_limit}), triggering hierarchical splitting")
                
                return await self.trigger_hierarchical_splitting(
                    source_node_id, 
                    target_node_id, 
                    edge_data,
                    out_degree_edges
                )
            
            if target_needs_splitting:
                out_degree_edges = await self.knowledge_graph_inst.node_out_edges(target_node_id)
                logger.info(f"Target node {target_node_id} exceeds outbound edge limit ({len(out_degree_edges)} >= {self.edge_limit}), triggering hierarchical splitting")
                
                # For target splitting, we need to reverse the edge direction logic
                return await self.trigger_hierarchical_splitting(
                    target_node_id, 
                    source_node_id, 
                    edge_data,
                    out_degree_edges
                )
            
            # Perform bidirectional smart routing if both nodes exist
            if source_node_exists and target_node_exists:
                return await self._bidirectional_smart_routing(
                    source_node_id, target_node_id, 
                    source_node_data, target_node_data, 
                    edge_data
                )
            
            # If only one node exists, try to find best parent for the new node
            elif source_node_exists and not target_node_exists:
                # Source exists, target is new - find best parent for target
                best_source = await self._find_best_entity_match(
                    target_node_id, edge_data, "target", source_node_id
                )
                await self.knowledge_graph_inst.upsert_edge(best_source, target_node_id, edge_data)
                logger.info(f"Routed new target {target_node_id} to best source: {best_source}")
                return True
                
            elif target_node_exists and not source_node_exists:
                # Target exists, source is new - find best parent for source
                best_target = await self._find_best_entity_match(
                    source_node_id, edge_data, "source", target_node_id
                )
                await self.knowledge_graph_inst.upsert_edge(source_node_id, best_target, edge_data)
                logger.info(f"Routed new source {source_node_id} to best target: {best_target}")
                return True
            
            else:
                # Both nodes are new, create edge normally
                await self.knowledge_graph_inst.upsert_edge(source_node_id, target_node_id, edge_data)
                logger.info(f"Created edge between new entities: {source_node_id} -> {target_node_id}")
                return True
            
        except Exception as e:
            logger.error(f"Error in bidirectional smart edge routing: {e}")
            return False
