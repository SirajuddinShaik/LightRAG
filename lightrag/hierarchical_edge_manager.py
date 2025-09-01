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
        """Calculate similarity between two entities based on subcategories and descriptions."""
        
        try:
            # Get subcategories
            cat1 = set(entity1_data.get("subcategories", []))
            cat2 = set(entity2_data.get("subcategories", []))
            
            # Calculate Jaccard similarity for subcategories
            if cat1 or cat2:
                intersection = len(cat1.intersection(cat2))
                union = len(cat1.union(cat2))
                category_similarity = intersection / union if union > 0 else 0.0
            else:
                category_similarity = 0.0
            
            # Calculate entity type similarity
            type1 = entity1_data.get("entity_type", "").lower()
            type2 = entity2_data.get("entity_type", "").lower()
            type_similarity = 1.0 if type1 == type2 else 0.0
            
            # Calculate description similarity using simple keyword overlap
            desc1_words = set(entity1_data.get("description", "").lower().split())
            desc2_words = set(entity2_data.get("description", "").lower().split())
            
            if desc1_words or desc2_words:
                desc_intersection = len(desc1_words.intersection(desc2_words))
                desc_union = len(desc1_words.union(desc2_words))
                desc_similarity = desc_intersection / desc_union if desc_union > 0 else 0.0
            else:
                desc_similarity = 0.0
            
            # Weighted combination
            final_similarity = (
                0.5 * category_similarity + 
                0.3 * type_similarity + 
                0.2 * desc_similarity
            )
            
            return final_similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
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
            from .utils import split_string_by_multi_markers
            import re
            
            # Split response into records
            records = split_string_by_multi_markers(
                response,
                [context["record_delimiter"], context["completion_delimiter"]]
            )
            
            new_nodes = []
            new_edges = []
            entity_renames = []
            
            for record in records:
                record = record.strip()
                if not record:
                    continue
                    
                # Extract content within parentheses
                match = re.search(r'\((.*?)\)', record)
                if not match:
                    continue
                    
                content = match.group(1)
                parts = content.split(context["tuple_delimiter"])
                
                if len(parts) < 2:
                    continue
                
                action = parts[0].strip().strip('"')
                
                if action == "new_node" and len(parts) >= 5:
                    new_nodes.append({
                        "name": parts[1].strip().strip('"'),
                        "type": parts[2].strip().strip('"'),
                        "description": parts[3].strip().strip('"'),
                        "confidence": float(parts[4].strip().strip('"')) if parts[4].strip().strip('"').replace('.', '').isdigit() else 0.8
                    })
                    
                elif action == "new_edge" and len(parts) >= 6:
                    new_edges.append({
                        "source": parts[1].strip().strip('"'),
                        "target": parts[2].strip().strip('"'),
                        "description": parts[3].strip().strip('"'),
                        "keywords": parts[4].strip().strip('"'),
                        "weight": float(parts[5].strip().strip('"')) if parts[5].strip().strip('"').replace('.', '').isdigit() else 8.0
                    })
                    
                elif action == "rename_entity" and len(parts) >= 4:
                    entity_renames.append({
                        "old_name": parts[1].strip().strip('"'),
                        "new_name": parts[2].strip().strip('"'),
                        "reason": parts[3].strip().strip('"')
                    })
            
            return {
                "new_nodes": new_nodes,
                "new_edges": new_edges,
                "entity_renames": entity_renames
            }
            
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
    
    async def smart_edge_routing(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: Dict[str, Any]
    ) -> bool:
        """
        Main entry point for smart edge routing with hierarchical management.
        
        Returns:
            bool: True if edge was successfully routed
        """
        try:
            # Check if source node exists and exceeds edge limit
            source_node_exists = await self.knowledge_graph_inst.has_node(source_node_id)
            if source_node_exists and await self.check_edge_limit(source_node_id):
                out_degree_edges = await self.knowledge_graph_inst.node_out_edges(source_node_id)
                logger.info(f"Node {source_node_id} exceeds outbound edge limit ({len(out_degree_edges)} >= {self.edge_limit}), triggering hierarchical splitting")
                
                return await self.trigger_hierarchical_splitting(
                    source_node_id, 
                    target_node_id, 
                    edge_data,
                    out_degree_edges
                )
            
            # For new nodes or nodes that don't exceed limit, just create the edge normally
            # Skip smart routing if target node doesn't exist yet
            target_node_data = await self.knowledge_graph_inst.get_node(target_node_id)
            if not target_node_data:
                # Target node doesn't exist yet, just create edge normally
                await self.knowledge_graph_inst.upsert_edge(source_node_id, target_node_id, edge_data)
                return True
            
            # Find best parent node using smart routing only if both nodes exist
            if source_node_exists:
                best_parent, max_score = await self.find_best_parent_node(
                    target_node_id,
                    target_node_data,
                    source_node_id
                )
                
                # Only route to different parent if it has a meaningfully better score
                if best_parent != source_node_id and max_score > 0.1:  # Minimum meaningful score
                    logger.info(f"Routing edge to best parent: {best_parent} (max score: {max_score:.3f})")
                    await self.knowledge_graph_inst.upsert_edge(best_parent, target_node_id, edge_data)
                else:
                    # Create edge normally with original source
                    await self.knowledge_graph_inst.upsert_edge(source_node_id, target_node_id, edge_data)
            else:
                # Source node doesn't exist, create edge normally
                await self.knowledge_graph_inst.upsert_edge(source_node_id, target_node_id, edge_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error in smart edge routing: {e}")
            return False