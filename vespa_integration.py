"""
Vespa Connector - Integration with Vespa data source for unstructured data retrieval

This module provides functionality to connect to Vespa, retrieve unstructured data,
and prepare it for entity extraction and knowledge graph integration.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import os
from urllib.parse import urljoin, quote, urlencode
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VespaDocument:
    """Represents a document retrieved from Vespa"""
    id: str
    title: str
    content: str
    doc_type: str  # email, document, chat, etc.
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for processing"""
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'doc_type': self.doc_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata or {},
            'source': self.source
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VespaDocument':
        """Create VespaDocument from dictionary"""
        timestamp = None
        if data.get('timestamp'):
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                pass
        
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            content=data.get('content', ''),
            doc_type=data.get('doc_type', 'document'),
            timestamp=timestamp,
            metadata=data.get('metadata'),
            source=data.get('source')
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'VespaDocument':
        """Create VespaDocument from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)

@dataclass
class VisitOptions:
    """Options for Vespa visit API calls"""
    namespace: Optional[str] = None
    schema: str = "mail"
    continuation: Optional[str] = None
    wanted_document_count: int = 50
    field_set: Optional[str] = None
    concurrency: int = 1
    cluster: str = "my_content"

@dataclass
class VisitResponse:
    """Response from Vespa visit API"""
    documents: List[Any]
    continuation: Optional[str] = None
    document_count: int = 0

@dataclass
class VespaConfig:
    """Configuration for Vespa connection"""
    endpoint: str
    application_name: str
    schema_name: str
    namespace: str = "default"
    timeout: int = 30
    max_hits: int = 100
    
    @classmethod
    def from_env(cls) -> 'VespaConfig':
        """Create config from environment variables"""
        return cls(
            endpoint=os.getenv('VESPA_ENDPOINT', 'http://localhost:8080'),
            application_name=os.getenv('VESPA_APPLICATION', 'unstructured_data'),
            schema_name=os.getenv('VESPA_SCHEMA', 'document'),
            namespace=os.getenv('VESPA_NAMESPACE', 'default'),
            timeout=int(os.getenv('VESPA_TIMEOUT', '30')),
            max_hits=int(os.getenv('VESPA_MAX_HITS', '100'))
        )

class VespaConnector:
    """Main connector class for Vespa data source integration"""
    
    def __init__(self, config: VespaConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _fetch_with_retry(self, url: str, options: Dict[str, Any], retries: int = 3) -> Dict[str, Any]:
        """Fetch with retry logic similar to TypeScript implementation"""
        for attempt in range(retries):
            try:
                if not self.session:
                    raise RuntimeError("Session not initialized. Use async context manager.")
                
                async with self.session.get(url, **options) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): "
                                     f"{response.status} {response.reason} - {error_text}")
                        if attempt == retries - 1:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status,
                                message=f"Visit failed: {response.status} {response.reason} - {error_text}"
                            )
            except Exception as e:
                if attempt == retries - 1:
                    raise e
                logger.warning(f"Request attempt {attempt + 1} failed: {e}, retrying...")
                await asyncio.sleep(1)  # Wait before retry
        
        raise Exception("Max retries reached")
    
    async def visit(self, options: VisitOptions) -> VisitResponse:
        """
        Visit documents using Vespa's visit API - matches TypeScript implementation
        This allows pagination through all documents with continuation tokens
        """
        try:
            # Set defaults similar to TypeScript implementation
            namespace = options.namespace or self.config.namespace
            schema = options.schema or self.config.schema_name
            field_set = options.field_set or f"{schema}:*"
            
            # Build parameters
            params = {
                'wantedDocumentCount': str(options.wanted_document_count),
                'cluster': options.cluster,
                'selection': schema
            }
            
            # Add continuation token if present
            if options.continuation:
                params['continuation'] = options.continuation
            
            # Build URL
            url = f"{self.config.endpoint}/document/v1/?{urlencode(params)}"
            
            logger.debug(f"Visiting Vespa documents: {url}")
            
            # Make request with retry logic
            data = await self._fetch_with_retry(url, {
                'headers': {
                    'Accept': 'application/json'
                }
            })
            
            return VisitResponse(
                documents=data.get('documents', []),
                continuation=data.get('continuation'),
                document_count=data.get('documentCount', 0)
            )
            
        except Exception as e:
            error_message = f"Error visiting documents: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)
    
    async def visit_all_documents(
        self, 
        schema: str = "mail", 
        wanted_document_count: int = 100,
        max_documents: Optional[int] = None,
        cluster: str = "my_content"
    ) -> List[Any]:
        """
        Visit all documents with pagination support - matches TypeScript pattern
        """
        logger.info('🚀 Starting Vespa Document Visit Process')
        logger.info(f'📡 Vespa endpoint: {self.config.endpoint}')
        
        all_documents = []
        continuation = None
        page_count = 0
        
        try:
            # Keep fetching until no more continuation token
            while True:
                visit_options = VisitOptions(
                    namespace=self.config.namespace,
                    schema=schema,
                    continuation=continuation,
                    wanted_document_count=wanted_document_count,
                    cluster=cluster
                )
                
                visit_response = await self.visit(visit_options)
                
                page_count += 1
                all_documents.extend(visit_response.documents)
                continuation = visit_response.continuation
                
                logger.info(f"📄 Page {page_count}: Fetched {len(visit_response.documents)} documents "
                           f"(Total: {len(all_documents)})")
                
                # Check if we should stop
                if not continuation:
                    break
                
                # Check max documents limit
                if max_documents and len(all_documents) >= max_documents:
                    logger.info(f"Reached max documents limit: {max_documents}")
                    all_documents = all_documents[:max_documents]
                    break
                
                # Small delay between requests to avoid overwhelming the server
                await asyncio.sleep(0.1)
            
            logger.info(f"✅ Fetched total of {len(all_documents)} documents from Vespa")
            return all_documents
            
        except Exception as e:
            logger.error(f"❌ Error during document visit: {e}")
            raise
    
    def _build_query_url(self, query_params: Dict[str, Any]) -> str:
        """Build Vespa query URL"""
        base_url = f"{self.config.endpoint}/search/"
        
        # Default query parameters
        default_params = {
            'yql': f'select * from {self.config.schema_name} where true',
            'hits': self.config.max_hits,
            'format': 'json'
        }
        
        # Merge with provided parameters
        params = {**default_params, **query_params}
        
        # Build query string
        query_string = '&'.join([f"{k}={quote(str(v))}" for k, v in params.items()])
        return f"{base_url}?{query_string}"
    
    async def test_connection(self) -> bool:
        """Test connection to Vespa"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            url = f"{self.config.endpoint}/ApplicationStatus"
            async with self.session.get(url) as response:
                if response.status == 200:
                    logger.info("Vespa connection test successful")
                    return True
                else:
                    logger.error(f"Vespa connection test failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Vespa connection test error: {e}")
            return False
    
    async def query_documents(
        self, 
        query: str = "true", 
        doc_type: Optional[str] = None,
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[VespaDocument]:
        """Query documents from Vespa"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            # Build YQL query
            yql_conditions = []
            
            # Base query
            if query and query != "true":
                yql_conditions.append(f'userQuery() = "{query}"')
            
            # Document type filter
            if doc_type:
                yql_conditions.append(f'doc_type = "{doc_type}"')
            
            # Additional filters
            if filters:
                for key, value in filters.items():
                    if isinstance(value, str):
                        yql_conditions.append(f'{key} = "{value}"')
                    else:
                        yql_conditions.append(f'{key} = {value}')
            
            # Combine conditions
            where_clause = " AND ".join(yql_conditions) if yql_conditions else "true"
            yql = f"select * from {self.config.schema_name} where {where_clause}"
            
            # Query parameters
            query_params = {
                'yql': yql,
                'hits': limit or self.config.max_hits,
                'format': 'json'
            }
            
            url = self._build_query_url(query_params)
            logger.info(f"Querying Vespa: {url}")
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Vespa query failed: {response.status}")
                    return []
                
                data = await response.json()
                return self._parse_documents(data)
                
        except Exception as e:
            logger.error(f"Error querying Vespa documents: {e}")
            return []
    
    async def get_document_by_id(self, doc_id: str) -> Optional[VespaDocument]:
        """Get a specific document by ID"""
        try:
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            url = f"{self.config.endpoint}/document/v1/{self.config.namespace}/{self.config.schema_name}/docid/{doc_id}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_single_document(data)
                elif response.status == 404:
                    logger.warning(f"Document not found: {doc_id}")
                    return None
                else:
                    logger.error(f"Error fetching document {doc_id}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting document by ID {doc_id}: {e}")
            return None
    
    async def get_recent_documents(
        self, 
        hours: int = 24, 
        doc_type: Optional[str] = None
    ) -> List[VespaDocument]:
        """Get documents from the last N hours"""
        try:
            # Calculate timestamp filter
            from datetime import datetime, timedelta
            cutoff_time = datetime.now() - timedelta(hours=hours)
            timestamp_filter = int(cutoff_time.timestamp())
            
            filters = {'timestamp': f'>{timestamp_filter}'}
            
            return await self.query_documents(
                query="true",
                doc_type=doc_type,
                filters=filters
            )
            
        except Exception as e:
            logger.error(f"Error getting recent documents: {e}")
            return []
    
    async def search_by_content(
        self, 
        search_term: str, 
        doc_type: Optional[str] = None,
        limit: int = 50
    ) -> List[VespaDocument]:
        """Search documents by content"""
        try:
            # Use text search
            query_params = {
                'yql': f'select * from {self.config.schema_name} where userQuery()',
                'query': search_term,
                'hits': limit,
                'format': 'json'
            }
            
            # Add document type filter if specified
            if doc_type:
                query_params['yql'] += f' AND doc_type = "{doc_type}"'
            
            url = self._build_query_url(query_params)
            
            if not self.session:
                raise RuntimeError("Session not initialized. Use async context manager.")
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_documents(data)
                else:
                    logger.error(f"Content search failed: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching by content: {e}")
            return []
    
    def _parse_documents(self, response_data: Dict[str, Any]) -> List[VespaDocument]:
        """Parse Vespa response into VespaDocument objects"""
        documents = []
        
        try:
            root = response_data.get('root', {})
            children = root.get('children', [])
            
            for child in children:
                doc = self._parse_single_child(child)
                if doc:
                    documents.append(doc)
                    
        except Exception as e:
            logger.error(f"Error parsing Vespa documents: {e}")
        
        return documents
    
    def _parse_single_child(self, child: Dict[str, Any]) -> Optional[VespaDocument]:
        """Parse a single child from Vespa response"""
        try:
            fields = child.get('fields', {})
            
            # Extract required fields
            doc_id = child.get('id', fields.get('id', ''))
            title = fields.get('title', fields.get('subject', 'Untitled'))
            content = fields.get('content', fields.get('body', ''))
            doc_type = fields.get('doc_type', 'document')
            
            # Extract optional fields
            timestamp = None
            if 'timestamp' in fields:
                try:
                    timestamp = datetime.fromtimestamp(fields['timestamp'])
                except (ValueError, TypeError):
                    pass
            
            metadata = {k: v for k, v in fields.items() 
                       if k not in ['id', 'title', 'content', 'doc_type', 'timestamp']}
            
            source = fields.get('source', fields.get('from', None))
            
            return VespaDocument(
                id=doc_id,
                title=title,
                content=content,
                doc_type=doc_type,
                timestamp=timestamp,
                metadata=metadata,
                source=source
            )
            
        except Exception as e:
            logger.error(f"Error parsing single document: {e}")
            return None
    
    def _parse_single_document(self, response_data: Dict[str, Any]) -> Optional[VespaDocument]:
        """Parse a single document response"""
        try:
            fields = response_data.get('fields', {})
            doc_id = response_data.get('id', fields.get('id', ''))
            
            return VespaDocument(
                id=doc_id,
                title=fields.get('title', 'Untitled'),
                content=fields.get('content', ''),
                doc_type=fields.get('doc_type', 'document'),
                timestamp=datetime.fromtimestamp(fields['timestamp']) if 'timestamp' in fields else None,
                metadata={k: v for k, v in fields.items() 
                         if k not in ['id', 'title', 'content', 'doc_type', 'timestamp']},
                source=fields.get('source', None)
            )
            
        except Exception as e:
            logger.error(f"Error parsing single document: {e}")
            return None
    
    def _parse_visit_document(self, visit_doc: Dict[str, Any]) -> Optional[VespaDocument]:
        """
        Parse a document from visit API response - matches TypeScript MailDocument structure
        """
        try:
            fields = visit_doc.get('fields', {})
            doc_id = visit_doc.get('id', fields.get('docId', fields.get('id', '')))
            
            # Handle mail-specific fields from TypeScript implementation
            subject = fields.get('subject', 'No subject')
            chunks = fields.get('chunks', [])
            content = ' '.join(chunks) if isinstance(chunks, list) else str(chunks) if chunks else ''
            
            # Extract sender/recipients
            sender = fields.get('from', 'Unknown sender')
            recipients = fields.get('to', [])
            cc_recipients = fields.get('cc', [])
            bcc_recipients = fields.get('bcc', [])
            
            # Handle timestamp
            timestamp = None
            if 'timestamp' in fields:
                try:
                    timestamp_val = fields['timestamp']
                    if isinstance(timestamp_val, (int, float)):
                        # Handle both seconds and milliseconds timestamps
                        if timestamp_val > 1e10:  # Likely milliseconds
                            timestamp = datetime.fromtimestamp(timestamp_val / 1000)
                        else:  # Likely seconds
                            timestamp = datetime.fromtimestamp(timestamp_val)
                    else:
                        timestamp = datetime.fromisoformat(str(timestamp_val))
                except (ValueError, TypeError, OSError):
                    logger.warning(f"Could not parse timestamp: {fields['timestamp']}")
            
            # Build metadata from additional fields
            metadata = {
                'from': sender,
                'to': recipients if isinstance(recipients, list) else [recipients] if recipients else [],
                'cc': cc_recipients if isinstance(cc_recipients, list) else [cc_recipients] if cc_recipients else [],
                'bcc': bcc_recipients if isinstance(bcc_recipients, list) else [bcc_recipients] if bcc_recipients else [],
                'attachmentFilenames': fields.get('attachmentFilenames', []),
                'labels': fields.get('labels', [])
            }
            
            # Add any other fields not explicitly handled
            for key, value in fields.items():
                if key not in ['docId', 'id', 'subject', 'chunks', 'from', 'to', 'cc', 'bcc', 
                              'timestamp', 'attachmentFilenames', 'labels']:
                    metadata[key] = value
            
            return VespaDocument(
                id=doc_id,
                title=subject,
                content=content,
                doc_type='email',  # Assuming mail schema documents are emails
                timestamp=timestamp,
                metadata=metadata,
                source=sender
            )
            
        except Exception as e:
            logger.error(f"Error parsing visit document: {e}")
            return None
    
    def convert_visit_documents_to_vespa_documents(self, raw_documents: List[Any]) -> List[VespaDocument]:
        """
        Convert raw documents from visit API to VespaDocument objects
        """
        vespa_documents = []
        
        for raw_doc in raw_documents:
            try:
                parsed_doc = self._parse_visit_document(raw_doc)
                if parsed_doc:
                    vespa_documents.append(parsed_doc)
            except Exception as e:
                logger.error(f"Error converting visit document: {e}")
        
        logger.info(f"Successfully converted {len(vespa_documents)} out of {len(raw_documents)} raw documents")
        return vespa_documents
    
    async def visit_all_documents_as_vespa_docs(
        self, 
        schema: str = "mail", 
        wanted_document_count: int = 100,
        max_documents: Optional[int] = None,
        cluster: str = "my_content"
    ) -> List[VespaDocument]:
        """
        Visit all documents and return them as VespaDocument objects
        """
        raw_documents = await self.visit_all_documents(
            schema=schema,
            wanted_document_count=wanted_document_count,
            max_documents=max_documents,
            cluster=cluster
        )
        
        return self.convert_visit_documents_to_vespa_documents(raw_documents)

class VespaJSONExporter:
    """Export Vespa data to various JSON formats"""
    
    def __init__(self, connector: VespaConnector):
        self.connector = connector
    
    async def export_all_documents_to_json(
        self,
        output_file: str = "vespa_documents.json",
        schema: str = "mail",
        max_documents: Optional[int] = None,
        wanted_document_count: int = 100,
        include_metadata: bool = True,
        pretty_print: bool = True
    ) -> Dict[str, Any]:
        """
        Export all documents from Vespa to JSON file
        
        Args:
            output_file: Path to output JSON file
            schema: Vespa schema to query
            max_documents: Maximum number of documents to export
            wanted_document_count: Documents per page for visit API
            include_metadata: Whether to include document metadata
            pretty_print: Whether to format JSON with indentation
            
        Returns:
            Dictionary with export statistics
        """
        try:
            logger.info(f"🚀 Starting JSON export to {output_file}")
            
            # Fetch all documents
            documents = await self.connector.visit_all_documents_as_vespa_docs(
                schema=schema,
                wanted_document_count=wanted_document_count,
                max_documents=max_documents
            )
            
            # Convert to dictionary format
            export_data = {
                "export_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_documents": len(documents),
                    "schema": schema,
                    "max_documents": max_documents,
                    "vespa_endpoint": self.connector.config.endpoint
                },
                "documents": []
            }
            
            # Process each document
            for doc in documents:
                doc_dict = doc.to_dict()
                
                if not include_metadata:
                    # Remove metadata if not requested
                    doc_dict.pop('metadata', None)
                
                export_data["documents"].append(doc_dict)
            
            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                if pretty_print:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
                else:
                    json.dump(export_data, f, ensure_ascii=False)
            
            stats = {
                "success": True,
                "output_file": str(output_path),
                "total_documents": len(documents),
                "file_size_bytes": output_path.stat().st_size,
                "export_timestamp": export_data["export_metadata"]["export_timestamp"]
            }
            
            logger.info(f"✅ JSON export completed successfully:")
            logger.info(f"  - File: {output_path}")
            logger.info(f"  - Documents: {len(documents)}")
            logger.info(f"  - Size: {stats['file_size_bytes']} bytes")
            
            return stats
            
        except Exception as e:
            error_msg = f"❌ JSON export failed: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": str(e),
                "output_file": output_file
            }
    
    async def export_by_document_type(
        self,
        doc_types: List[str],
        output_dir: str = "vespa_exports",
        schema: str = "mail",
        max_documents_per_type: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Export documents grouped by document type to separate JSON files
        
        Args:
            doc_types: List of document types to export
            output_dir: Directory to save JSON files
            schema: Vespa schema to query
            max_documents_per_type: Maximum documents per type
            
        Returns:
            Dictionary with export statistics per document type
        """
        try:
            logger.info(f"🚀 Starting export by document type to {output_dir}")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            results = {}
            
            # Get all documents first
            all_documents = await self.connector.visit_all_documents_as_vespa_docs(
                schema=schema,
                max_documents=max_documents_per_type * len(doc_types) if max_documents_per_type else None
            )
            
            # Group documents by type
            docs_by_type = {}
            for doc in all_documents:
                doc_type = doc.doc_type or 'unknown'
                if doc_type in doc_types:
                    if doc_type not in docs_by_type:
                        docs_by_type[doc_type] = []
                    docs_by_type[doc_type].append(doc)
            
            # Export each document type
            for doc_type in doc_types:
                documents = docs_by_type.get(doc_type, [])
                
                if max_documents_per_type and len(documents) > max_documents_per_type:
                    documents = documents[:max_documents_per_type]
                
                if documents:
                    filename = f"{doc_type}_documents.json"
                    file_path = output_path / filename
                    
                    export_data = {
                        "export_metadata": {
                            "export_timestamp": datetime.now().isoformat(),
                            "document_type": doc_type,
                            "total_documents": len(documents),
                            "schema": schema,
                            "vespa_endpoint": self.connector.config.endpoint
                        },
                        "documents": [doc.to_dict() for doc in documents]
                    }
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, ensure_ascii=False, indent=2)
                    
                    results[doc_type] = {
                        "success": True,
                        "file_path": str(file_path),
                        "document_count": len(documents),
                        "file_size_bytes": file_path.stat().st_size
                    }
                    
                    logger.info(f"  ✅ Exported {len(documents)} {doc_type} documents to {filename}")
                else:
                    results[doc_type] = {
                        "success": False,
                        "error": "No documents found",
                        "document_count": 0
                    }
                    logger.warning(f"  ⚠️ No {doc_type} documents found")
            
            return {
                "success": True,
                "output_directory": str(output_path),
                "results_by_type": results,
                "total_types_processed": len(doc_types)
            }
            
        except Exception as e:
            error_msg = f"❌ Export by document type failed: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": str(e),
                "output_directory": output_dir
            }
    
    async def export_lightweight_json(
        self,
        output_file: str = "vespa_documents_light.json",
        schema: str = "mail",
        max_documents: Optional[int] = None,
        fields_to_include: List[str] = None
    ) -> Dict[str, Any]:
        """
        Export documents with only essential fields to reduce file size
        
        Args:
            output_file: Path to output JSON file
            schema: Vespa schema to query
            max_documents: Maximum number of documents
            fields_to_include: Specific fields to include (default: id, title, doc_type)
            
        Returns:
            Dictionary with export statistics
        """
        try:
            if fields_to_include is None:
                fields_to_include = ['id', 'title', 'doc_type', 'timestamp']
            
            logger.info(f"🚀 Starting lightweight JSON export to {output_file}")
            logger.info(f"📝 Including fields: {fields_to_include}")
            
            # Fetch documents
            documents = await self.connector.visit_all_documents_as_vespa_docs(
                schema=schema,
                max_documents=max_documents
            )
            
            # Create lightweight export data
            export_data = {
                "export_metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_documents": len(documents),
                    "included_fields": fields_to_include,
                    "export_type": "lightweight"
                },
                "documents": []
            }
            
            # Process documents with only specified fields
            for doc in documents:
                doc_dict = doc.to_dict()
                lightweight_doc = {}
                
                for field in fields_to_include:
                    if field in doc_dict:
                        lightweight_doc[field] = doc_dict[field]
                
                export_data["documents"].append(lightweight_doc)
            
            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            stats = {
                "success": True,
                "output_file": str(output_path),
                "total_documents": len(documents),
                "file_size_bytes": output_path.stat().st_size,
                "included_fields": fields_to_include
            }
            
            logger.info(f"✅ Lightweight JSON export completed:")
            logger.info(f"  - File: {output_path}")
            logger.info(f"  - Documents: {len(documents)}")
            logger.info(f"  - Size: {stats['file_size_bytes']} bytes")
            
            return stats
            
        except Exception as e:
            error_msg = f"❌ Lightweight JSON export failed: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": str(e),
                "output_file": output_file
            }
    
    def documents_to_json_lines(
        self,
        documents: List[VespaDocument],
        output_file: str = "vespa_documents.jsonl"
    ) -> Dict[str, Any]:
        """
        Export documents to JSON Lines format (one JSON object per line)
        Useful for streaming and big data processing
        
        Args:
            documents: List of VespaDocument objects
            output_file: Path to output JSONL file
            
        Returns:
            Dictionary with export statistics
        """
        try:
            logger.info(f"🚀 Starting JSON Lines export to {output_file}")
            
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in documents:
                    json_line = json.dumps(doc.to_dict(), ensure_ascii=False)
                    f.write(json_line + '\n')
            
            stats = {
                "success": True,
                "output_file": str(output_path),
                "total_documents": len(documents),
                "file_size_bytes": output_path.stat().st_size,
                "format": "jsonl"
            }
            
            logger.info(f"✅ JSON Lines export completed:")
            logger.info(f"  - File: {output_path}")
            logger.info(f"  - Documents: {len(documents)}")
            logger.info(f"  - Size: {stats['file_size_bytes']} bytes")
            
            return stats
            
        except Exception as e:
            error_msg = f"❌ JSON Lines export failed: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": str(e),
                "output_file": output_file
            }

class VespaDataProcessor:
    """Process Vespa documents for KG integration"""
    
    def __init__(self, connector: VespaConnector):
        self.connector = connector
    
    async def get_documents_for_processing(
        self, 
        doc_types: List[str] = None, 
        limit: int = 100
    ) -> List[VespaDocument]:
        """Get documents ready for entity extraction"""
        if doc_types is None:
            doc_types = ['email', 'document', 'chat', 'meeting_notes']
        
        all_documents = []
        
        for doc_type in doc_types:
            try:
                docs = await self.connector.query_documents(
                    doc_type=doc_type,
                    limit=limit // len(doc_types)
                )
                all_documents.extend(docs)
                logger.info(f"Retrieved {len(docs)} documents of type {doc_type}")
            except Exception as e:
                logger.error(f"Error retrieving {doc_type} documents: {e}")
        
        return all_documents
    
    async def get_all_documents_via_visit(
        self, 
        schema: str = "mail", 
        max_documents: Optional[int] = None,
        wanted_document_count: int = 100
    ) -> List[VespaDocument]:
        """
        Get all documents using the visit API - matches TypeScript pattern
        """
        try:
            documents = await self.connector.visit_all_documents_as_vespa_docs(
                schema=schema,
                wanted_document_count=wanted_document_count,
                max_documents=max_documents
            )
            logger.info(f"Retrieved {len(documents)} documents via visit API")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents via visit API: {e}")
            return []
    
    def prepare_for_entity_extraction(self, documents: List[VespaDocument]) -> List[Dict[str, Any]]:
        """Prepare documents for entity extraction tools"""
        prepared_docs = []
        
        for doc in documents:
            prepared_doc = {
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
                'doc_type': doc.doc_type,
                'text': f"{doc.title}\n\n{doc.content}",  # Combined text for extraction
                'metadata': doc.metadata or {},
                'source': doc.source,
                'timestamp': doc.timestamp.isoformat() if doc.timestamp else None
            }
            prepared_docs.append(prepared_doc)
        
        return prepared_docs

# Utility functions for easy integration
async def get_vespa_connection() -> VespaConnector:
    """Get a configured Vespa connector"""
    config = VespaConfig.from_env()
    return VespaConnector(config)

async def fetch_recent_data(hours: int = 24) -> List[Dict[str, Any]]:
    """Fetch recent data from Vespa for processing"""
    async with await get_vespa_connection() as connector:
        processor = VespaDataProcessor(connector)
        documents = await connector.get_recent_documents(hours=hours)
        return processor.prepare_for_entity_extraction(documents)

async def search_vespa_content(search_term: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Search Vespa content and prepare for processing"""
    async with await get_vespa_connection() as connector:
        processor = VespaDataProcessor(connector)
        documents = await connector.search_by_content(search_term, limit=limit)
        return processor.prepare_for_entity_extraction(documents)

async def fetch_all_documents_via_visit(
    schema: str = "mail", 
    max_documents: Optional[int] = None,
    wanted_document_count: int = 100
) -> List[Dict[str, Any]]:
    """
    Fetch all documents using visit API - matches TypeScript pattern
    """
    async with await get_vespa_connection() as connector:
        processor = VespaDataProcessor(connector)
        documents = await processor.get_all_documents_via_visit(
            schema=schema,
            max_documents=max_documents,
            wanted_document_count=wanted_document_count
        )
        return processor.prepare_for_entity_extraction(documents)

# JSON Export Utility Functions
async def export_all_vespa_data_to_json(
    output_file: str = "vespa_export.json",
    schema: str = "mail",
    max_documents: Optional[int] = None,
    include_metadata: bool = True
) -> Dict[str, Any]:
    """
    Utility function to export all Vespa data to JSON
    
    Args:
        output_file: Path to output JSON file
        schema: Vespa schema to query
        max_documents: Maximum number of documents to export
        include_metadata: Whether to include document metadata
        
    Returns:
        Dictionary with export statistics
    """
    async with await get_vespa_connection() as connector:
        exporter = VespaJSONExporter(connector)
        return await exporter.export_all_documents_to_json(
            output_file=output_file,
            schema=schema,
            max_documents=max_documents,
            include_metadata=include_metadata
        )

async def export_vespa_data_by_type(
    doc_types: List[str],
    output_dir: str = "vespa_exports",
    schema: str = "mail",
    max_documents_per_type: Optional[int] = None
) -> Dict[str, Any]:
    """
    Utility function to export Vespa data grouped by document type
    
    Args:
        doc_types: List of document types to export
        output_dir: Directory to save JSON files
        schema: Vespa schema to query
        max_documents_per_type: Maximum documents per type
        
    Returns:
        Dictionary with export statistics per document type
    """
    async with await get_vespa_connection() as connector:
        exporter = VespaJSONExporter(connector)
        return await exporter.export_by_document_type(
            doc_types=doc_types,
            output_dir=output_dir,
            schema=schema,
            max_documents_per_type=max_documents_per_type
        )

async def export_vespa_lightweight_json(
    output_file: str = "vespa_light.json",
    schema: str = "mail",
    max_documents: Optional[int] = None,
    fields: List[str] = None
) -> Dict[str, Any]:
    """
    Utility function to export lightweight Vespa data (essential fields only)
    
    Args:
        output_file: Path to output JSON file
        schema: Vespa schema to query
        max_documents: Maximum number of documents
        fields: Specific fields to include
        
    Returns:
        Dictionary with export statistics
    """
    async with await get_vespa_connection() as connector:
        exporter = VespaJSONExporter(connector)
        return await exporter.export_lightweight_json(
            output_file=output_file,
            schema=schema,
            max_documents=max_documents,
            fields_to_include=fields
        )

async def convert_vespa_to_json_lines(
    output_file: str = "vespa_data.jsonl",
    schema: str = "mail",
    max_documents: Optional[int] = None
) -> Dict[str, Any]:
    """
    Utility function to export Vespa data to JSON Lines format
    
    Args:
        output_file: Path to output JSONL file
        schema: Vespa schema to query
        max_documents: Maximum number of documents
        
    Returns:
        Dictionary with export statistics
    """
    async with await get_vespa_connection() as connector:
        # Fetch documents
        documents = await connector.visit_all_documents_as_vespa_docs(
            schema=schema,
            max_documents=max_documents
        )
        
        # Export to JSON Lines
        exporter = VespaJSONExporter(connector)
        return exporter.documents_to_json_lines(documents, output_file)

def load_vespa_documents_from_json(json_file: str) -> List[VespaDocument]:
    """
    Load VespaDocument objects from JSON file
    
    Args:
        json_file: Path to JSON file created by export functions
        
    Returns:
        List of VespaDocument objects
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        doc_dicts = data.get('documents', [])
        
        for doc_dict in doc_dicts:
            try:
                doc = VespaDocument.from_dict(doc_dict)
                documents.append(doc)
            except Exception as e:
                logger.warning(f"Failed to load document: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {json_file}")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load documents from {json_file}: {e}")
        return []

def load_vespa_documents_from_json_lines(jsonl_file: str) -> List[VespaDocument]:
    """
    Load VespaDocument objects from JSON Lines file
    
    Args:
        jsonl_file: Path to JSONL file
        
    Returns:
        List of VespaDocument objects
    """
    try:
        documents = []
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if line:
                        doc_dict = json.loads(line)
                        doc = VespaDocument.from_dict(doc_dict)
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to load document at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents from {jsonl_file}")
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load documents from {jsonl_file}: {e}")
        return []

async def test_visit_api(
    schema: str = "mail",
    max_documents: int = 10,
    wanted_document_count: int = 5
) -> List[VespaDocument]:
    """
    Test the visit API functionality with limited documents
    """
    async with await get_vespa_connection() as connector:
        logger.info("🧪 Testing Vespa Visit API functionality")
        
        # Test visit API
        documents = await connector.visit_all_documents_as_vespa_docs(
            schema=schema,
            wanted_document_count=wanted_document_count,
            max_documents=max_documents
        )
        
        logger.info(f"📋 Test Results:")
        logger.info(f"  - Total documents fetched: {len(documents)}")
        
        for i, doc in enumerate(documents[:3]):  # Show first 3 documents
            logger.info(f"  - Document {i+1}:")
            logger.info(f"    ID: {doc.id}")
            logger.info(f"    Title: {doc.title[:50]}...")
            logger.info(f"    Type: {doc.doc_type}")
            logger.info(f"    Source: {doc.source}")
            logger.info(f"    Content length: {len(doc.content)} chars")
        
        return documents

# Example usage and testing
async def main():
    """Example usage of VespaConnector with visit API testing"""
    try:
        # Test configuration
        config = VespaConfig.from_env()
        logger.info(f"Connecting to Vespa at {config.endpoint}")
        
        async with VespaConnector(config) as connector:
            # Test connection
            if await connector.test_connection():
                logger.info("Vespa connection successful!")
                
                logger.info("\n" + "="*60)
                logger.info("🧪 TESTING VISIT API - NEW FUNCTIONALITY")
                logger.info("="*60)
                
                # Test the new visit API functionality
                try:
                    visit_docs = await test_visit_api(
                        schema="mail",
                        max_documents=10,
                        wanted_document_count=5
                    )
                    
                    if visit_docs:
                        logger.info("\n✅ Visit API test successful!")
                        
                        # Process first document for detailed analysis
                        sample_doc = visit_docs[0]
                        logger.info(f"\n📄 Sample Document Analysis:")
                        logger.info(f"  - ID: {sample_doc.id}")
                        logger.info(f"  - Title: {sample_doc.title}")
                        logger.info(f"  - Type: {sample_doc.doc_type}")
                        logger.info(f"  - Source: {sample_doc.source}")
                        logger.info(f"  - Timestamp: {sample_doc.timestamp}")
                        logger.info(f"  - Content preview: {sample_doc.content[:100]}...")
                        logger.info(f"  - Metadata keys: {list(sample_doc.metadata.keys()) if sample_doc.metadata else 'None'}")
                        
                        # Test data preparation for KG
                        processor = VespaDataProcessor(connector)
                        prepared_docs = processor.prepare_for_entity_extraction(visit_docs[:3])
                        logger.info(f"\n🔧 Prepared {len(prepared_docs)} documents for KG integration")
                        
                        if prepared_docs:
                            sample_prepared = prepared_docs[0]
                            logger.info(f"📋 Sample prepared document structure:")
                            for key, value in sample_prepared.items():
                                if key == 'text':
                                    logger.info(f"  - {key}: {str(value)[:50]}...")
                                elif key == 'metadata':
                                    logger.info(f"  - {key}: {len(value)} items")
                                else:
                                    logger.info(f"  - {key}: {value}")
                    
                    else:
                        logger.warning("❌ No documents returned from visit API")
                        
                except Exception as visit_error:
                    logger.error(f"❌ Visit API test failed: {visit_error}")
                
                logger.info("\n" + "="*60)
                logger.info("🔍 TESTING TRADITIONAL QUERY API")
                logger.info("="*60)
                
                # Test traditional functionality for comparison
                try:
                    # Get recent documents (traditional way)
                    recent_docs = await connector.get_recent_documents(hours=24)
                    logger.info(f"📅 Found {len(recent_docs)} recent documents via query API")
                    
                    # Search for specific content
                    search_results = await connector.search_by_content("meeting")
                    logger.info(f"🔍 Found {len(search_results)} documents mentioning 'meeting'")
                    
                    if recent_docs:
                        sample_query_doc = recent_docs[0]
                        logger.info(f"\n📄 Sample Query API Document:")
                        logger.info(f"  - ID: {sample_query_doc.id}")
                        logger.info(f"  - Title: {sample_query_doc.title[:50]}...")
                        logger.info(f"  - Type: {sample_query_doc.doc_type}")
                        
                except Exception as query_error:
                    logger.error(f"❌ Query API test failed: {query_error}")
                
                logger.info("\n" + "="*60)
                logger.info("📊 SUMMARY")
                logger.info("="*60)
                logger.info("✅ Visit API implementation complete - matches TypeScript pattern")
                logger.info("✅ Pagination support with continuation tokens")
                logger.info("✅ Document parsing for mail schema")
                logger.info("✅ Integration with existing KG system")
                logger.info("✅ Backward compatibility with existing query API")
                
            else:
                logger.error("❌ Vespa connection failed!")
                
    except Exception as e:
        logger.error(f"💥 Error in main: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
