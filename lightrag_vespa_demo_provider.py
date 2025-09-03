# LightRAG Vespa Data Demo with Gemini Provider
# Uses exported Vespa data from JSON file with enhanced processing

import os
import gc
import torch
import numpy as np
import google.generativeai as genai
import json 
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status
from vespa_integration import VespaDocument, load_vespa_documents_from_json

import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

WORKING_DIR = "./data/vespa_emails_rag1"
VESPA_JSON_FILE = "vespa_complete_export_20250902_164758.json"

# Clean and recreate working directory
if os.path.exists(WORKING_DIR):
    import shutil
    shutil.rmtree(WORKING_DIR)

os.makedirs(WORKING_DIR, exist_ok=True)


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
        _gemini_provider = GeminiProvider(
            api_key=gemini_api_key,
            model_id="models/gemini-2.5-pro",  # Using latest model
            temperature=0.1,
            max_tokens=10000,
            system_prompt=None
        )
        print("Gemini Provider initialized successfully!")
    return _gemini_provider


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    try:
        provider = get_gemini_provider()
        
        # Combine prompts: system prompt, history, and user prompt
        if history_messages is None:
            history_messages = []

        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n"

        for msg in history_messages:
            # Each msg is expected to be a dict: {"role": "...", "content": "..."}
            combined_prompt += f"{msg['role']}: {msg['content']}\n"

        # Finally, add the new user prompt
        combined_prompt += f"user: {prompt}"

        # Call the Gemini provider
        response = await provider.generate_content(combined_prompt)
        
        if response:
            return response
        else:
            print(f"Warning: Gemini API returned empty response for prompt: {prompt[:100]}...")
            return "I apologize, but I cannot provide a response at this time."
        
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        return "I apologize, but I encountered an error while processing your request."


# Global embedding model to avoid recreating it multiple times
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("Loading SentenceTransformer model (this will only happen once)...")
        _embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
        print("SentenceTransformer model loaded successfully!")
    return _embedding_model

def clear_mps_cache():
    """Clear MPS cache to free up memory"""
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
            print("MPS cache cleared")
        except Exception as e:
            print(f"Warning: Could not clear MPS cache: {e}")

async def embedding_func(texts: list[str]) -> np.ndarray:
    """Memory-efficient embedding function with batching"""
    model = get_embedding_model()
    
    # Determine optimal batch size based on available memory and text count
    if len(texts) <= 5:
        batch_size = len(texts)
    elif len(texts) <= 20:
        batch_size = 3
    else:
        batch_size = 8
    
    print(f"Processing {len(texts)} texts in batches of {batch_size}")
    
    all_embeddings = []
    
    try:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Clear cache before each batch
            clear_mps_cache()
            gc.collect()
            
            # Process the batch
            batch_embeddings = model.encode(
                batch_texts, 
                convert_to_numpy=True,
                batch_size=min(len(batch_texts), 4),  # Further limit internal batch size
                show_progress_bar=True if len(texts) > 10 else False
            )
            
            all_embeddings.append(batch_embeddings)
            
            # Clear cache after each batch
            clear_mps_cache()
            gc.collect()
    
    except Exception as e:
        print(f"Error during embedding: {e}")
        # Try to recover with smaller batches
        print("Attempting recovery with smaller batches...")
        
        all_embeddings = []
        for i, text in enumerate(texts):
            try:
                print(f"Processing text {i+1}/{len(texts)} individually")
                clear_mps_cache()
                gc.collect()
                
                single_embedding = model.encode([text], convert_to_numpy=True, batch_size=1)
                all_embeddings.append(single_embedding)
                
                clear_mps_cache()
                gc.collect()
            except Exception as single_e:
                print(f"Failed to process text {i+1}: {single_e}")
                # Create a zero embedding as fallback
                embedding_dim = 1024
                fallback_embedding = np.zeros((1, embedding_dim))
                all_embeddings.append(fallback_embedding)
    
    # Concatenate all embeddings
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        print(f"Successfully created embeddings with shape: {final_embeddings.shape}")
        return final_embeddings
    else:
        # Fallback: return zero embeddings
        embedding_dim = 1024
        return np.zeros((len(texts), embedding_dim))


class VespaDataProcessor:
    """Process Vespa documents for LightRAG integration"""
    
    def __init__(self, vespa_documents):
        self.documents = vespa_documents
        print(f"Initialized processor with {len(self.documents)} Vespa documents")
    
    def format_document_for_rag(self, doc: VespaDocument) -> str:
        """Format a VespaDocument for LightRAG processing"""
        
        # Extract metadata
        metadata = doc.metadata or {}
        sender = metadata.get('from', doc.source or 'Unknown')
        recipients = metadata.get('to', [])
        cc_recipients = metadata.get('cc', [])
        
        # Format recipients
        to_list = recipients if isinstance(recipients, list) else [recipients] if recipients else []
        cc_list = cc_recipients if isinstance(cc_recipients, list) else [cc_recipients] if cc_recipients else []
        
        # Build formatted text
        formatted_text = f"""
Title: {doc.title}
Timestamp: {doc.timestamp.isoformat() if doc.timestamp else 'Unknown'}

Content:
{doc.content}

---
"""
        return formatted_text.strip()
    
    def get_documents_by_type(self, doc_type: str = None):
        """Get documents filtered by type"""
        if doc_type:
            return [doc for doc in self.documents if doc.doc_type == doc_type]
        return self.documents
    
    def get_document_statistics(self):
        """Get statistics about the documents"""
        stats = {
            'total_documents': len(self.documents),
            'document_types': {},
            'date_range': {'earliest': None, 'latest': None},
            'senders': set(),
            'recipients': set()
        }
        
        for doc in self.documents:
            # Count by type
            doc_type = doc.doc_type or 'unknown'
            stats['document_types'][doc_type] = stats['document_types'].get(doc_type, 0) + 1
            
            # Track date range
            if doc.timestamp:
                if stats['date_range']['earliest'] is None or doc.timestamp < stats['date_range']['earliest']:
                    stats['date_range']['earliest'] = doc.timestamp
                if stats['date_range']['latest'] is None or doc.timestamp > stats['date_range']['latest']:
                    stats['date_range']['latest'] = doc.timestamp
            
            # Track senders and recipients
            if doc.source:
                stats['senders'].add(doc.source)
            
            if doc.metadata:
                to_list = doc.metadata.get('to', [])
                if isinstance(to_list, list):
                    stats['recipients'].update(to_list)
                elif to_list:
                    stats['recipients'].add(to_list)
        
        # Convert sets to lists for JSON serialization
        stats['senders'] = list(stats['senders'])
        stats['recipients'] = list(stats['recipients'])
        
        return stats


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=embedding_func,
        ),
        # Enhanced concurrent operations with provider's semaphore control
        llm_model_max_async=15,   # Increased since provider handles concurrency with semaphore
        embedding_func_max_async=6,  # Moderate for embedding processing
        max_parallel_insert=5,   # Conservative for document insertion
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def load_vespa_data(json_file: str):
    """Load Vespa documents from JSON file"""
    print(f"Loading Vespa data from {json_file}...")
    
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Vespa JSON file not found: {json_file}")
    
    # Load documents using the vespa_integration function
    documents = load_vespa_documents_from_json(json_file)
    
    if not documents:
        raise ValueError("No documents found in Vespa JSON file")
    
    print(f"Successfully loaded {len(documents)} documents from Vespa export")
    return documents


async def process_vespa_documents(rag, processor: VespaDataProcessor):
    """Process and insert Vespa documents into LightRAG"""
    
    documents = processor.documents
    print(f"Processing {len(documents)} Vespa documents for LightRAG...")
    
    # Get statistics
    stats = processor.get_document_statistics()
    print(f"\n📊 Document Statistics:")
    print(f"   Total documents: {stats['total_documents']}")
    print(f"   Document types: {stats['document_types']}")
    print(f"   Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    print(f"   Unique senders: {len(stats['senders'])}")
    print(f"   Unique recipients: {len(stats['recipients'])}")
    
    # Process in batches for memory efficiency
    batch_size = 10  # Conservative batch size
    successful_inserts = 0
    
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_end = min(i + batch_size, len(documents))
        
        print(f"\n📄 Processing batch {i//batch_size + 1}: documents {i+1} to {batch_end}")
        
        # Format documents for this batch
        formatted_texts = []
        for doc in batch_docs:
            try:
                formatted_text = processor.format_document_for_rag(doc)
                formatted_texts.append(formatted_text)
            except Exception as e:
                print(f"   ⚠️ Error formatting document {doc.id}: {e}")
                continue
        
        if not formatted_texts:
            print(f"   ⚠️ No valid documents in batch {i//batch_size + 1}")
            continue
        
        # Combine batch into single text for LightRAG
        batch_text = "\n\n" + ("="*80) + "\n\n".join(formatted_texts)
        
        # Clear memory before processing
        clear_mps_cache()
        gc.collect()
        
        try:
            print(f"   🔄 Inserting {len(formatted_texts)} documents into RAG...")
            rag.insert(batch_text)
            successful_inserts += len(formatted_texts)
            print(f"   ✅ Successfully processed {len(formatted_texts)} documents")
        except Exception as e:
            print(f"   ❌ Error inserting batch {i//batch_size + 1}: {e}")
            print("   Continuing with next batch...")
            continue
        
        # Clear memory after processing
        clear_mps_cache()
        gc.collect()
        
        # Small delay between batches
        await asyncio.sleep(2)
    
    print(f"\n✅ Successfully inserted {successful_inserts}/{len(documents)} documents into LightRAG")
    return successful_inserts


async def test_vespa_queries(rag, processor: VespaDataProcessor):
    """Test queries on the processed Vespa data"""
    
    print("\n🔍 Testing Queries on Vespa Data")
    print("=" * 50)
    
    # Get some statistics for informed queries
    stats = processor.get_document_statistics()
    
    # Enhanced query set based on email data
    test_queries = [
        ("What is this email dataset about?", "hybrid"),
        ("Who are the most frequent email senders?", "local"),
        ("What are the main topics discussed in these emails?", "global"),
        ("What companies or organizations are mentioned?", "hybrid"),
        ("Are there any important announcements or notifications?", "global"),
        ("What types of emails are in this dataset?", "local"),
        ("Who communicates with whom frequently?", "local"),
        ("What are the key relationships between people?", "global"),
    ]
    
    # Add specific queries based on actual data
    if stats['senders']:
        sample_sender = stats['senders'][0]
        test_queries.append((f"What emails were sent by {sample_sender}?", "local"))
    
    if stats['date_range']['earliest'] and stats['date_range']['latest']:
        test_queries.append((f"What happened between {stats['date_range']['earliest'].date()} and {stats['date_range']['latest'].date()}?", "hybrid"))
    
    successful_queries = 0
    
    for query, mode in test_queries:
        print(f"\n💭 Query: {query}")
        print(f"📊 Mode: {mode}")
        
        try:
            response = rag.query(
                query=query,
                param=QueryParam(mode=mode, top_k=5, response_type="single line"),
            )
            print(f"✅ Response: {response}")
            successful_queries += 1
        except Exception as e:
            print(f"❌ Error during query: {e}")
    
    print(f"\n📈 Query Results: {successful_queries}/{len(test_queries)} successful")


async def main():
    # Check if API key is available
    if not gemini_api_key:
        print("Error: GEMINI_API_KEY not found in environment variables!")
        print("Please set your Gemini API key in the .env file:")
        print("GEMINI_API_KEY=your_api_key_here")
        return

    # Set environment variables for better memory management
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable MPS memory limit
    
    # Clear any existing cache
    clear_mps_cache()
    gc.collect()

    print("=== LightRAG Vespa Data Demo ===")
    print("Using Vespa exported data with Gemini Provider")
    print(f"Working directory: {WORKING_DIR}")
    print(f"Vespa data file: {VESPA_JSON_FILE}")
    
    try:
        # Load Vespa data
        vespa_documents = load_vespa_data(VESPA_JSON_FILE)
        processor = VespaDataProcessor(vespa_documents)
        
        # Initialize RAG instance
        print("\n🚀 Initializing LightRAG...")
        rag = await initialize_rag()
        
        # Process Vespa documents
        print("\n📊 Processing Vespa documents...")
        successful_inserts = await process_vespa_documents(rag, processor)
        
        if successful_inserts == 0:
            print("❌ No documents were successfully inserted. Exiting.")
            return
        
        print(f"\n📁 Knowledge graph created at: {WORKING_DIR}/graph_chunk_entity_relation.graphml")
        
        # Test queries
        await test_vespa_queries(rag, processor)
        
        # Save processing summary
        stats = processor.get_document_statistics()
        summary = {
            'processing_timestamp': datetime.now().isoformat(),
            'vespa_source_file': VESPA_JSON_FILE,
            'working_directory': WORKING_DIR,
            'documents_processed': successful_inserts,
            'total_documents': len(vespa_documents),
            'statistics': stats
        }
        
        summary_file = os.path.join(WORKING_DIR, "processing_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n📋 Processing summary saved to: {summary_file}")
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print(f"Please ensure {VESPA_JSON_FILE} exists in the current directory.")
        return
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 Demo Complete!")
    print(f"📁 Results saved in: {WORKING_DIR}")
    print("📊 Knowledge graph ready for queries!")


def run_main():
    """Wrapper function to run the async main function"""
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
