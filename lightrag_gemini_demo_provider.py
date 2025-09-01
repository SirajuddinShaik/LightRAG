# LightRAG Gemini Demo with Custom Provider
# Uses google-generativeai with enhanced retry logic and concurrency control

import os
import gc
import torch
import numpy as np
import google.generativeai as genai
import json 
import asyncio
from dotenv import load_dotenv
from lightrag.utils import EmbeddingFunc
from lightrag import LightRAG, QueryParam
from sentence_transformers import SentenceTransformer
from lightrag.kg.shared_storage import initialize_pipeline_status

import nest_asyncio

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

WORKING_DIR = "./data/emails_provider"

if os.path.exists(WORKING_DIR):
    import shutil
    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


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
        # print(f"Prompt: {prompt}")
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

    print("=== LightRAG Gemini Provider Demo ===")
    print("Using enhanced Gemini Provider with retry logic and concurrency control")
    print(f"Working directory: {WORKING_DIR}")
    
    # Initialize RAG instance
    rag = await initialize_rag()
    
    # Check if email.txt exists
    file_path = "email.txt"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found!")
        print("Please ensure email.txt exists in the current directory.")
        return
    
    print("Reading and processing document...")
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().split("Email ID:")

    print(f"{len(text)} emails found in the document.")

    print("Inserting document into LightRAG...")
    
    # Process in smaller batches with memory cleanup
    batch_size = 12  # Conservative batch size for provider testing
    
    for i in range(0, len(text), batch_size):
        j = min(i + batch_size, len(text))
        input_text = "Email ID:".join(text[i:j])
        print(f"Inserting emails {i + 1} to {j} into the RAG system...")
        
        # Clear memory before each batch
        clear_mps_cache()
        gc.collect()
        
        try:
            rag.insert(input_text, split_by_character="Email ID:")
            print(f"✓ Successfully processed emails {i + 1} to {j}")
        except Exception as e:
            print(f"✗ Error processing emails {i + 1} to {j}: {e}")
            print("Continuing with next batch...")
            continue
        
        # Clear memory after each batch
        clear_mps_cache()
        gc.collect()
        
        # Small delay between batches to prevent rate limiting
        await asyncio.sleep(2)

    print(f"Document processed! GraphML file created at: {WORKING_DIR}/graph_chunk_entity_relation.graphml")
    
    print("\n=== Testing Queries ===")
    
    # Test different query modes
    test_queries = [
        ("What is this data about?", "hybrid"),
        ("Who are the main people mentioned?", "local"), 
        ("What are the key relationships?", "global")
    ]
    
    for query, mode in test_queries:
        print(f"\nQuery: {query} (mode: {mode})")
        try:
            response = rag.query(
                query=query,
                param=QueryParam(mode=mode, top_k=5, response_type="single line"),
            )
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error during query: {e}")

    print("\n=== Demo Complete ===")
    print(f"Results saved in: {WORKING_DIR}")


def run_main():
    """Wrapper function to run the async main function"""
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
