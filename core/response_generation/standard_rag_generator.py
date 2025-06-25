"""
Standard RAG Generator: Classic Retrieval-Augmented Generation Implementation

This module provides a standard RAG generation system using OpenAI models with strict
context management and configurable chunk limitations. It implements classic RAG
methodology with external prompt templates, token-aware passage selection, and
comprehensive error handling for reliable document-based response generation.

Key components:
- OpenAI GPT integration with configurable model parameters and token limits
- Strict context chunk limitation with hierarchical passage selection
- External prompt template loading with fallback mechanisms
- Token estimation and budget management for optimal context utilization
- Comprehensive generation statistics and performance monitoring

Dependencies: openai, yaml, pathlib, dotenv
Usage: Import OpenAIGenerator for standard RAG response generation with context-aware processing
"""

import os
import yaml
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class OpenAIGenerator:
    """
    Standard RAG generator with strict context management and configurable limitations.
    
    Implements classic retrieval-augmented generation using OpenAI models with
    comprehensive context management, token budgeting, and external configuration
    support for optimal performance and reliability.
    """
    
    def __init__(self, model: str = "gpt-4o", context_token_limit: int = 6000, max_tokens: int = 2000):
        """
        Initialize OpenAI generator with configurable parameters and settings loading.
        
        Sets up the generator with model configuration, token limits, and loads
        additional settings from external configuration files.
        
        Args:
            model (str): OpenAI model name to use for generation
            context_token_limit (int): Maximum tokens allowed for context
            max_tokens (int): Maximum tokens for generated response
        """
        self.model = model
        self.context_token_limit = context_token_limit
        self.max_tokens = max_tokens
        
        # Load max_context_chunks from settings.yaml
        try:
            with open("config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            gen_cfg = settings.get("generation", {})
            self.max_context_chunks = gen_cfg.get("max_context_chunks", 3)
            print(f"OpenAIGenerator (Standard RAG) configuration:")
            print(f"   Max chunks: {self.max_context_chunks}")
            print(f"   Token limit: {self.context_token_limit}")
        except Exception as e:
            print(f"Warning: Error loading settings.yaml: {e}, using default value")
            self.max_context_chunks = 3
        
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """
        Load prompt template from external file with fallback mechanism.
        
        Attempts to load the standard RAG prompt template from external file,
        with fallback to built-in template if file is not available.
        
        Returns:
            str: Prompt template with context and query placeholders
        """
        try:
            prompt_path = Path("config/prompts/standard_rag_prompt.txt")
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to original prompt if file absent
            return """Here are excerpts from technical documents:

{context}

Question: {query}

Answer clearly and precisely, strictly based on the provided context. 
If the answer is not in the context, explicitly state "Information not available in the provided context".
Answer in English only.
Answer:"""

    def _estimate_tokens(self, text: str) -> int:
        """
        Simple token count estimation for budget management.
        
        Provides approximate token count using word-based estimation for
        efficient context management without expensive tokenization calls.
        
        Args:
            text (str): Text to estimate token count for
            
        Returns:
            int: Estimated token count
        """
        return int(len(text.split()) * 0.75)

    def select_passages_with_limits(self, passages: List[str]) -> tuple:
        """
        Strict passage selection with respect for maximum context chunks.
        
        Implements hierarchical selection with chunk count priority followed by
        token budget validation to ensure optimal context utilization.
        
        Args:
            passages (List[str]): List of candidate passages for selection
            
        Returns:
            tuple: (selected_passages, total_tokens) - Final passages and token count
        """
        # Limit 1: Maximum number of chunks (priority)
        max_chunks = self.max_context_chunks
        selected_passages = passages[:max_chunks]
        
        # Limit 2: Token verification (safety)
        total_tokens = 0
        final_passages = []
        
        for i, passage in enumerate(selected_passages):
            token_estimate = self._estimate_tokens(passage)
            
            # Verify that this passage does not exceed limit
            if total_tokens + token_estimate > self.context_token_limit:
                print(f"Warning: Passage {i+1} ignored: token limit exceeded ({total_tokens + token_estimate} > {self.context_token_limit})")
                break
                
            final_passages.append(passage)
            total_tokens += token_estimate

        print(f"Standard RAG - Selected passages: {len(final_passages)}/{len(passages)} (tokens: {total_tokens})")
        return final_passages, total_tokens

    def generate_answer(self, query: str, passages: List[str]) -> str:
        """
        Generate response with strict respect for maximum context chunks.
        
        Performs complete RAG generation pipeline including passage selection,
        context assembly, prompt construction, and OpenAI API interaction with
        comprehensive error handling.
        
        Args:
            query (str): User question or query
            passages (List[str]): List of retrieved passages for context
            
        Returns:
            str: Generated response based on provided context
        """
        if not passages:
            return "Information not available in the provided context."
        
        # Strict selection with limits
        selected_passages, total_tokens = self.select_passages_with_limits(passages)
        
        if not selected_passages:
            return "Information not available in the provided context."

        context = "\n\n".join(selected_passages)

        # Use externalized template
        prompt = self.prompt_template.format(
            context=context,
            query=query
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=self.max_tokens,
                stop=["\n\n", "\nQuestion"]
            )
            
            generated_answer = response.choices[0].message.content.strip()
            
            # Diagnostic logging
            print(f"Standard RAG response generated: {len(generated_answer)} characters")
            print(f"Context used: {len(selected_passages)} passages, {total_tokens} tokens")
            
            return generated_answer

        except Exception as e:
            error_msg = f"OpenAI API error for Standard RAG: {str(e)}"
            print(error_msg)
            return error_msg

    def get_generation_stats(self) -> dict:
        """
        Return comprehensive generator statistics and configuration.
        
        Provides detailed information about generator configuration, limitations,
        and operational parameters for monitoring and debugging purposes.
        
        Returns:
            dict: Complete generator statistics and configuration
        """
        return {
            "model": self.model,
            "max_context_chunks": self.max_context_chunks,
            "context_token_limit": self.context_token_limit,
            "max_tokens": self.max_tokens,
            "generator_type": "standard_rag",
            "prompt_loaded": bool(self.prompt_template)
        }