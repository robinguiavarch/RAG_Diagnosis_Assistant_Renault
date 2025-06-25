"""
LLM Judge Client: Enhanced Response Evaluation with Consistency Verification

This module provides an advanced LLM-based evaluation client for assessing RAG response
quality with built-in consistency checking and retry mechanisms. It supports deterministic
evaluation with configurable parameters for reliable and coherent response scoring across
multiple evaluation attempts.

Key components:
- OpenAI GPT-4 integration with optimized parameters for evaluation tasks
- Consistency verification between similar responses and their assigned scores
- Automatic retry mechanism for inconsistent evaluations
- Response similarity analysis using sequence matching algorithms
- Configurable evaluation parameters via YAML settings

Dependencies: openai, yaml, json, difflib, dotenv
Usage: Import LLMJudgeClient for automated response evaluation with consistency guarantees
"""

import os
import yaml
import json
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher
from typing import List

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class LLMJudgeClient:
    """LLM judge client with consistency verification and retry mechanisms"""
    
    def __init__(self):
        """
        Initialize LLM judge client with configuration loading and parameter setup.
        
        Loads evaluation parameters from settings.yaml with fallback to optimized
        defaults. Configures consistency checking and retry mechanisms for reliable
        evaluation performance.
        """
        # Load configuration from settings.yaml
        try:
            with open("config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            
            judge_cfg = settings.get("evaluation", {}).get("llm_judge", {})
            self.model = judge_cfg.get("model", "gpt-4o")
            self.temperature = judge_cfg.get("temperature", 0.0)
            self.max_tokens = judge_cfg.get("max_tokens", 500)
            
            # Enhanced parameters for consistency
            self.seed = judge_cfg.get("seed", 42)
            self.retry_on_inconsistency = judge_cfg.get("retry_on_inconsistency", True)
            self.max_retries = judge_cfg.get("max_retries", 2)
            self.similarity_threshold = judge_cfg.get("similarity_threshold", 0.9)
            self.max_score_difference = judge_cfg.get("max_score_difference", 0.3)
            
            print(f"LLMJudgeClient initialized:")
            print(f"   Model: {self.model}")
            print(f"   Temperature: {self.temperature}")
            print(f"   Max tokens: {self.max_tokens}")
            print(f"   Retry on inconsistency: {self.retry_on_inconsistency}")
            
        except FileNotFoundError:
            # Enhanced default configuration
            print("Warning: settings.yaml not found, using enhanced default configuration")
            self.model = "gpt-4o"
            self.temperature = 0.0
            self.max_tokens = 500
            self.seed = 42
            self.retry_on_inconsistency = True
            self.max_retries = 2
            self.similarity_threshold = 0.9
            self.max_score_difference = 0.3
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text responses.
        
        Uses sequence matching to determine textual similarity for consistency
        verification of evaluation results.
        
        Args:
            text1 (str): First text response
            text2 (str): Second text response
            
        Returns:
            float: Similarity ratio between 0.0 and 1.0
        """
        return SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
    
    def extract_responses_from_prompt(self, prompt: str) -> List[str]:
        """
        Extract 4 responses from evaluation prompt for similarity verification.
        
        Parses the prompt to identify and extract individual responses for
        consistency checking during evaluation validation.
        
        Args:
            prompt (str): Full evaluation prompt containing multiple responses
            
        Returns:
            List[str]: List of extracted response texts (maximum 4 responses)
        """
        try:
            responses = []
            lines = prompt.split('\n')
            current_response = ""
            capturing = False
            
            for line in lines:
                if "RESPONSE" in line and any(keyword in line for keyword in ["RAG", "CLASSIQUE", "DENSE", "SPARSE"]):
                    if current_response and capturing:
                        responses.append(current_response.strip())
                    current_response = ""
                    capturing = True
                    continue
                elif capturing and (line.startswith("RESPONSE") or "Provide JSON" in line or "EVALUATION CRITERIA" in line):
                    if current_response:
                        responses.append(current_response.strip())
                    if "Provide JSON" in line:
                        break
                    current_response = ""
                    capturing = True
                elif capturing and line.strip():
                    current_response += line.strip() + " "
            
            # Add final response if necessary
            if current_response and capturing:
                responses.append(current_response.strip())
            
            return responses[:4]  # Maximum 4 responses
            
        except Exception as e:
            print(f"Warning: Error extracting responses: {e}")
            return []
    
    def check_response_consistency(self, responses: List[str], scores: List[float]) -> bool:
        """
        Verify consistency between similar responses and their assigned scores.
        
        Checks that responses with high textual similarity receive scores within
        an acceptable range to ensure evaluation coherence.
        
        Args:
            responses (List[str]): List of response texts
            scores (List[float]): List of corresponding evaluation scores
            
        Returns:
            bool: True if evaluation is consistent, False otherwise
        """
        if len(responses) != 4 or len(scores) != 4:
            return True  # Skip if not 4 responses
            
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity_score = self.similarity(responses[i], responses[j])
                
                if similarity_score >= self.similarity_threshold:
                    score_diff = abs(scores[i] - scores[j])
                    if score_diff > self.max_score_difference:
                        print(f"Warning: Inconsistency detected:")
                        print(f"   Response similarity {i+1}-{j+1}: {similarity_score:.2f}")
                        print(f"   Score difference: {score_diff:.2f} > threshold {self.max_score_difference}")
                        print(f"   Scores: {scores[i]:.1f} vs {scores[j]:.1f}")
                        return False
        return True
    
    def parse_evaluation_for_check(self, llm_response: str) -> dict:
        """
        Quick parse evaluation response for consistency verification.
        
        Extracts JSON evaluation data for consistency checking without
        full error handling to enable rapid validation.
        
        Args:
            llm_response (str): Raw LLM response containing JSON evaluation
            
        Returns:
            dict: Parsed evaluation dictionary or empty dict on error
        """
        try:
            import re
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            json_match = re.search(json_pattern, llm_response, re.DOTALL | re.IGNORECASE)
            
            if json_match:
                json_str = json_match.group(1)
            else:
                json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
                json_match = re.search(json_pattern, llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    return {}
            
            return json.loads(json_str)
        except Exception:
            return {}
    
    def evaluate(self, prompt: str) -> str:
        """
        Evaluate responses with consistency verification and retry mechanism.
        
        Performs LLM-based evaluation with deterministic parameters and automatic
        retry on inconsistent results. Ensures reliable and coherent scoring across
        multiple evaluation attempts.
        
        Args:
            prompt (str): Evaluation prompt containing responses to assess
            
        Returns:
            str: JSON-formatted evaluation response from LLM
            
        Raises:
            Exception: If all retry attempts fail
        """
        for attempt in range(self.max_retries + 1):
            try:
                # LLM call with optimized deterministic parameters
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                llm_response = response.choices[0].message.content.strip()
                
                # Consistency verification if enabled
                if self.retry_on_inconsistency and attempt < self.max_retries:
                    try:
                        # Parse evaluation
                        evaluation = self.parse_evaluation_for_check(llm_response)
                        
                        # Extract scores
                        scores = [
                            evaluation.get("score_classic", 0),
                            evaluation.get("score_dense", 0),
                            evaluation.get("score_sparse", 0),
                            evaluation.get("score_dense_sc", 0)
                        ]
                        
                        # Extract responses from prompt
                        responses = self.extract_responses_from_prompt(prompt)
                        
                        # Consistency verification
                        if len(responses) == 4 and len(scores) == 4:
                            if self.check_response_consistency(responses, scores):
                                print(f"Consistent evaluation (attempt {attempt + 1})")
                                return llm_response
                            else:
                                print(f"Retry {attempt + 1}/{self.max_retries} - Inconsistency detected")
                                continue
                        else:
                            # If verification is not possible, accept result
                            return llm_response
                            
                    except Exception as e:
                        print(f"Warning: Consistency verification error: {e}")
                        return llm_response
                else:
                    # Final attempt or verification disabled
                    return llm_response
                
            except Exception as e:
                if attempt == self.max_retries:
                    error_response = json.dumps({
                        "error": f"LLM judge error after {self.max_retries + 1} attempts: {str(e)}",
                        "score_classic": 2.5,
                        "score_dense": 2.5,
                        "score_sparse": 2.5,
                        "score_dense_sc": 2.5,
                        "best_approach": "Evaluation unavailable (technical error)",
                        "comparative_analysis": "Error during automatic evaluation"
                    })
                    return error_response
                
                print(f"Attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback (should never occur)
        return '{"error": "Unexpected error in evaluator"}'


def create_judge_client() -> LLMJudgeClient:
    """
    Create enhanced judge client instance.
    
    Factory function for convenient instantiation of LLM judge client with
    all consistency verification and retry mechanisms properly configured.
    
    Returns:
        LLMJudgeClient: Fully configured judge client instance
    """
    return LLMJudgeClient()