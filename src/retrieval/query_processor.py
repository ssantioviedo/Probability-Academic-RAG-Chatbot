"""
Query Processor module.

Handles query expansion and refinement using the LLM to improve
retrieval performance. Uses a separate model from the main generation
to avoid rate limit conflicts.
"""

import os
import time
from typing import Optional
from google import genai
from google.genai import types
from src.utils.logger import get_logger, LoggerMixin

# Default model for query rewriting (lightweight, separate quota)
DEFAULT_REWRITE_MODEL = "gemini-2.0-flash"


class QueryProcessor(LoggerMixin):
    """
    Process and expand user queries using a dedicated LLM model.
    
    Uses a separate model from the main generation pipeline to avoid
    rate limit conflicts.
    """
    
    def __init__(self, api_key: str, rewrite_model: Optional[str] = None):
        """
        Initialize the QueryProcessor with its own API client.
        
        Args:
            api_key: Google API key.
            rewrite_model: Model name for query rewriting. 
                          Defaults to QUERY_REWRITE_MODEL env var or gemini-2.5-flash.
        """
        self.model_name = rewrite_model or os.getenv("QUERY_REWRITE_MODEL", DEFAULT_REWRITE_MODEL)
        self.client = genai.Client(api_key=api_key)
        self.logger.info(f"Initializing QueryProcessor with model: {self.model_name}")
        
    def generate_optimized_query(self, original_query: str, chat_history: Optional[list[dict]] = None) -> str:
        """
        Rewrite the user query to be more suitable for semantic search.
        
        Args:
            original_query: The raw user query.
            chat_history: Optional chat history for context.
            
        Returns:
            The optimized query string.
        """
        if not original_query.strip():
            return original_query
            
        # Build the system instruction for query rewriting
        system_instruction = (
            "You are an expert Query Optimizer for an academic RAG system specialized in Probability and Statistics. "
            "Your task is to REWRITE the input query to be precise, academic, and optimized for semantic search. "
            "\n\nRules:"
            "\n1. **Intent Preservation (CRITICAL)**: If the user asks for a PROOF ('demostración', 'proof'), your rewritten query MUST focus on the proof (e.g., 'proof of X'). "
            "   If they ask for EXAMPLES ('ejemplo'), focus on examples. "
            "   DO NOT change the focus! converting a proof request into a general topic search is a FAILURE."
            "\n2. **Translation**: ALWAYS TRANSLATE the query to English for the search index. "
            "   (e.g., 'demostración ley débil' -> 'Weak Law of Large Numbers proof'). "
            "   Your output MUST BE ENGLISH. If the input is Spanish, TRANSLATE IT."
            "\n3. **Context**: Interpret all terms within Probability Theory and Statistics."
            "\n4. **Acronyms**: Expand standard acronyms (CLT, LLN, MGF, PDF, CDF). "
            "\n5. **Format**: Output ONLY the rewritten English query. No explanations."
        )
        
        if chat_history:
            # Format history for context
            history_text = ""
            for msg in chat_history[-3:]: # Last 3 turns
                role = "USER" if msg["role"] == "user" else "ASSISTANT"
                content = msg["content"][:200] + "..." if len(msg["content"]) > 200 else msg["content"]
                history_text += f"{role}: {content}\n"
            
            user_prompt = f"CHAT_HISTORY:\n{history_text}\n\nCURRENT_QUERY:\n{original_query}\n\nREWRITTEN_QUERY:"
            
            system_instruction += (
                "\n6. **Conversation Context (MANDATORY)**: The user often refers to previous topics (e.g., 'and the proof?', 'how to calculate it?'). "
                "   You MUST resolve these references using CHAT_HISTORY. "
                "   Rewrite the query to be fully **SELF-CONTAINED**. "
                "   Example: History='What is CLT?', User='proof of it' -> Rewritten='Central Limit Theorem proof'. "
                "   DO NOT return 'proof of it'. The search engine has no memory."
            )
        else:
            user_prompt = f"QUERY: {original_query}\n\nREWRITTEN_QUERY:"

        # Retry logic for 429 Rate Limits
        max_retries = 3
        retry_delay = 5
        
        # Disable safety settings to prevent silent blocks on academic content
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE,
            ),
        ]
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Optimizing query: {original_query} (Attempt {attempt+1})")
                
                rewrite_config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                    max_output_tokens=200,
                    top_p=0.95,
                    safety_settings=safety_settings,
                )
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=rewrite_config,
                )
                
                optimized_query = response.text.strip()
                
                # Strip surrounding quotes if the LLM wrapped the query
                if optimized_query.startswith('"') and optimized_query.endswith('"'):
                    optimized_query = optimized_query[1:-1].strip()
                if optimized_query.startswith("'") and optimized_query.endswith("'"):
                    optimized_query = optimized_query[1:-1].strip()
                
                # Sanity check: if response is too long or empty, fallback
                if not optimized_query or len(optimized_query) > max(300, len(original_query) * 10):
                    self.logger.warning(f"Optimization produced suspicious result: {optimized_query}")
                    return original_query
                    
                self.logger.info(f"Query optimized: '{original_query}' -> '{optimized_query}'")
                return optimized_query
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str:
                    if attempt < max_retries - 1:
                        wait = retry_delay * (2 ** attempt)  # 5s, 10s, 20s
                        self.logger.warning(f"Rate limit hit during query expansion. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        self.logger.error("Query expansion failed after retries due to rate limit.")
                        return original_query
                
                self.logger.error(f"Query optimization failed: {e}")
                return original_query
        
        return original_query

    def classify_scope(self, query: str) -> bool:
        """
        Check if a query is within scope (probability/statistics).
        
        Uses a lightweight LLM call to classify whether the query
        is related to probability theory or statistics.
        
        Args:
            query: The user's query (original or optimized).
            
        Returns:
            True if the query is in-scope (probability/statistics).
        """
        system_instruction = (
            "You are a classifier for an academic RAG system about Probability and Statistics. "
            "Given a user query, determine if it is related to probability theory, statistics, "
            "stochastic processes, random variables, distributions, or mathematical statistics. "
            "\n\nRespond with ONLY 'YES' or 'NO'. Nothing else."
        )
        
        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
                max_output_tokens=5,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"Is this query about probability or statistics?\n\nQUERY: {query}",
                config=config,
            )
            
            answer = response.text.strip().upper()
            is_in_scope = answer.startswith("YES")
            self.logger.info(f"Scope classification for '{query[:50]}': {'IN-SCOPE' if is_in_scope else 'OUT-OF-SCOPE'}")
            return is_in_scope
            
        except Exception as e:
            self.logger.warning(f"Scope classification failed: {e}. Assuming in-scope.")
            return True  # Default to in-scope to avoid blocking valid queries

    def verify_content_relevance(self, query: str, content: str) -> bool:
        """
        Verify if the retrieved content ACTUALLY matches the user intent.
        
        This detects 'False Positives' where the text uses all the right keywords
        but explicitly says the information is missing (e.g. 'proof is omitted', 
        'see reference X for details').
        
        Args:
            query: The user's query.
            content: The text snippet to verify.
            
        Returns:
            True if the content is relevant and likely contains the answer.
            False if the content is irrelevant or explicitly non-answering.
        """
        # Quick length check
        if len(content) < 50:
            return False
            
        system_instruction = (
            "You are an expert Relevance Verifier for a Probability Retrieval System.\n"
            "Your Task: Check if the 'Retrieved Text' contains the answer to the 'User Query'.\n\n"
            "Step 1: Determine the User's Intent (Definition, Proof, Example, or General Concept).\n"
            "Step 2: Apply the rules for that intent.\n\n"
            "RULES:\n"
            "   - **RELEVANT** if the text defines, explains, discusses, OR **APPLIES** the concept.\n"
            "   - **RELEVANT** if the text provides examples, problems, or historical context.\n"
            "   - **RELEVANT** if the text mentions the term in a SECTION HEADER (e.g., 'Chapter 5: CLT'), even if the text itself is introductory.\n"
            "   - **IGNORE** phrases like 'proof omitted' or 'results stated without proof' — these are still valid context.\n"
            "   - **IRRELEVANT** ONLY if the text is completely unrelated or triggers a false positive (e.g. matching 'Normal' in 'Normal School' vs 'Normal Distribution').\n\n"
            "CRITICAL EXAMPLES:\n"
            "Query: 'CLT definition'\nText: 'The CLT states that... Proof is omitted.' -> RELEVANT\n"
            "Query: 'CLT definition'\nText: 'We use the CLT to approximate the binomial...' -> RELEVANT (Application is good context)\n"
            "Query: 'CLT proof'\nText: 'The CLT states that... Proof is omitted.' -> RELEVANT (Partial match is better than nothing, do not discard)\n\n"
            "OUTPUT:\n"
            "Respond ONLY with 'RELEVANT' or 'IRRELEVANT'. Do not explain."
        )
        
        # Retry logic for verification
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                    max_output_tokens=20, # Increased slightly
                )
                
                prompt = f"USER QUERY: {query}\n\nRETRIEVED TEXT:\n{content[:2000]}\n\nVERDICT:"
                
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                
                answer = response.text.strip().upper()
                
                # Log verdict
                self.logger.debug(f"Relevance check [{attempt+1}]: {answer} for '{query[:20]}...'")
                
                # "IRRELEVANT" contains "RELEVANT", so we must check for "IRRELEVANT" first
                if "IRRELEVANT" in answer:
                     return False
                elif "RELEVANT" in answer:
                     return True
                else:
                    # Ambiguous response? Default to True to be safe
                    self.logger.warning(f"Ambiguous relevance verdict: {answer}. Assuming Relevant.")
                    return True
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource_exhausted" in error_str:
                    if attempt < max_retries - 1:
                        wait = retry_delay * (2 ** attempt)
                        self.logger.warning(f"Rate limit during verification. Retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    else:
                        self.logger.error("Relevance verification failed (Rate Limit). Assuming Relevant to avoid data loss.")
                        return True
                
                self.logger.warning(f"Relevance verification error: {e}. Assuming relevant.")
                return True
        
        return True
