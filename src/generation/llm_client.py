"""
Gemini LLM Client module for response generation.

Provides a wrapper around Google's GenAI API
for generating responses based on retrieved context.

Features:
- Automatic retry with exponential backoff for rate limits
- Response caching to avoid redundant API calls
- Request rate limiting to stay within API quotas
"""

import time
import hashlib
from collections import OrderedDict
from threading import Lock
from typing import Optional
from dataclasses import dataclass, field

from google import genai
from google.genai import types

from src.utils.logger import get_logger, LoggerMixin
from src.generation.prompts import PromptTemplates, FormattedPrompt


# Rate limiting configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # Start with 2 seconds (paid tier is more lenient)
MAX_RETRY_DELAY = 60  # Max 1 minute between retries

# Cache configuration
CACHE_MAX_SIZE = 50  # Max cached responses
CACHE_TTL_SECONDS = 3600  # 1 hour TTL

# Rate limiting: requests per minute (paid tier = 2000 RPM, but be conservative)
REQUESTS_PER_MINUTE = 60  # 60 RPM = plenty of headroom
MIN_REQUEST_INTERVAL = 60.0 / REQUESTS_PER_MINUTE  # 1 second between requests


@dataclass
class CacheEntry:
    """Entry in the response cache."""
    response: 'GenerationResult'
    timestamp: float
    
    def is_expired(self) -> bool:
        """Check if this cache entry has expired."""
        return time.time() - self.timestamp > CACHE_TTL_SECONDS


class ResponseCache:
    """
    LRU Cache for LLM responses with TTL expiration.
    
    Caches responses by query+context hash to avoid
    redundant API calls for similar questions.
    """
    
    def __init__(self, max_size: int = CACHE_MAX_SIZE):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()
    
    def _make_key(self, query: str, context_hash: str) -> str:
        """Create a cache key from query and context."""
        combined = f"{query.lower().strip()}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, query: str, context: str) -> Optional['GenerationResult']:
        """Get cached response if available and not expired."""
        context_hash = hashlib.md5(context[:1000].encode()).hexdigest()
        key = self._make_key(query, context_hash)
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    return entry.response
                else:
                    # Remove expired entry
                    del self._cache[key]
        return None
    
    def put(self, query: str, context: str, response: 'GenerationResult') -> None:
        """Store a response in the cache."""
        context_hash = hashlib.md5(context[:1000].encode()).hexdigest()
        key = self._make_key(query, context_hash)
        
        with self._lock:
            # Remove oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(
                response=response,
                timestamp=time.time()
            )
    
    def clear(self) -> None:
        """Clear all cached responses."""
        with self._lock:
            self._cache.clear()


class RateLimiter:
    """
    Simple rate limiter to control API request frequency.
    
    Ensures minimum interval between requests to avoid
    hitting API rate limits.
    """
    
    def __init__(self, min_interval: float = MIN_REQUEST_INTERVAL):
        self._min_interval = min_interval
        self._last_request_time = 0.0
        self._lock = Lock()
    
    def wait_if_needed(self) -> float:
        """
        Wait if needed to respect rate limit.
        
        Returns:
            Time waited in seconds.
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            
            if elapsed < self._min_interval:
                wait_time = self._min_interval - elapsed
                time.sleep(wait_time)
                self._last_request_time = time.time()
                return wait_time
            
            self._last_request_time = now
            return 0.0


@dataclass
class GenerationResult:
    """
    Result of LLM generation.
    
    Attributes:
        answer: Generated response text.
        sources: List of source dictionaries.
        model: Model used for generation.
        prompt_tokens: Estimated prompt tokens.
        completion_tokens: Estimated completion tokens.
        generation_time: Time taken for generation.
        success: Whether generation was successful.
        error: Error message if generation failed.
    """
    answer: str
    sources: list[dict] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_time: float = 0.0
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "sources": self.sources,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "generation_time": self.generation_time,
            "success": self.success,
            "error": self.error,
        }


class GeminiClient(LoggerMixin):
    """
    Client for Google Gemini API interactions.
    
    Handles API configuration, request formatting, rate limiting,
    and error handling for LLM-based response generation.
    
    Attributes:
        model_name: Name of the Gemini model to use.
        temperature: Temperature setting for generation.
        max_tokens: Maximum tokens for response.
        api_key: Google API key.
    
    Example:
        >>> client = GeminiClient(api_key="your_key")
        >>> result = client.generate(
        ...     query="What is variance?",
        ...     context="Variance is defined as..."
        ... )
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.2,
        max_tokens: int = 2048
    ) -> None:
        """
        Initialize the Gemini client.
        
        Args:
            api_key: Google Generative AI API key.
            model_name: Gemini model to use.
            temperature: Generation temperature (0-1).
            max_tokens: Maximum response tokens.
        
        Raises:
            ValueError: If API key is empty.
        """
        if not api_key:
            raise ValueError(
                "Google API key is required. "
                "Set GOOGLE_API_KEY in your .env file."
            )
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize the client with the new google-genai library
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize prompt templates
        self.prompt_templates = PromptTemplates()
        
        # Initialize cache and rate limiter
        self._cache = ResponseCache()
        self._rate_limiter = RateLimiter()
        
        self.logger.info(
            f"GeminiClient initialized: model={model_name}, "
            f"temperature={temperature}"
        )
    
    def _create_generation_config(self) -> types.GenerateContentConfig:
        """
        Create generation configuration for the API.
        
        Returns:
            GenerateContentConfig object.
        """
        # Disable safety settings for academic content (textbooks might contain "violence" in examples etc)
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
        
        return types.GenerateContentConfig(
            system_instruction=self.prompt_templates.system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            top_p=0.95,
            top_k=40,
            safety_settings=safety_settings
        )
    
    def _handle_rate_limit(self, attempt: int) -> float:
        """
        Handle rate limit errors with exponential backoff.
        
        Args:
            attempt: Current retry attempt number.
            
        Returns:
            Delay time in seconds.
        """
        delay = min(INITIAL_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
        self.logger.warning(
            f"Rate limit hit (attempt {attempt + 1}/{MAX_RETRIES}). "
            f"Waiting {delay} seconds before retry..."
        )
        time.sleep(delay)
        return delay
    
    def generate(
        self,
        query: str,
        context: str,
        chat_history: Optional[list[dict]] = None,
        sources: Optional[list[dict]] = None,
        use_cache: bool = True,
        target_language: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate a response using the Gemini model.
        
        Features automatic caching, rate limiting, and retry with
        exponential backoff for rate limit errors.
        
        Args:
            query: User's question.
            context: Retrieved context from bibliography.
            chat_history: Optional conversation history.
            sources: Source metadata for citation.
            use_cache: Whether to use response cache.
            target_language: Target language code ('en' or 'es').
        
        Returns:
            GenerationResult with answer and metadata.
        """
        start_time = time.time()
        
        # Check cache first (if no chat history - history makes responses unique)
        if use_cache and not chat_history:
            cached = self._cache.get(query, context)
            if cached:
                self.logger.info(
                    f"Cache hit for query: {query[:50]}..."
                )
                # Update timing but keep cached response
                cached.generation_time = time.time() - start_time
                return cached
        
        # Format the prompt
        formatted_prompt = self.prompt_templates.format_query(
            question=query,
            context=context,
            chat_history=chat_history,
            target_language=target_language
        )
        
        self.logger.debug(
            f"Generating response for: {query[:50]}... "
            f"(prompt ~{formatted_prompt.token_estimate} tokens)"
        )
        
        # Apply rate limiting before making request
        wait_time = self._rate_limiter.wait_if_needed()
        if wait_time > 0:
            self.logger.debug(f"Rate limiter: waited {wait_time:.2f}s")
        
        # Attempt generation with retries
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=formatted_prompt.full_prompt,
                    config=self._create_generation_config()
                )
                
                # Robust response extraction
                answer = ""
                if response.text:
                   answer = response.text
                elif response.parts:
                   # Fallback manual join if .text property fails but parts exist
                   answer = "".join([p.text for p in response.parts if p.text])

                if not answer:
                    # Handle blocked or empty responses
                    reason = "Unknown"
                    try:
                         # Attempt to inspect finish reason if available
                         if response.candidates and response.candidates[0].finish_reason:
                             reason = response.candidates[0].finish_reason
                    except:
                        pass
                        
                    self.logger.warning(f"Response empty. Finish Reason: {reason}")
                    
                    answer = (
                        f"⚠️ The AI could not generate a response (Status: {reason}). "
                        "This might be due to safety filters or an internal model error."
                    )
                
                generation_time = time.time() - start_time
                
                self.logger.info(
                    f"Response generated in {generation_time:.2f}s"
                )
                
                result = GenerationResult(
                    answer=answer,
                    sources=sources or [],
                    model=self.model_name,
                    prompt_tokens=formatted_prompt.token_estimate,
                    completion_tokens=len(answer) // 4,  # Rough estimate
                    generation_time=generation_time,
                    success=True
                )
                
                # Cache successful response
                if use_cache and not chat_history:
                    self._cache.put(query, context, result)
                
                return result
                
            except Exception as e:
                error_str = str(e)
                error_lower = error_str.lower()
                
                # Detect rate limit / quota errors (429, RESOURCE_EXHAUSTED)
                is_rate_limit = any(x in error_lower for x in [
                    "429", "resource_exhausted", "quota", "rate"
                ])
                
                if is_rate_limit:
                    if attempt < MAX_RETRIES - 1:
                        self._handle_rate_limit(attempt)
                        continue
                    else:
                        self.logger.error(f"Rate limit exceeded after {MAX_RETRIES} retries: {e}")
                        return GenerationResult(
                            answer="",
                            success=False,
                            error=(
                                "⚠️ API rate limit exceeded. The free tier allows ~15 requests/minute. "
                                "Please wait 1-2 minutes and try again, or consider upgrading to a paid plan."
                            ),
                            generation_time=time.time() - start_time
                        )
                
                self.logger.error(f"Generation error: {e}")
                return GenerationResult(
                    answer="",
                    success=False,
                    error=f"Failed to generate response: {error_str}",
                    generation_time=time.time() - start_time
                )
        
        # Should not reach here, but just in case
        return GenerationResult(
            answer="",
            success=False,
            error="Maximum retries exceeded",
            generation_time=time.time() - start_time
        )
    
    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
        self.logger.info("Response cache cleared")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache._cache),
            "max_size": self._cache._max_size,
        }
    
    def test_connection(self) -> bool:
        """
        Test the API connection with a simple query.
        
        Returns:
            True if connection is successful.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Say 'Connection successful' and nothing else.",
                config=types.GenerateContentConfig(max_output_tokens=20)
            )
            success = bool(response.text)
            self.logger.info(f"API connection test: {'passed' if success else 'failed'}")
            return success
        except Exception as e:
            self.logger.error(f"API connection test failed: {e}")
            return False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }


def create_client_from_config(config) -> GeminiClient:
    """
    Create a GeminiClient from application config.
    
    Args:
        config: Config object with LLM settings.
    
    Returns:
        Configured GeminiClient instance.
    """
    return GeminiClient(
        api_key=config.google_api_key,
        model_name=config.llm_model,
        temperature=config.llm_temperature,
        max_tokens=config.llm_max_tokens
    )
