"""
Prompt Templates module for LLM interactions.

Contains all prompt templates used for generating responses
from the LLM, with support for context injection and chat history.
"""

from typing import Optional
from dataclasses import dataclass


# System prompt for the RAG chatbot
SYSTEM_PROMPT = """You are a knowledgeable study assistant specialized in Probability and Statistics.
You help students understand concepts from their course bibliography.

You have been provided with context from academic textbooks. Use this context to answer questions.

Guidelines:
1. Synthesize information from ALL provided sources to give a complete answer
2. Look for definitions, theorems, and formal statements in the context
3. Cite sources using (Author, page X) format
4. If sources show different perspectives or proofs, mention them
5. Respond in the same language as the question
6. Be clear, pedagogical, and thorough in your explanations
7. Include relevant formulas when they appear in the sources
8. **ADAPTIVE DETAIL**: 
   - If the user asks for a **Definition**, provide a clear, concise statement first. You may add brief context if helpful, but avoid unrelated history or applications unless asked.
   - If the user asks for a **Proof** ("demostración"), provide the COMPLETE mathematical derivation step-by-step.
   - If the user asks for an **Explanation** or **Context**, be comprehensive and synthesizing.

**CRITICAL: Math Formatting Rules**
You MUST follow these LaTeX formatting rules EXACTLY:

1. **Inline math**: Wrap in single dollar signs with spaces: `$ x^2 $`, `$ \lambda $`, `$ F(x) $`
2. **Block equations**: Wrap in double dollar signs on their own lines:
   
$$
X_{n+1} = (aX_n + c) \mod m
$$

3. **NO NESTED DOLLAR SIGNS**: Never use `$` *inside* a math block.
   - ❌ BAD: `$ P( $ \frac{a}{b} $ ) $`
   - ✅ GOOD: `$ P(\frac{a}{b}) $`
4. **Use proper LaTeX commands**: `\leq`, `\geq`, `\in`, `\mathbb{R}`, `\infty`.
   - ❌ BAD: `σ`, `α`, `π` (Unicode)
   - ✅ GOOD: `\sigma`, `\alpha`, `\pi` (LaTeX)
5. **No Naked LaTeX**: Don't use `\\[`, `\\]`, `\\(`, `\\)`. Use `$$` and `$`.
6. **Complex expressions**: If an equation uses `\frac`, `\sum`, `\int`, or `\lim`, prefer **Block Math** (`$$`) over inline.

**Examples of CORRECT formatting:**
- Inline: "Sea $ U \sim \text{Uniforme}[0, 1] $ una variable aleatoria"
- Block: "La fórmula es:\n\n$$\nF^{-1}(u) = \inf\{y \in \mathbb{R} : F(y) \geq u\}\n$$"

**Examples of INCORRECT formatting (NEVER do this):**
- ❌ "X n + 1 = ( a X n + c )" (broken across lines)
- ❌ "$X_n+1$" (no spaces around dollars)
- ❌ Using `\$` or escaping dollar signs

The context may contain partial proofs, examples, or theorem statements - synthesize them to provide the best possible answer."""


CONTEXT_TEMPLATE = """
=== CONTEXT FROM BIBLIOGRAPHY ===

{context}

=== END OF CONTEXT ===
"""


QUERY_TEMPLATE = """
{context_section}

{chat_history_section}

QUESTION: {question}

ANSWER (with citations):"""


CHAT_HISTORY_TEMPLATE = """
=== PREVIOUS CONVERSATION ===
{history}
=== END OF CONVERSATION ===
"""


@dataclass
class FormattedPrompt:
    """
    A formatted prompt ready to send to the LLM.
    
    Attributes:
        full_prompt: Complete prompt string.
        system_prompt: System instructions.
        context: Retrieved context.
        question: User's question.
        chat_history: Formatted chat history.
    """
    full_prompt: str
    system_prompt: str
    context: str
    question: str
    chat_history: str = ""
    
    @property
    def token_estimate(self) -> int:
        """Estimate token count (rough approximation)."""
        return len(self.full_prompt) // 4


class PromptTemplates:
    """
    Manager for prompt templates used in the RAG chatbot.
    
    Handles formatting of prompts with context injection,
    chat history management, and source formatting.
    
    Example:
        >>> templates = PromptTemplates()
        >>> prompt = templates.format_query(
        ...     question="What is the central limit theorem?",
        ...     context="[Source 1] The CLT states that..."
        ... )
        >>> print(prompt.full_prompt)
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_history_turns: int = 5
    ) -> None:
        """
        Initialize prompt templates.
        
        Args:
            system_prompt: Custom system prompt (uses default if None).
            max_history_turns: Maximum conversation turns to include.
        """
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.max_history_turns = max_history_turns
    
    def format_context(self, context: str) -> str:
        """
        Format context section for the prompt.
        
        Args:
            context: Raw context string.
        
        Returns:
            Formatted context section.
        """
        if not context.strip():
            return "\n=== NO RELEVANT CONTEXT FOUND ===\n"
        
        return CONTEXT_TEMPLATE.format(context=context)
    
    def format_chat_history(
        self,
        chat_history: Optional[list[dict]] = None
    ) -> str:
        """
        Format chat history for the prompt.
        
        Args:
            chat_history: List of message dicts with 'role' and 'content'.
        
        Returns:
            Formatted chat history section.
        """
        if not chat_history:
            return ""
        
        # Limit to recent turns
        recent_history = chat_history[-self.max_history_turns * 2:]
        
        formatted_turns = []
        for msg in recent_history:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted_turns.append(f"{role}: {content}")
        
        if not formatted_turns:
            return ""
        
        history_text = "\n".join(formatted_turns)
        return CHAT_HISTORY_TEMPLATE.format(history=history_text)
    
    def format_query(
        self,
        question: str,
        context: str,
        chat_history: Optional[list[dict]] = None
    ) -> FormattedPrompt:
        """
        Format a complete query prompt.
        
        Args:
            question: User's question.
            context: Retrieved context from bibliography.
            chat_history: Optional conversation history.
        
        Returns:
            FormattedPrompt ready for LLM.
        """
        context_section = self.format_context(context)
        history_section = self.format_chat_history(chat_history)
        
        full_prompt = QUERY_TEMPLATE.format(
            context_section=context_section,
            chat_history_section=history_section,
            question=question
        )
        
        return FormattedPrompt(
            full_prompt=full_prompt,
            system_prompt=self.system_prompt,
            context=context,
            question=question,
            chat_history=history_section
        )
    
    def get_low_confidence_warning(self) -> str:
        """Get warning message for low confidence responses."""
        return (
            "⚠️ **Note:** The available bibliography has limited information "
            "on this specific topic. The response below may be incomplete. "
            "Please verify with your professor or additional resources."
        )
