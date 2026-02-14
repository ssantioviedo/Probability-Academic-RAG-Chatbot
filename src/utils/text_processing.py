r"""
Text processing utilities for RAG Chatbot.

Handles LaTeX normalization for Streamlit's KaTeX renderer.

Strategy:
  1. Convert \[...\] to $$...$$ and \(...\) to $...$
  2. Detect standalone math lines and wrap in $$ ... $$
  3. Find naked \commands in prose, expand to full expression, wrap in $ ... $
  4. Convert backtick-wrapped LaTeX (e.g. `\frac{a}{b}`) to $ ... $
"""
import re

# All LaTeX command names we detect. Sorted longest-first so
# e.g. "infty" matches before "in", "limsup" before "lim".
_LATEX_COMMANDS = sorted([
    # Greek
    "alpha","beta","gamma","delta","epsilon","varepsilon","zeta","eta",
    "theta","vartheta","iota","kappa","lambda","mu","nu","xi","pi","varpi",
    "rho","varrho","sigma","varsigma","tau","upsilon","phi","varphi","chi",
    "psi","omega","Gamma","Delta","Theta","Lambda","Xi","Pi","Sigma",
    "Upsilon","Phi","Psi","Omega",
    # Big operators
    "sum","prod","int","iint","iiint","oint","bigcup","bigcap",
    "bigoplus","bigotimes","coprod",
    # Limits
    "lim","limsup","liminf","inf","sup","max","min","argmax","argmin",
    # Trig / functions
    "log","ln","exp","sin","cos","tan","cot","sec","csc",
    "arcsin","arccos","arctan","sinh","cosh","tanh",
    "det","dim","ker","gcd","deg","mod","bmod","pmod",
    # Fractions / binomials
    "frac","dfrac","tfrac","cfrac","binom","dbinom","tbinom",
    # Roots
    "sqrt",
    # Relations
    "leq","geq","neq","approx","equiv","sim","simeq","propto","cong",
    "ll","gg","prec","succ","preceq","succeq",
    "subset","supset","subseteq","supseteq","in","notin","ni",
    "cup","cap","setminus","emptyset","varnothing",
    # Arrows
    "to","rightarrow","leftarrow","Rightarrow","Leftarrow",
    "leftrightarrow","Leftrightarrow","mapsto","implies","iff","xrightarrow",
    # Misc symbols
    "infty","partial","nabla","forall","exists","nexists","neg",
    "cdot","cdots","ldots","ddots","vdots","times","div",
    "pm","mp","circ","bullet","star","dagger","angle","perp","parallel",
    # Formatting
    "text","textbf","textit","textrm","texttt",
    "mathrm","mathbf","mathbb","mathcal","mathfrak","mathsf","mathtt",
    "operatorname",
    # Accents
    "overline","underline","hat","tilde","bar","vec",
    "dot","ddot","widehat","widetilde",
    # Stacking
    "overset","underset","stackrel","overbrace","underbrace",
    # Spacing
    "quad","qquad",
    # Delimiters
    "left","right","bigl","bigr","Bigl","Bigr",
], key=len, reverse=True)

# Build a set for O(1) lookup
_LATEX_CMD_SET = set(_LATEX_COMMANDS)

# Words that are math function names (allowed in "math-only" lines)
_MATH_FUNC_NAMES = {
    "var","cov","corr","log","sin","cos","tan","exp","det","dim",
    "ker","gcd","max","min","sup","inf","lim","mod","arg",
}

# Citation patterns like "(Rice, page 138)", "(Ross, page 403)"
_CITATION_PATTERN = re.compile(
    r'\(\s*[A-Z][a-z]+(?:\s+(?:and|y|&)\s+[A-Z][a-z]+)*'
    r'(?:\s*,\s*(?:page|pages|p\.|pp\.|cap|chapter)\s*\.?\s*\d+(?:\s*[-–]\s*\d+)?)*\s*\)'
)


def normalize_latex(text: str) -> str:
    """Normalize LaTeX in LLM output for Streamlit KaTeX."""
    if not text:
        return ""

    # 0. Fix star-bullets that interfere with KaTeX
    text = _fix_star_bullets(text)

    # 0.5 Replace Unicode math symbols
    text = _replace_unicode_math(text)

    # 0.1 Replace \xrightarrow{X} with \overset{X}{\to} for better KaTeX compatibility
    text = re.sub(r'\\xrightarrow\{([^}]+)\}', r'\\overset{\1}{\\to}', text)

    # 1. Unescape \$
    text = re.sub(r'\\+\$', '$', text)

    # 2. Convert \[...\] → $$...$$ and \(...\) → $...$
    text = re.sub(r'\\\[(.*?)\\\]', r'\n$$\n\1\n$$\n', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.*?)\\\)', r'$ \1 $', text, flags=re.DOTALL)

    # 3. Convert backtick-wrapped LaTeX to dollar-wrapped
    text = _convert_backtick_latex(text)

    # 3.5. Aggressively wrap common naked math symbols
    # This catches things like "\alpha" or "\approx" that appear in prose without $
    # valid_symbols = ["alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "mu", "pi", "sigma", "phi", "omega",
    #                  "approx", "le", "ge", "leq", "geq", "ne", "neq", "cdot", "times", "frac", "sum", "prod", "int"]
    # We use a broad regex for these common cases
    text = re.sub(
        r'(?<!\$)(?<!\\)(\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|pi|sigma|phi|omega|approx|le|ge|leq|geq|ne|neq|cdot|times|frac|sum|prod|int|hat|bar|tilde)\b(?:\{[^}]*\})?)', 
        r'$\1$', 
        text
    )

    # 4. Wrap standalone math lines as block math
    text = _wrap_math_lines(text)

    # 5. Wrap remaining naked LaTeX in prose (with boundary expansion)
    text = _wrap_inline_math(text)

    # 6. Cleanup
    text = re.sub(r'\${3,}', '$$', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# Unicode to LaTeX mapping
_GREEK_UNICODE = {
    'α': '\\alpha', 'β': '\\beta', 'γ': '\\gamma', 'δ': '\\delta', 'ε': '\\epsilon',
    'ζ': '\\zeta', 'η': '\\eta', 'θ': '\\theta', 'ι': '\\iota', 'κ': '\\kappa',
    'λ': '\\lambda', 'μ': '\\mu', 'ν': '\\nu', 'ξ': '\\xi', 'ο': 'o',
    'π': '\\pi', 'ρ': '\\rho', 'σ': '\\sigma', 'τ': '\\tau', 'υ': '\\upsilon',
    'φ': '\\phi', 'χ': '\\chi', 'ψ': '\\psi', 'ω': '\\omega',
    'Γ': '\\Gamma', 'Δ': '\\Delta', 'Θ': '\\Theta', 'Λ': '\\Lambda',
    'Ξ': '\\Xi', 'Π': '\\Pi', 'Σ': '\\Sigma', 'Υ': '\\Upsilon',
    'Φ': '\\Phi', 'Ψ': '\\Psi', 'Ω': '\\Omega',
    '∞': '\\infty', '≈': '\\approx', '≠': '\\neq', '≤': '\\leq', '≥': '\\geq',
    '×': '\\times', '·': '\\cdot', '±': '\\pm'
}

def _replace_unicode_math(text: str) -> str:
    """Replace Unicode math symbols with LaTeX equivalents."""
    # We only replace if they are likely used in a math context or single letters.
    # Since we can't easily distinguish Spanish text from math variables for some chars (like 'o'),
    # we'll focus on the unambiguous ones.
    for char, latex in _GREEK_UNICODE.items():
        if char in text:
            text = text.replace(char, latex + " ") # Add space after to prevent command merging
    return text

# ---------------------------------------------------------------------------
# Star-bullet fix (prevents Markdown italic/bold from breaking KaTeX)
# ---------------------------------------------------------------------------

def _fix_star_bullets(text: str) -> str:
    """Convert '* $...' and '* \\cmd' bullet lines to '- ...' to avoid
    Markdown italic parsing interfering with KaTeX."""
    # * $math or * \command at line start → replace * with -
    text = re.sub(
        r'^(\s*)\*\s+(\$|\\[a-zA-Z])',
        r'\1- \2',
        text,
        flags=re.MULTILINE
    )
    return text


# ---------------------------------------------------------------------------
# Backtick-wrapped LaTeX conversion
# ---------------------------------------------------------------------------

def _convert_backtick_latex(text: str) -> str:
    """Convert backtick-wrapped LaTeX to $...$.
    
    Handles two cases:
    1. `\\command...` patterns (e.g., `\\frac{a}{b}` → $\\frac{a}{b}$)
    2. Math variables with subscripts/superscripts (e.g., `u_n` → $u_n$, `X^2` → $X^2$)
    """
    def _replace_backtick(match):
        content = match.group(2)
        # Case 1: Has LaTeX commands → always convert
        if '\\' in content and _has_latex_command(content):
            return f'$ {content} $'
        # Case 2: Has subscript/superscript like u_n, X_{n+1}, x^2
        if re.search(r'[A-Za-z][_^]', content):
            return f'${content}$'
        return match.group(0)  # Keep original backticks
    
    # Match single backtick-wrapped text (not triple backticks for code blocks)
    text = re.sub(r'(?<!`)(`)((?:[^`])+?)(`)', _replace_backtick, text)
    return text


def _has_latex_command(text: str) -> bool:
    """Check if text contains a known LaTeX command."""
    i = 0
    while i < len(text):
        if text[i] == '\\':
            cmd = _match_command(text, i)
            if cmd:
                return True
        i += 1
    return False


# ---------------------------------------------------------------------------
# Line-level block math detection
# ---------------------------------------------------------------------------

def _is_math_line(line: str) -> bool:
    """A line is 'pure math' if it has a \\command and no prose words (≥3 chars)."""
    stripped = line.strip()
    # If it's already wrapped in $ or contains $ (mixed content), don't wrap in $$
    if not stripped or '\\' not in stripped or '$' in stripped:
        return False

    # Remove \command sequences (and their brace args)
    temp = re.sub(r'\\[a-zA-Z]+', '', stripped)
    temp = re.sub(r'\{[^{}]*\}', '', temp)   # inner braces
    temp = re.sub(r'\{[^{}]*\}', '', temp)   # second pass for nesting

    # Find remaining alphabetic words
    words = re.findall(r'[A-Za-z]{3,}', temp)
    # Allow known math function names
    prose_words = [w for w in words if w.lower() not in _MATH_FUNC_NAMES]
    return len(prose_words) == 0


def _wrap_math_lines(text: str) -> str:
    """Detect standalone math lines and wrap them in $$ ... $$."""
    lines = text.split('\n')
    result = []
    in_block = False

    for line in lines:
        stripped = line.strip()

        # Track existing $$ blocks
        if '$$' in stripped:
            count = stripped.count('$$')
            if count % 2 == 1:
                in_block = not in_block
            result.append(line)
            continue

        if in_block:
            result.append(line)
            continue

        # Check if line already has inline $ wrapping most of it
        if stripped.startswith('$') and stripped.endswith('$'):
            result.append(line)
            continue

        if _is_math_line(stripped):
            result.append(f'\n$$\n{stripped}\n$$\n')
        else:
            result.append(line)

    return '\n'.join(result)


# ---------------------------------------------------------------------------
# Inline math wrapping with boundary expansion
# ---------------------------------------------------------------------------

def _find_math_regions(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans for all existing $/$$ regions."""
    regions = []
    i, n = 0, len(text)

    while i < n:
        if text[i] == '$':
            start = i
            if i + 1 < n and text[i + 1] == '$':
                # Block $$...$$
                i += 2
                while i < n - 1:
                    if text[i] == '$' and text[i + 1] == '$':
                        i += 2
                        break
                    i += 1
                else:
                    i = n
            else:
                # Inline $...$
                i += 1
                while i < n:
                    if text[i] == '$' and (i + 1 >= n or text[i + 1] != '$'):
                        i += 1
                        break
                    i += 1
            regions.append((start, i))
        else:
            i += 1
    return regions


def _in_region(pos: int, regions: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in regions)


def _match_command(text: str, pos: int) -> str | None:
    """Try to match a known LaTeX command at text[pos] (the backslash)."""
    n = len(text)
    for cmd in _LATEX_COMMANDS:
        end = pos + 1 + len(cmd)
        if end <= n and text[pos + 1:end] == cmd:
            if end >= n or not text[end].isalpha():
                return cmd
    return None


def _is_citation_ahead(text: str, pos: int) -> bool:
    """Check if text at pos starts a citation like (Author, page N)."""
    if pos >= len(text) or text[pos] != '(':
        return False
    # Look ahead for citation pattern
    remaining = text[pos:pos + 80]  # check up to 80 chars ahead
    return bool(_CITATION_PATTERN.match(remaining))


def _expand_backward(text: str, pos: int, regions: list) -> int:
    """Expand backward from pos to find where the math expression starts."""
    best = pos
    i = pos - 1

    while i >= 0:
        if _in_region(i, regions):
            break
        c = text[i]

        if c.isdigit() or c in '[]{}=+-*/<>^_|':
            best = i
            i -= 1
        elif c == '(':
            # Only include if it's part of math, not a citation
            # Check if this parens starts something like "P(" or "E("
            if i > 0 and text[i-1].isalpha():
                best = i
                i -= 1
            else:
                break
        elif c == ')':
            # Check backward for matching open paren
            best = i
            i -= 1
        elif c == ' ':
            i -= 1          # skip space tentatively
        elif c == '\\':
            if _match_command(text, i):
                best = i
                i -= 1
            else:
                break
        elif c.isalpha():
            # Collect full word backwards
            j = i
            while j >= 0 and text[j].isalpha():
                j -= 1
            word = text[j + 1:i + 1]
            if len(word) >= 3 and word.lower() not in _MATH_FUNC_NAMES:
                break  # prose word → stop
            best = j + 1
            i = j
        else:
            break
    return best


def _expand_forward(text: str, pos: int, regions: list) -> int:
    """Expand forward from pos to find where the math expression ends."""
    n = len(text)
    best = pos
    i = pos

    while i < n:
        if _in_region(i, regions):
            break
        c = text[i]

        if c in '_^':
            # Subscript/superscript: always consume this AND the next token
            i += 1
            if i < n:
                if text[i] == '{':
                    _, i = _balanced(text, i, '{', '}')
                else:
                    # Capture next single char or command, avoiding space
                    while i < n and text[i] == ' ':
                         i += 1
                    if i < n:
                         if text[i] == '\\':
                             cmd = _match_command(text, i)
                             if cmd:
                                 _, i = _collect_expression(text, i, cmd)
                             else:
                                 i += 1
                         else:
                             i += 1
            best = i
        elif c.isdigit() or c in '[]{}=+-*/<>|':
            i += 1
            best = i
        elif c == '(':
            # Check if this starts a citation like (Author, page N) 
            if _is_citation_ahead(text, i):
                break  # stop before citation
            # Otherwise it's likely a math grouping like P(X)
            i += 1
            best = i
        elif c == ')':
            # Close paren is part of math expression
            i += 1
            best = i
        elif c == ',':
            # Comma: check context. Could be part of math (e.g., P(X,Y))
            # or a natural language separator
            # If next non-space char is a digit or \command, it's math
            j = i + 1
            while j < n and text[j] == ' ':
                j += 1
            if j < n and (text[j] == '\\' or text[j].isdigit() or text[j] in '({['):
                i += 1
                best = i
            else:
                break
        elif c == ' ':
            i += 1          # skip space tentatively
        elif c == '\\':
            cmd = _match_command(text, i)
            if cmd:
                _, end = _collect_expression(text, i, cmd)
                i = end
                best = end
            else:
                break
        elif c.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            word = text[i:j]
            if len(word) >= 3 and word.lower() not in _MATH_FUNC_NAMES:
                break  # prose word → stop
            # Single/double-letter token: check if it's a connector
            # (e.g., "y para", "o sea", "a la") rather than a math variable
            if len(word) <= 2:
                # Look ahead past spaces for the next word
                k = j
                while k < n and text[k] == ' ':
                    k += 1
                # If next word is prose (3+ chars), this letter is a connector → stop
                if k < n and text[k].isalpha():
                    word_end = k
                    while word_end < n and text[word_end].isalpha():
                        word_end += 1
                    next_word = text[k:word_end]
                    if len(next_word) >= 3 and next_word.lower() not in _MATH_FUNC_NAMES:
                        break  # "y para", "o sea" → stop before the connector
            i = j
            best = j
        elif c == '.' and i + 1 < n and text[i + 1].isdigit():
            i += 1          # decimal point
            best = i
        else:
            break
    return best


def _merge(regions: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not regions:
        return []
    regions.sort()
    merged = [regions[0]]
    for s, e in regions[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _wrap_inline_math(text: str) -> str:
    """Find naked \\commands in prose, expand boundaries, wrap in $ ... $."""
    existing = _find_math_regions(text)

    # Find all naked commands
    naked = []
    i = 0
    while i < len(text):
        if text[i] == '\\' and not _in_region(i, existing):
            cmd = _match_command(text, i)
            if cmd:
                naked.append((i, cmd))
                i += 1 + len(cmd)
                continue
        i += 1

    if not naked:
        return text

    # Compute expanded regions
    new_regions = []
    for cmd_pos, cmd_name in naked:
        _, cmd_end = _collect_expression(text, cmd_pos, cmd_name)
        start = _expand_backward(text, cmd_pos, existing)
        end = _expand_forward(text, cmd_end, existing)
        new_regions.append((start, end))

    new_regions = _merge(new_regions)

    # Filter overlaps with existing math
    filtered = [r for r in new_regions
                 if not any(r[0] < me and ms < r[1] for ms, me in existing)]

    # Build output
    result = []
    pos = 0
    for start, end in filtered:
        result.append(text[pos:start])
        expr = text[start:end].strip()
        # Space padding
        if result and result[-1] and result[-1][-1] not in (' ', '\n', '\t'):
            result.append(' ')
        result.append(f'$ {expr} $')
        if end < len(text) and text[end] not in (' ', '\n', '\t', '.', ',', ';', ':', '!', '?', ')', ']', '}'):
            result.append(' ')
        pos = end
    result.append(text[pos:])
    return ''.join(result)


# ---------------------------------------------------------------------------
# Expression collection helpers
# ---------------------------------------------------------------------------

def _collect_expression(text: str, start: int, cmd: str) -> tuple[str, int]:
    """Collect \\command with its brace/bracket args and sub/superscripts."""
    n = len(text)

    if cmd == 'left':
        return _collect_left_right(text, start)

    i = start + 1 + len(cmd)
    expr = text[start:i]

    two_arg = {'frac','dfrac','tfrac','cfrac','binom','dbinom','tbinom',
               'overset','underset','stackrel'}
    one_arg = {'sqrt','text','textbf','textit','textrm','texttt',
               'mathrm','mathbf','mathbb','mathcal','mathfrak','mathsf',
               'mathtt','operatorname','overline','underline','hat',
               'tilde','bar','vec','dot','ddot','widehat','widetilde',
               'overbrace','underbrace'}

    num_args = 2 if cmd in two_arg else (1 if cmd in one_arg else 0)

    # Optional [] arg  (e.g. \sqrt[3]{x})
    while i < n and text[i] == ' ':
        i += 1
    if i < n and text[i] == '[':
        _, i = _balanced(text, i, '[', ']')

    for _ in range(num_args):
        while i < n and text[i] == ' ':
            i += 1
        if i < n and text[i] == '{':
            _, i = _balanced(text, i, '{', '}')

    # Trailing sub/superscripts
    while i < n and text[i] in '_^':
        i += 1
        if i < n:
            if text[i] == '{':
                _, i = _balanced(text, i, '{', '}')
            elif text[i] != ' ':
                i += 1

    return text[start:i], i


def _collect_left_right(text: str, start: int) -> tuple[str, int]:
    """Collect \\left ... \\right pair."""
    n = len(text)
    depth = 1
    i = start + 5  # past \left
    if i < n:
        i += 2 if text[i] == '\\' else 1  # skip delimiter

    while i < n and depth > 0:
        if text[i:i+5] == '\\left' and (i+5 >= n or not text[i+5].isalpha()):
            depth += 1; i += 5; continue
        if text[i:i+6] == '\\right' and (i+6 >= n or not text[i+6].isalpha()):
            depth -= 1
            i += 6
            if depth == 0:
                if i < n:
                    i += 2 if text[i] == '\\' else 1
                return text[start:i], i
            continue
        i += 1
    return text[start:i], i


def _balanced(text: str, start: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Collect from open_ch to matching close_ch, handling nesting."""
    depth, i = 1, start + 1
    while i < len(text) and depth > 0:
        if text[i] == open_ch:
            depth += 1
        elif text[i] == close_ch:
            depth -= 1
        i += 1
    return text[start:i], i
