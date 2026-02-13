"""
Query expansion for mathematical and statistical terms.

Expands queries with synonyms, abbreviations, and translations
to improve retrieval of mathematical content.
"""

# Spanish -> English + Abbreviations
MATH_TERM_EXPANSIONS = {
    # Laws of Large Numbers
    "ley débil grandes números": [
        "LDGN", "weak law large numbers", "WLLN",
        "convergencia en probabilidad", "convergence in probability"
    ],
    "ley débil": [
        "LDGN", "weak law", "WLLN"
    ],
    "ley fuerte grandes números": [
        "LFGN", "strong law large numbers", "SLLN",
        "convergencia casi segura", "almost sure convergence"
    ],
    "ley fuerte": [
        "LFGN", "strong law", "SLLN"
    ],
    
    # Central Limit Theorem
    "teorema central límite": [
        "TCL", "central limit theorem", "CLT",
        "convergencia en distribución", "convergence in distribution",
        "distribución normal", "normal distribution"
    ],
    "teorema central": [
        "TCL", "CLT", "central limit"
    ],
    
    # Probability Concepts
    "variable aleatoria": [
        "random variable", "RV", "v.a.", "r.v."
    ],
    "función de distribución": [
        "distribution function", "CDF", "FDA", "cumulative distribution"
    ],
    "función de densidad": [
        "density function", "PDF", "fdp", "probability density"
    ],
    "esperanza matemática": [
        "expected value", "expectation", "mean", "media", "E[X]"
    ],
    "esperanza": [
        "expected value", "expectation", "mean"
    ],
    "varianza": [
        "variance", "var", "Var", "σ²"
    ],
    "desviación estándar": [
        "standard deviation", "std", "σ"
    ],
    
    # Convergence Types
    "convergencia en probabilidad": [
        "convergence in probability", "converge in probability", "probability convergence"
    ],
    "convergencia casi segura": [
        "almost sure convergence", "a.s. convergence", "convergence almost surely"
    ],
    "convergencia en distribución": [
        "convergence in distribution", "weak convergence", "distributional convergence"
    ],
    "convergencia en media cuadrática": [
        "mean square convergence", "L2 convergence", "quadratic mean convergence"
    ],
    
    # Estimators
    "estimador insesgado": [
        "unbiased estimator", "estimador no sesgado", "unbiased estimate"
    ],
    "estimador consistente": [
        "consistent estimator", "consistent estimate"
    ],
    "estimador eficiente": [
        "efficient estimator", "UMVUE", "minimum variance"
    ],
    "estimador máxima verosimilitud": [
        "maximum likelihood estimator", "MLE", "EMV"
    ],
    
    # Distributions
    "distribución normal": [
        "normal distribution", "Gaussian", "gaussiana", "N(μ,σ²)"
    ],
    "distribución binomial": [
        "binomial distribution", "Bin(n,p)"
    ],
    "distribución poisson": [
        "Poisson distribution", "Pois(λ)"
    ],
    "distribución exponencial": [
        "exponential distribution", "Exp(λ)"
    ],
    "distribución uniforme": [
        "uniform distribution", "U(a,b)"
    ],
    
    # Hypothesis Testing
    "prueba de hipótesis": [
        "hypothesis test", "hypothesis testing", "statistical test"
    ],
    "nivel de significancia": [
        "significance level", "alpha", "α"
    ],
    "p-valor": [
        "p-value", "p value", "significance probability"
    ],
    
    # Regression
    "regresión lineal": [
        "linear regression", "least squares", "mínimos cuadrados"
    ],
    "coeficiente de correlación": [
        "correlation coefficient", "Pearson correlation", "r"
    ],
    
    # Probability Theory
    "espacio de probabilidad": [
        "probability space", "(Ω,F,P)"
    ],
    "sigma álgebra": [
        "sigma algebra", "σ-algebra", "σ-field"
    ],
    "medida de probabilidad": [
        "probability measure", "measure"
    ],
    "independencia": [
        "independence", "independent", "independiente"
    ],
    
    # Common Terms
    "demostración": [
        "proof", "prueba", "dem."
    ],
    "teorema": [
        "theorem", "thm"
    ],
    "lema": [
        "lemma"
    ],
    "proposición": [
        "proposition", "prop"
    ],
    "corolario": [
        "corollary", "cor"
    ],
    "definición": [
        "definition", "def"
    ],
}


def expand_query(query: str, max_expansions: int = 3) -> list[str]:
    """
    Expand a query with mathematical term synonyms and abbreviations.
    
    Args:
        query: Original query string.
        max_expansions: Maximum number of expansion terms to add.
        
    Returns:
        List of query variations (original + expansions).
    """
    query_lower = query.lower()
    expansions = [query]  # Always include original
    
    # Find matching terms (longest match first)
    matched_terms = []
    for term in sorted(MATH_TERM_EXPANSIONS.keys(), key=len, reverse=True):
        if term in query_lower:
            matched_terms.append(term)
            # Only match once to avoid explosion
            break
    
    # Add expansions from matched terms
    for term in matched_terms:
        term_expansions = MATH_TERM_EXPANSIONS[term][:max_expansions]
        expansions.extend(term_expansions)
    
    return expansions


def expand_query_for_hybrid(query: str, max_expansions: int = 2) -> str:
    """
    Create a single expanded query string for hybrid search.
    
    Combines original query with top expansions, optimized for BM25.
    
    Args:
        query: Original query string.
        max_expansions: Maximum number of expansion terms to add.
        
    Returns:
        Expanded query string with additional terms.
    """
    expansions = expand_query(query, max_expansions=max_expansions)
    
    # Join with spaces (BM25 will treat as OR)
    return " ".join(expansions)


def get_expansion_info(query: str) -> dict:
    """
    Get detailed information about query expansion.
    
    Useful for debugging and understanding what terms were expanded.
    
    Args:
        query: Query string to analyze.
        
    Returns:
        Dictionary with expansion details.
    """
    query_lower = query.lower()
    
    # Find all matching terms
    matches = []
    for term, expansions in MATH_TERM_EXPANSIONS.items():
        if term in query_lower:
            matches.append({
                "matched_term": term,
                "expansions": expansions,
                "position": query_lower.find(term)
            })
    
    # Sort by position in query
    matches.sort(key=lambda x: x["position"])
    
    return {
        "original_query": query,
        "matched_terms": [m["matched_term"] for m in matches],
        "all_expansions": [exp for m in matches for exp in m["expansions"]],
        "expanded_query": expand_query_for_hybrid(query),
        "match_details": matches
    }
