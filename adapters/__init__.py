"""
Universal Feature Adapters for Cross-Format Malicious Document Detection.

Provides hand-designed semantic adapters mapping format-specific static features
(Word, Excel, PDF, HTML) into unified, format-agnostic feature slots (1 to 10).
"""

from .universal_adapter import UniversalFeatureAdapter, sanitize_columns, safe_col

__all__ = ["UniversalFeatureAdapter", "sanitize_columns", "safe_col"]
