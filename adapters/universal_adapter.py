"""
Universal Feature Adapter Implementation.

Maps heterogeneous static features from Word, Excel, PDF, and HTML documents into
a 10-slot format-agnostic abstraction:
  Slot 1:  Content Entropy
  Slot 2:  Active Content Flag (macro, JS, embedded scripts)
  Slot 3:  External Reference Count (URIs, hyperlinks, external scripts)
  Slot 4:  Log File Size (log1p of total byte length)
  Slot 5:  Structural Complexity Ratio (density of elements, cells, objects)
  Slot 6:  Obfuscation Indicator (hex encoding, filters, obfuscated names)
  Slot 7:  Script / Text Volume (script length, text volume)
  Slot 8:  Encoded Content Counts (base64, escaped characters, OLE types)
  Slot 9:  Embedded Object Count (sheets, font/image objects, iframes)
  Slot 10: Stream / Script Size (stream counts, script entropy product)

Also supports a shortcut-free adapter variant for Word ("Word (No Shortcuts)")
which excludes the 41 shortcut/missing columns flagged during the L2 audit.
"""

import re
import numpy as np
import pandas as pd


def sanitize_columns(columns):
    """Sanitize column names by replacing non-alphanumeric characters with underscores."""
    return [re.sub(r'[^\w]', '_', c) for c in columns]


def safe_col(df, col, default=0.0):
    """Safely extract a column as float series, returning default series if absent."""
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


class UniversalFeatureAdapter:
    """
    Extracts format-agnostic feature representations across document formats.
    """

    FEATURE_SLOTS = [
        "1_content_entropy",
        "2_active_content_flag",
        "3_external_reference_count",
        "4_log_file_size",
        "5_structural_complexity_ratio",
        "6_obfuscation_indicator",
        "7_script_text_volume",
        "8_encoded_content_counts",
        "9_embedded_object_count",
        "10_stream_script_size",
    ]

    def __init__(self):
        self.mapping_log = {}

    def extract_features(self, fmt, X_in, k=6, include_shortcuts=True):
        """
        Extract the first k universal features for a given format dataframe.

        Args:
            fmt (str): Format name ('Word', 'Excel', 'PDF', 'HTML', or 'Word (No Shortcuts)')
            X_in (pd.DataFrame): Raw feature matrix without label column
            k (int): Feature budget in {4, 6, 8, 10}
            include_shortcuts (bool): Whether to include shortcut-bearing features for Word

        Returns:
            pd.DataFrame: Sliced universal feature dataframe with k columns
        """
        X = X_in.copy()
        uni = pd.DataFrame(index=X.index)

        is_no_sc = (fmt == 'Word (No Shortcuts)') or (not include_shortcuts)

        if fmt == 'Word' or fmt == 'Word (No Shortcuts)':
            if is_no_sc:
                # Shortcut-free adapter: only uses clean surviving features struct_script & struct_typeface
                uni['1_content_entropy'] = 0.0
                uni['2_active_content_flag'] = (safe_col(X, 'struct_script') > 0).astype(float)
                uni['3_external_reference_count'] = 0.0
                uni['4_log_file_size'] = 0.0
                uni['5_structural_complexity_ratio'] = safe_col(X, 'struct_typeface') / 10.0
                uni['6_obfuscation_indicator'] = 0.0
                uni['7_script_text_volume'] = safe_col(X, 'struct_typeface')
                uni['8_encoded_content_counts'] = 0.0
                uni['9_embedded_object_count'] = 0.0
                uni['10_stream_script_size'] = safe_col(X, 'struct_script')
            else:
                # Shortcut-bearing adapter (original benchmark extraction)
                uni['1_content_entropy'] = safe_col(X, 'entropy')
                uni['2_active_content_flag'] = (safe_col(X, 'macro_present') > 0).astype(float)
                uni['3_external_reference_count'] = 0.0
                uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
                uni['5_structural_complexity_ratio'] = safe_col(X, 'ole_object_count') / (safe_col(X, 'file_size') + 1.0)
                uni['6_obfuscation_indicator'] = safe_col(X, 'vba_keywords_count')
                uni['7_script_text_volume'] = safe_col(X, 'struct_typeface')
                uni['8_encoded_content_counts'] = safe_col(X, 'ole_object_type_count')
                uni['9_embedded_object_count'] = safe_col(X, 'ole_object_count')
                uni['10_stream_script_size'] = safe_col(X, 'struct_script')

            self.mapping_log['Word_NoShortcuts' if is_no_sc else 'Word'] = {
                '1_content_entropy': '0.0 (excluded shortcut)' if is_no_sc else 'entropy',
                '2_active_content_flag': 'struct_script > 0' if is_no_sc else 'macro_present > 0',
                '3_external_reference_count': 'EXCLUDED (Word dataset lacks link count column)',
                '4_log_file_size': '0.0 (excluded file_size shortcut)' if is_no_sc else 'log1p(file_size)',
                '5_structural_complexity_ratio': 'struct_typeface / 10' if is_no_sc else 'ole_object_count / file_size',
                '6_obfuscation_indicator': '0.0 (excluded vba shortcut)' if is_no_sc else 'vba_keywords_count',
                '7_script_text_volume': 'struct_typeface',
                '8_encoded_content_counts': '0.0 (excluded ole shortcut)' if is_no_sc else 'ole_object_type_count',
                '9_embedded_object_count': '0.0 (excluded ole shortcut)' if is_no_sc else 'ole_object_count',
                '10_stream_script_size': 'struct_script'
            }

        elif fmt == 'Excel':
            uni['1_content_entropy'] = safe_col(X, 'entropy_of_text')
            uni['2_active_content_flag'] = (safe_col(X, 'macro_token_count') > 0).astype(float)
            uni['3_external_reference_count'] = safe_col(X, 'remote_template_present') + safe_col(X, 'hyperlink_count')
            uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
            uni['5_structural_complexity_ratio'] = safe_col(X, 'total_cells') / (safe_col(X, 'file_size') + 1.0)
            uni['6_obfuscation_indicator'] = safe_col(X, 'macro_vocab_size') + safe_col(X, 'hex_pattern_count')
            uni['7_script_text_volume'] = safe_col(X, 'macro_chr_count')
            uni['8_encoded_content_counts'] = safe_col(X, 'base64_pattern_count')
            uni['9_embedded_object_count'] = safe_col(X, 'sheet_count')
            uni['10_stream_script_size'] = safe_col(X, 'macro_count')

            self.mapping_log['Excel'] = {
                '1_content_entropy': 'entropy_of_text',
                '2_active_content_flag': 'macro_token_count > 0',
                '3_external_reference_count': 'remote_template_present + hyperlink_count',
                '4_log_file_size': 'log1p(file_size)',
                '5_structural_complexity_ratio': 'total_cells / file_size',
                '6_obfuscation_indicator': 'macro_vocab_size + hex_pattern_count',
                '7_script_text_volume': 'macro_chr_count',
                '8_encoded_content_counts': 'base64_pattern_count',
                '9_embedded_object_count': 'sheet_count',
                '10_stream_script_size': 'macro_count'
            }

        elif fmt == 'PDF':
            uni['1_content_entropy'] = safe_col(X, 'entropy_of_streams')
            uni['2_active_content_flag'] = ((safe_col(X, 'js_count') + safe_col(X, 'launch_count') + safe_col(X, 'openaction_count')) > 0).astype(float)
            uni['3_external_reference_count'] = safe_col(X, 'uri_count') + safe_col(X, 'action_count')
            uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
            uni['5_structural_complexity_ratio'] = safe_col(X, 'object_count') / (safe_col(X, 'page_count') + 1.0)
            uni['6_obfuscation_indicator'] = safe_col(X, 'name_obfuscations') + safe_col(X, 'total_filters')
            uni['7_script_text_volume'] = safe_col(X, 'text_length')
            uni['8_encoded_content_counts'] = safe_col(X, 'total_filters')
            uni['9_embedded_object_count'] = safe_col(X, 'font_object_count') + safe_col(X, 'image_count')
            uni['10_stream_script_size'] = safe_col(X, 'stream_count')

            self.mapping_log['PDF'] = {
                '1_content_entropy': 'entropy_of_streams',
                '2_active_content_flag': '(js_count + launch_count + openaction_count) > 0',
                '3_external_reference_count': 'uri_count + action_count',
                '4_log_file_size': 'log1p(file_size)',
                '5_structural_complexity_ratio': 'object_count / page_count',
                '6_obfuscation_indicator': 'name_obfuscations + total_filters',
                '7_script_text_volume': 'text_length',
                '8_encoded_content_counts': 'total_filters',
                '9_embedded_object_count': 'font_object_count + image_count',
                '10_stream_script_size': 'stream_count'
            }

        elif fmt == 'HTML':
            uni['1_content_entropy'] = safe_col(X, 'entropy')
            uni['2_active_content_flag'] = ((safe_col(X, 'embedded_js_count') + safe_col(X, 'script_count')) > 0).astype(float)
            uni['3_external_reference_count'] = safe_col(X, 'external_links_count') + safe_col(X, 'external_link_count')
            uni['4_log_file_size'] = np.log1p(np.maximum(0, safe_col(X, 'file_size', 1.0)))
            uni['5_structural_complexity_ratio'] = safe_col(X, 'tag_count') / (safe_col(X, 'file_size') + 1.0)
            uni['6_obfuscation_indicator'] = safe_col(X, 'base64_string_count') + safe_col(X, 'hidden_iframe_count')
            uni['7_script_text_volume'] = safe_col(X, 'total_script_characters')
            uni['8_encoded_content_counts'] = safe_col(X, 'escaped_char_count')
            uni['9_embedded_object_count'] = safe_col(X, 'iframe_count') + safe_col(X, 'object_tag_count')
            uni['10_stream_script_size'] = safe_col(X, 'script_entropy') * safe_col(X, 'script_count') + 1e-4

            self.mapping_log['HTML'] = {
                '1_content_entropy': 'entropy',
                '2_active_content_flag': '(embedded_js_count + script_count) > 0',
                '3_external_reference_count': 'external_links_count + external_link_count',
                '4_log_file_size': 'log1p(file_size)',
                '5_structural_complexity_ratio': 'tag_count / file_size',
                '6_obfuscation_indicator': 'base64_string_count + hidden_iframe_count',
                '7_script_text_volume': 'total_script_characters',
                '8_encoded_content_counts': 'escaped_char_count',
                '9_embedded_object_count': 'iframe_count + object_tag_count',
                '10_stream_script_size': 'script_entropy * script_count'
            }

        uni = uni.dropna(axis=1, how='all')
        cols = list(uni.columns)[:k]
        return uni[cols]
