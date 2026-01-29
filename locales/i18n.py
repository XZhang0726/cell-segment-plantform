"""
Internationalization (i18n) module for the Cell Segmentation Platform.
Provides translation functions and language management.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st


class I18n:
    """Translation engine with caching and dynamic string formatting."""

    def __init__(self):
        self.locales_dir = Path(__file__).parent
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.supported_languages = ['en_US', 'zh_CN']
        self.default_language = 'en_US'  # English as default

        # Load all translations at initialization
        self._load_translations()

    def _load_translations(self):
        """Load all translation files into memory (cached)."""
        for lang in self.supported_languages:
            json_file = self.locales_dir / f"{lang}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.translations[lang] = json.load(f)
            else:
                print(f"Warning: Translation file {json_file} not found")
                self.translations[lang] = {}

    def get_current_language(self) -> str:
        """Get current language from session state."""
        if 'language' not in st.session_state:
            st.session_state['language'] = self.default_language
        return st.session_state['language']

    def set_language(self, language: str):
        """Set current language in session state."""
        if language in self.supported_languages:
            st.session_state['language'] = language
        else:
            raise ValueError(f"Unsupported language: {language}")

    def t(self, key: str, **kwargs) -> str:
        """
        Translate a key to the current language.

        Args:
            key: Dot-notation key (e.g., 'tabs.image_segmentation')
            **kwargs: Dynamic values for string formatting

        Returns:
            Translated string with dynamic values formatted

        Example:
            t('messages.cells_detected', count=42)
            # Returns: "🔍 Detected 42 cell regions" (English)
            # Returns: "🔍 检测到 42 个细胞区域" (Chinese)
        """
        lang = self.get_current_language()
        translation_dict = self.translations.get(lang, {})

        # Navigate nested dictionary using dot notation
        keys = key.split('.')
        value = translation_dict

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                value = None
                break

        # Fallback to key if translation not found
        if value is None:
            print(f"Warning: Translation key '{key}' not found for language '{lang}'")
            return key

        # Format dynamic strings
        if kwargs:
            try:
                return value.format(**kwargs)
            except KeyError as e:
                print(f"Warning: Missing format key {e} in translation '{key}'")
                return value
        return value

    def get_language_name(self, lang_code: str) -> str:
        """Get display name for language code."""
        names = {
            'en_US': 'English',
            'zh_CN': '中文'
        }
        return names.get(lang_code, lang_code)


# Global instance
_i18n = I18n()


# Convenience function for use in app
def t(key: str, **kwargs) -> str:
    """Shorthand translation function."""
    return _i18n.t(key, **kwargs)


def get_i18n() -> I18n:
    """Get the global i18n instance."""
    return _i18n
