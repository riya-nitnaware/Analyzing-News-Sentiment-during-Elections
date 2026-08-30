"""
PDF Report Generator for Election NLP
Uses project-local TTF fonts (Arial) so it works on any computer.
All errors are caught internally – callers receive None on failure.
"""
import os
import logging

logger = logging.getLogger(__name__)

# Resolve the fonts directory relative to this file so it works on any machine
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_FONTS_DIR = os.path.join(_BASE_DIR, "assets", "fonts")
_FONT_REGULAR = os.path.join(_FONTS_DIR, "arial.ttf")
_FONT_BOLD    = os.path.join(_FONTS_DIR, "arialbd.ttf")
_FONT_ITALIC  = os.path.join(_FONTS_DIR, "ariali.ttf")


def _sanitize(text: str) -> str:
    """
    Aggressively sanitise text for PDF rendering with Latin-1 fonts.
    - Known Unicode punctuation is mapped to readable ASCII equivalents.
    - Invisible/zero-width characters are stripped entirely.
    - Anything still outside Latin-1 is replaced with '?' rather than crashing.
    """
    if not text:
        return ""

    # 1. Strip invisible/control characters (zero-width space, BOM, etc.)
    import unicodedata
    import re
    # Remove zero-width and control characters
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]", "", text)

    # 2. Map common Unicode punctuation to ASCII equivalents
    replacements = {
        "\u2018": "'", "\u2019": "'",   # curly single quotes
        "\u201c": '"', "\u201d": '"',   # curly double quotes
        "\u2013": "-",                   # en dash
        "\u2014": "--",                  # em dash
        "\u2015": "--",                  # horizontal bar
        "\u2026": "...",                 # ellipsis
        "\u00a0": " ",                   # non-breaking space
        "\u00e9": "e", "\u00e8": "e", "\u00ea": "e", "\u00eb": "e",
        "\u00e0": "a", "\u00e2": "a", "\u00e4": "a",
        "\u00f9": "u", "\u00fb": "u", "\u00fc": "u",
        "\u00ee": "i", "\u00ef": "i", "\u00ed": "i",
        "\u00f4": "o", "\u00f6": "o",
        "\u00e7": "c",
        "\u00f1": "n",
        "\u20b9": "Rs.", "\u0024": "$",
        "\u2022": "-", "\u2023": "-", "\u25cf": "-",
        "\u00ab": '"', "\u00bb": '"',   # guillemets
        "\u2039": "'", "\u203a": "'",
        "\u00b7": ".",                   # middle dot
        "\u2192": "->", "\u2190": "<-",
        "\u00d7": "x", "\u00f7": "/",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)

    # 3. Final encoding — replace anything still not encodable as Latin-1 with '?'
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(analysis_record: dict):
    """
    Generate a professional PDF report for a sentiment analysis.
    Returns raw bytes on success, or None on failure.
    Errors are logged internally and never propagated to the UI.
    """
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_margins(20, 20, 20)
        pdf.add_page()

        # Register Unicode-capable fonts from the project directory
        if os.path.exists(_FONT_REGULAR):
            pdf.add_font("Arial", "",  _FONT_REGULAR, uni=True)
            pdf.add_font("Arial", "B", _FONT_BOLD,    uni=True)
            pdf.add_font("Arial", "I", _FONT_ITALIC,  uni=True)
            use_arial = True
        else:
            # Fallback – built-in Helvetica (Latin-1 only, but still functional)
            logger.warning("Arial TTF not found in assets/fonts; falling back to Helvetica.")
            use_arial = False

        def set_font(style="", size=11):
            if use_arial:
                pdf.set_font("Arial", style, size)
            else:
                pdf.set_font("Helvetica", style, size)

        # ── Header ──────────────────────────────────────────────
        set_font("B", 18)
        pdf.cell(0, 12, "ELECTION NLP", new_x="LMARGIN", new_y="NEXT", align="C")

        set_font("", 11)
        pdf.cell(0, 7, "Analyzing News Sentiment During Elections", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

        # Thin horizontal rule
        pdf.set_draw_color(180, 180, 180)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        set_font("B", 14)
        pdf.cell(0, 10, "Sentiment Analysis Report", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(3)

        # ── Metadata ─────────────────────────────────────────────
        set_font("", 10)
        pdf.cell(0, 6, f"Analysis Date: {_sanitize(analysis_record.get('date', 'Unknown'))}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        # ── Input: Headline ──────────────────────────────────────
        headline = _sanitize(analysis_record.get('headline', '') or '')
        if headline:
            set_font("B", 11)
            pdf.cell(0, 7, "Headline:", new_x="LMARGIN", new_y="NEXT")
            set_font("", 10)
            pdf.multi_cell(0, 6, headline)
            pdf.ln(2)

        # ── Input: Article ───────────────────────────────────────
        article = _sanitize(analysis_record.get('article', '') or '')
        if article:
            set_font("B", 11)
            pdf.cell(0, 7, "News Article:", new_x="LMARGIN", new_y="NEXT")
            set_font("", 10)
            # Limit very long articles to keep the PDF manageable
            if len(article) > 1500:
                article = article[:1500] + "... [truncated for report]"
            pdf.multi_cell(0, 6, article)
            pdf.ln(3)

        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        # ── Result ───────────────────────────────────────────────
        sentiment  = analysis_record.get('sentiment', 'Unknown')
        confidence = analysis_record.get('confidence', 0.0)

        set_font("B", 13)
        pdf.cell(0, 8, "Predicted Sentiment", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

        # Colour-code the sentiment label
        if sentiment == "Positive":
            pdf.set_text_color(34, 139, 34)
        elif sentiment == "Negative":
            pdf.set_text_color(200, 30, 30)
        else:
            pdf.set_text_color(90, 90, 90)

        set_font("B", 18)
        pdf.cell(0, 12, sentiment.upper(), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_text_color(0, 0, 0)

        set_font("", 11)
        pdf.cell(0, 7, f"Confidence Score: {confidence:.1%}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)

        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        # ── Probabilities ─────────────────────────────────────────
        probs = analysis_record.get('probabilities', {})
        if probs:
            set_font("B", 11)
            pdf.cell(0, 7, "Sentiment Probabilities:", new_x="LMARGIN", new_y="NEXT")
            set_font("", 10)
            for cls in ["Positive", "Neutral", "Negative"]:
                p = probs.get(cls, 0.0)
                pdf.cell(0, 6, f"  {cls}: {p:.1%}", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(5)

        # ── Text Statistics ──────────────────────────────────────
        word_count = analysis_record.get('word_count', 0)
        full_text  = f"{headline} {article}".strip()
        char_count = len(full_text)

        set_font("B", 11)
        pdf.cell(0, 7, "Text Statistics:", new_x="LMARGIN", new_y="NEXT")
        set_font("", 10)
        pdf.cell(0, 6, f"  Word Count: {word_count}", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(0, 6, f"  Character Count: {char_count}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        # ── Interpretation ────────────────────────────────────────
        set_font("B", 11)
        pdf.cell(0, 7, "Analysis Summary:", new_x="LMARGIN", new_y="NEXT")
        set_font("I", 10)
        summary = (
            f"The analyzed news text was classified as {sentiment} by the sentiment "
            f"analysis model with a confidence score of {confidence:.1%}."
        )
        pdf.multi_cell(0, 6, _sanitize(summary))
        pdf.ln(5)

        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(5)

        # ── Disclaimer ────────────────────────────────────────────
        set_font("B", 10)
        pdf.cell(0, 6, "Disclaimer:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(100, 100, 100)
        set_font("", 9)
        disclaimer = (
            "This result is generated by an NLP-based machine learning model. "
            "It represents the predicted sentiment of the provided text and does not "
            "determine the factual accuracy, political fairness, bias, or truthfulness "
            "of the news article."
        )
        pdf.multi_cell(0, 5, _sanitize(disclaimer))
        pdf.set_text_color(0, 0, 0)

        return bytes(pdf.output())

    except Exception:
        logger.exception("PDF generation failed internally")
        return None
