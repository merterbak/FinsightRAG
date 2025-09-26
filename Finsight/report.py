from __future__ import annotations
import io
from typing import Dict, List, Optional
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


def markdown_to_pdf_bytes(markdown_text: str, title: Optional[str] = None):
    settings: Dict[str, object] = {
        "page_size": LETTER,
        "margin_left": 0.8 * inch,
        "margin_right": 0.8 * inch,
        "margin_top": 0.9 * inch,
        "margin_bottom": 0.9 * inch,
        "body_font": "Helvetica",
        "body_size": 10,
        "heading_font": "Helvetica-Bold",
        "heading_sizes": (16, 14, 12),
        "line_spacing": 1.35,
    }

    buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=settings["page_size"])
    page_width, page_height = settings["page_size"]
    x_start = settings["margin_left"]
    y_position = page_height - settings["margin_top"]
    text_width = page_width - settings["margin_left"] - settings["margin_right"]

    if title:
        heading_font = settings["heading_font"]
        heading_size = settings["heading_sizes"][0]
        pdf_canvas.setFont(heading_font, heading_size)
        for line in wrap_text(pdf_canvas, title, text_width, heading_font, heading_size):
            pdf_canvas.drawString(x_start, y_position, line)
            y_position -= heading_size * settings["line_spacing"]
        y_position -= heading_size * 0.5

    for line in markdown_text.splitlines():
        if line == "":
            y_position -= settings["body_size"] * settings["line_spacing"]
        else:
            y_position = draw_paragraph(pdf_canvas, x_start, y_position, line, text_width, settings)

        if y_position < settings["margin_bottom"] + settings["body_size"] * 2:
            pdf_canvas.showPage()
            y_position = page_height - settings["margin_top"]

    pdf_canvas.showPage()
    if title:
        pdf_canvas.setTitle(title)
    pdf_canvas.save()
    return buffer.getvalue()


def wrap_text(pdf_canvas: canvas.Canvas, text: str, width: float, font: str, size: int):
    pdf_canvas.setFont(font, size)
    words = (text or "").split()
    lines: List[str] = []
    current = ""

    for word in words:
        trial = f"{current} {word}".strip() if current else word
        if pdf_canvas.stringWidth(trial, font, size) <= width or not current:
            current = trial
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)
    return lines


def draw_paragraph(pdf_canvas: canvas.Canvas, x: float, y: float, text: str, width: float, settings: Dict[str, object],):
    headings = [("# ", 0, 0.5), ("## ", 1, 0.4), ("### ", 2, 0.3)]
    for prefix, size_index, gap in headings:
        if text.startswith(prefix):
            content = text[len(prefix):]
            font = settings["heading_font"]
            size = settings["heading_sizes"][size_index]
            pdf_canvas.setFont(font, size)
            for line in wrap_text(pdf_canvas, content, width, font, size):
                pdf_canvas.drawString(x, y, line)
                y -= size * settings["line_spacing"]
            y -= size * gap
            return y

    bullet_prefixes = ("- ", "* ")
    for prefix in bullet_prefixes:
        if text.startswith(prefix):
            bullet = "• "
            content = text[len(prefix):]
            font = settings["body_font"]
            size = settings["body_size"]
            pdf_canvas.setFont(font, size)
            bullet_width = pdf_canvas.stringWidth(bullet, font, size)
            lines = wrap_text(pdf_canvas, content, width - bullet_width, font, size)
            first_line = True
            for line in lines:
                if first_line:
                    pdf_canvas.drawString(x, y, bullet)
                    pdf_canvas.drawString(x + bullet_width, y, line)
                    first_line = False
                else:
                    pdf_canvas.drawString(x + bullet_width, y, line)
                y -= size * settings["line_spacing"]
            return y

    font = settings["body_font"]
    size = settings["body_size"]
    pdf_canvas.setFont(font, size)
    for line in wrap_text(pdf_canvas, text, width, font, size):
        pdf_canvas.drawString(x, y, line)
        y -= size * settings["line_spacing"]
    return y
