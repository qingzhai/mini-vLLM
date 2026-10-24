#!/usr/bin/env python3
"""Export the canonical Markdown resume while preserving the Word page style."""
import argparse
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read_source(path):
    parts = [part.strip() for part in path.read_text().strip().split('\n\n')]
    if len(parts) != 7 or not parts[0].startswith('# '):
        raise ValueError('Expected one title, one overview and five technical points')
    title, date = parts[0][2:].rsplit(' | ', 1)
    if not all(re.match(r'^\*\*[^*]+：\*\* ', part) for part in parts[1:]):
        raise ValueError('Every project point must start with its bold label')
    return title, date, parts[1:]


def plain(text):
    return text.replace('**', '')


def check(source, destination):
    from docx import Document
    title, date, body = read_source(source)
    paragraphs = [p.text for p in Document(destination).paragraphs if p.text]
    expected = [title, date+'  ·  项目经历     github.com/qingzhai/mini-vLLM', *map(plain, body)]
    if paragraphs != expected:
        raise ValueError('Word text differs from the canonical Markdown resume')
    return expected


def export(source, destination):
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor
    title, date, body = read_source(source)
    if not destination.is_file():
        raise FileNotFoundError('The existing Word document supplies the page/style template: '+str(destination))
    doc = Document(destination)
    # Keep section layout and styles, replace only document text.
    for child in list(doc._element.body):
        if child.tag != qn('w:sectPr'):
            doc._element.body.remove(child)
    normal = doc.styles['Normal']
    normal.font.name = 'Arial Unicode MS'
    normal.font.size = Pt(11)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), 'Arial Unicode MS')

    def font(run, size, bold=False, color='000000'):
        run.font.name = 'Arial Unicode MS'
        run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'), 'Arial Unicode MS')
        run.font.size, run.bold = Pt(size), bold
        run.font.color.rgb = RGBColor.from_string(color)

    p = doc.add_paragraph(style='Title')
    font(p.add_run(title), 16, True)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    p = doc.add_paragraph()
    font(p.add_run(date+'  ·  项目经历     github.com/qingzhai/mini-vLLM'), 10, color='444444')
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    for text in body:
        p = doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.08
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_together = True
        for index, piece in enumerate(re.split(r'\*\*(.*?)\*\*', text)):
            if piece:
                font(p.add_run(piece), 11, index % 2 == 1)
    doc.save(destination)
    check(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'docs/RESUME.zh.md')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/RESUME.zh.docx')
    parser.add_argument('--check', action='store_true', help='Check text equality without modifying files')
    args = parser.parse_args()
    if args.check:
        check(args.source, args.output)
    else:
        export(args.source, args.output)
    print('RESUME TEXT MATCH:', args.output)
    print('Visual page verification is a separate step after export.')


if __name__ == '__main__':
    main()
