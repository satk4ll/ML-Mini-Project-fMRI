
from pathlib import Path
import re

import markdown
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
)

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report" / "final_report.md"
OUTPUT = ROOT / "report" / "final_report.pdf"
RESULTS = ROOT / "results"

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle",
    parent=styles["Title"],
    alignment=TA_CENTER,
    fontSize=20,
    leading=25,
    spaceAfter=16,
    textColor=colors.HexColor("#17365D"),
))

styles.add(ParagraphStyle(
    name="H1Custom",
    parent=styles["Heading1"],
    fontSize=15,
    leading=19,
    spaceBefore=14,
    spaceAfter=8,
))

styles.add(ParagraphStyle(
    name="H2Custom",
    parent=styles["Heading2"],
    fontSize=12,
    leading=15,
    spaceBefore=10,
    spaceAfter=5,
))

styles.add(ParagraphStyle(
    name="BodyCustom",
    parent=styles["BodyText"],
    fontSize=9.5,
    leading=13,
    spaceAfter=7,
))

styles.add(ParagraphStyle(
    name="SmallCustom",
    parent=styles["BodyText"],
    fontSize=8,
    leading=10,
    spaceAfter=4,
))

styles.add(ParagraphStyle(
    name="CodeBlockCustom",
    fontName="Courier",
    fontSize=7.5,
    leading=9,
    leftIndent=8,
    rightIndent=8,
    backColor=colors.HexColor("#F2F4F7"),
    borderPadding=6,
    spaceAfter=8,
))


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(
        2 * cm,
        1.1 * cm,
        "Machine Learning Mini-Project",
    )
    canvas.drawRightString(
        A4[0] - 2 * cm,
        1.1 * cm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def clean_inline(text):
    """Convert basic Markdown inline formatting to ReportLab markup."""
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)
    text = re.sub(
        r"`(.*?)`",
        r"<font name='Courier'>\1</font>",
        text,
    )
    return text


def add_paragraph(story, text, style="BodyCustom"):
    text = clean_inline(text.strip())
    if text:
        story.append(Paragraph(text, styles[style]))


def parse_table(lines, story):
    rows = []

    for line in lines:
        if not line.strip().startswith("|"):
            continue

        cells = [
            cell.strip()
            for cell in line.strip().strip("|").split("|")
        ]

        # Skip Markdown table separator rows.
        if all(
            re.fullmatch(r":?-{3,}:?", cell)
            for cell in cells
        ):
            continue

        rows.append([
            Paragraph(clean_inline(cell), styles["SmallCustom"])
            for cell in cells
        ])

    if not rows:
        return

    column_count = len(rows[0])
    available_width = A4[0] - 4 * cm

    table = Table(
        rows,
        colWidths=[available_width / column_count] * column_count,
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#D9EAF7"),
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.HexColor("#17365D"),
        ),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))

    story.append(table)
    story.append(Spacer(1, 8))


def add_figures(story):
    """Append figures while keeping each caption with its image."""
    figures = [
        (
            "class_distribution.png",
            "Figure 1. Target class distribution.",
        ),
        (
            "model_comparison.png",
            "Figure 2. Test-set model performance.",
        ),
        (
            "confusion_matrices.png",
            "Figure 3. Confusion matrices.",
        ),
    ]

    for filename, caption in figures:
        image_path = RESULTS / filename

        if not image_path.exists():
            print(f"Warning: figure not found: {image_path}")
            continue

        img = Image(str(image_path))
        img._restrictSize(
            A4[0] - 4 * cm,
            8 * cm,
        )

        # Keep the caption and graph together on the same page.
        story.append(Spacer(1, 8))
        story.append(KeepTogether([
            Paragraph(caption, styles["H2Custom"]),
            img,
        ]))


def main():
    if not REPORT.exists():
        raise FileNotFoundError(
            f"Markdown report not found: {REPORT}"
        )

    text = REPORT.read_text(encoding="utf-8")
    lines = text.splitlines()

    story = []
    i = 0
    first_heading = True

    while i < len(lines):
        line = lines[i].strip()

        if not line:
            i += 1
            continue

        # Markdown tables.
        if line.startswith("|"):
            table_lines = []

            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
            ):
                table_lines.append(lines[i])
                i += 1

            parse_table(table_lines, story)
            continue

        # Fenced code blocks.
        if line.startswith("```"):
            code_lines = []
            i += 1

            while (
                i < len(lines)
                and not lines[i].strip().startswith("```")
            ):
                code_lines.append(lines[i])
                i += 1

            escaped_code = [
                code_line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                for code_line in code_lines
            ]

            code_markup = "<br/>".join(escaped_code)

            story.append(Paragraph(
                code_markup,
                styles["CodeBlockCustom"],
            ))

            i += 1
            continue

        # Main heading.
        if line.startswith("# "):
            heading = clean_inline(line[2:])

            if first_heading:
                story.append(Spacer(1, 2 * cm))
                story.append(Paragraph(
                    heading,
                    styles["ReportTitle"],
                ))
                first_heading = False
            else:
                story.append(Paragraph(
                    heading,
                    styles["H1Custom"],
                ))

            i += 1
            continue

        # Section heading.
        if line.startswith("## "):
            story.append(Paragraph(
                clean_inline(line[3:]),
                styles["H1Custom"],
            ))
            i += 1
            continue

        # Subsection heading.
        if line.startswith("### "):
            story.append(Paragraph(
                clean_inline(line[4:]),
                styles["H2Custom"],
            ))
            i += 1
            continue

        # Bullet and numbered lists.
        if line.startswith("- ") or re.match(r"^\d+\.\s", line):
            while i < len(lines):
                item = lines[i].strip()

                if item.startswith("- "):
                    content = "&#8226; " + clean_inline(item[2:])
                elif re.match(r"^\d+\.\s", item):
                    content = clean_inline(item)
                else:
                    break

                story.append(Paragraph(
                    content,
                    styles["BodyCustom"],
                ))
                i += 1

            continue

        add_paragraph(story, line)
        i += 1

    # Append all three actual experiment figures.
    add_figures(story)

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="Machine Learning Mini-Project Report",
        author="PES University",
    )

    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer,
    )

    print(f"PDF created successfully: {OUTPUT}")


if __name__ == "__main__":
    main()

