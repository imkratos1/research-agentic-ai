from datetime import datetime
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph, Spacer,
                                Table, TableStyle)
from reportlab.platypus.tableofcontents import TableOfContents

from agent import charts, writer

# DejaVu Sans ships with matplotlib and covers symbols like ≥ and curly quotes that the built-in PDF fonts lack
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DejaVu", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Italic", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold", italic="DejaVu-Italic", boldItalic="DejaVu-Bold")

INDIGO = colors.HexColor("#1A237E")
ACCENT = colors.HexColor("#3949AB")
INK = colors.HexColor("#263238")
MUTED = colors.HexColor("#607D8B")
TINT = colors.HexColor("#E8EAF6")
RULE = colors.HexColor("#C5CAE9")
LINK = "#3949AB"
BADGES = {"VERIFIED": ("#C8E6C9", "#1B5E20"), "LIKELY": ("#DCEDC8", "#33691E"),
          "CONTESTED": ("#FFE0B2", "#7A3E00"), "UNVERIFIED": ("#FFCDD2", "#8E1B1B")}
PAGE_WIDTH = A4[0] - 4.4 * cm


def make_styles():
    body = ParagraphStyle("body", fontName="DejaVu", fontSize=10, leading=14.5, textColor=INK, spaceAfter=7)
    return {
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8.5, leading=11.5, textColor=MUTED),
        "kicker": ParagraphStyle("kicker", parent=body, fontName="DejaVu-Bold", fontSize=9, textColor=ACCENT, spaceAfter=6),
        "title": ParagraphStyle("title", parent=body, fontName="DejaVu-Bold", fontSize=24, leading=30, textColor=INDIGO, spaceAfter=12),
        "subtitle": ParagraphStyle("subtitle", parent=body, fontSize=12, leading=16, textColor=MUTED, spaceAfter=20),
        "h1": ParagraphStyle("h1", parent=body, fontName="DejaVu-Bold", fontSize=16, leading=20, textColor=INDIGO, spaceBefore=4, spaceAfter=10),
        "h2": ParagraphStyle("h2", parent=body, fontName="DejaVu-Bold", fontSize=11.5, leading=15, textColor=ACCENT, spaceBefore=10, spaceAfter=5),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=3),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8, leading=10.5, spaceAfter=0),
        "head": ParagraphStyle("head", parent=body, fontName="DejaVu-Bold", fontSize=8, leading=10.5, spaceAfter=0, textColor=INDIGO),
        "ref": ParagraphStyle("ref", parent=body, fontSize=8.5, leading=11.5, leftIndent=22, firstLineIndent=-22, spaceAfter=5),
        "toc1": ParagraphStyle("toc1", parent=body, fontName="DejaVu-Bold", fontSize=10.5, leading=16),
        "toc2": ParagraphStyle("toc2", parent=body, fontSize=9.5, leading=13, leftIndent=16),
    }


class ReportDocument(BaseDocTemplate):
    def __init__(self, path, title):
        super().__init__(str(path), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=2 * cm,
                         bottomMargin=2.2 * cm, title=title, author="DeepResearch Agent")
        self.short_title = title if len(title) < 95 else title[:92] + "..."
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=self.draw_footer)])

    def draw_footer(self, canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.line(self.leftMargin, 1.55 * cm, A4[0] - self.rightMargin, 1.55 * cm)
        canvas.setFont("DejaVu", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, 1.15 * cm, self.short_title)
        canvas.drawRightString(A4[0] - self.rightMargin, 1.15 * cm, f"Page {doc.page}")
        canvas.restoreState()

    def afterFlowable(self, flowable):
        # every h1/h2 heading becomes a table of contents entry and a PDF bookmark
        if isinstance(flowable, Paragraph) and flowable.style.name in ("h1", "h2"):
            level = 0 if flowable.style.name == "h1" else 1
            text = flowable.getPlainText()
            key = f"heading-{self.seq.nextf('heading')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(text, key, level=level, closed=False)
            self.notify("TOCEntry", (level, text, self.page, key))


def badge(label):
    back, ink = BADGES[label]
    return f'<font backColor="{back}" color="{ink}" name="DejaVu-Bold" size="7.5">&nbsp;{label}&nbsp;</font>'


def rich(text, numbers):
    # escape the text for ReportLab, then turn [S3] into a clickable [n] that jumps to the reference list
    safe = escape(text)
    return writer.CITATION.sub(
        lambda m: f'<a href="#ref{numbers[m.group(1)]}" color="{LINK}">[{numbers[m.group(1)]}]</a>' if m.group(1) in numbers else "",
        safe)


def ref_links(source_ids, numbers):
    return " ".join(f'<a href="#ref{numbers[s]}" color="{LINK}">[{numbers[s]}]</a>' for s in source_ids if s in numbers) or "-"


def styled_table(rows, widths, s):
    table = Table([[c if not isinstance(c, str) else Paragraph(c, s["cell"]) for c in row] for row in rows], colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), TINT),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def header_row(names, s):
    return [Paragraph(n, s["head"]) for n in names]


def label_counts(groups):
    return {label: sum(1 for g in groups if g.confidence == label) for label in BADGES}


def title_page(state, s):
    counts = label_counts(state.groups)
    websites = len({x.domain for x in state.sources})
    facts = [
        ("Research questions", len(state.questions)),
        ("Searches run", len(state.searches)),
        ("Sources read", f"{len(state.sources)} from {websites} websites"),
        ("Claims extracted", len(state.claims)),
        ("Claim groups", " · ".join(f"{k.title()} {v}" for k, v in counts.items())),
        ("Depth", state.depth),
        ("Generated", datetime.now().strftime("%d %B %Y, %H:%M")),
    ]
    rows = [[Paragraph(k, s["head"]), Paragraph(escape(str(v)), s["cell"])] for k, v in facts]
    table = Table(rows, colWidths=[4.2 * cm, PAGE_WIDTH - 4.2 * cm])
    table.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.3, RULE), ("TOPPADDING", (0, 0), (-1, -1), 5),
                               ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    disclaimer = ("This report was written by an AI research agent. Every finding is linked to its sources and has a confidence "
                  "label that estimates how well independent, trusted sources support it. The labels are not proof: check "
                  "important facts in the cited sources before relying on them.")
    return [Spacer(1, 2.5 * cm), Paragraph("DEEPRESEARCH AGENT · RESEARCH REPORT", s["kicker"]),
            Paragraph(escape(state.report.title), s["title"]), Paragraph(f"Topic: {escape(state.topic)}", s["subtitle"]),
            table, Spacer(1, 1.2 * cm), Paragraph(disclaimer, s["small"]), PageBreak()]


def method_section(state, s):
    websites = len({x.domain for x in state.sources})
    story = [Paragraph("2. Research questions and method", s["h1"])]
    for q in state.questions:
        story.append(Paragraph(f"<b>{q.id}</b>&nbsp; {escape(q.text)}", s["bullet"], bulletText="•"))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"The Researcher agent ran {len(state.searches)} searches on DuckDuckGo, Wikipedia and OpenAlex, opened "
        f"{len(state.sources)} sources from {websites} different websites and gave each a trust tier. "
        f"{len(state.failed_urls)} pages could not be opened. The Extractor pulled {len(state.claims)} single-fact claims "
        "from the sources and kept a claim only if its supporting quote was found in the source text. "
        "The Verifier grouped claims that say the same thing (embeddings from gemini-embedding-2, cosine similarity ≥ 0.85), "
        "asked Gemini whether each source supports or contradicts each group, counted independent websites "
        "(copies of the same story count once) and gave every group a confidence label.", s["body"]))
    rules = [header_row(["Label", "Rule used by the Verifier"], s),
             [badge("VERIFIED"), "2 or more independent websites agree, at least one is tier 1 or 2, and no tier 1 or 2 source disagrees"],
             [badge("LIKELY"), "Supported by one tier 1 source, or by 2 or more independent lower-tier websites"],
             [badge("CONTESTED"), "A tier 1 or 2 source disagrees with the claim; both sides are shown in section 4"],
             [badge("UNVERIFIED"), "Only one low-trust source; a targeted search was tried, and the claim is left out of the findings"]]
    tiers = [header_row(["Tier", "Weight", "Type of website"], s),
             ["1", "0.9", "Government, university, international organisation, scholarly journal"],
             ["2", "0.7", "Reputable news agency or reference work (Wikipedia 0.6, arXiv preprints 0.75)"],
             ["3", "0.4", "Any other website"],
             ["4", "0.1", "Forum, social media or user-generated content (never above 0.2)"]]
    # small tables stay on one page together with their heading
    story += [KeepTogether([Paragraph("Confidence labels", s["h2"]), styled_table(rules, [2.8 * cm, PAGE_WIDTH - 2.8 * cm], s)]),
              KeepTogether([Paragraph("Trust tiers", s["h2"]), styled_table(tiers, [1.2 * cm, 1.8 * cm, PAGE_WIDTH - 3.0 * cm], s)])]
    return story


def source_quality_section(state, numbers, s):
    story = [Paragraph("5. Source quality", s["h1"])]
    chart_width = PAGE_WIDTH / 2 - 0.2 * cm
    images = [Image(BytesIO(charts.tier_chart(state.sources)), width=chart_width, height=chart_width / 2),
              Image(BytesIO(charts.confidence_chart(state.groups)), width=chart_width, height=chart_width / 2)]
    story.append(Table([images], colWidths=[PAGE_WIDTH / 2, PAGE_WIDTH / 2]))
    story.append(Spacer(1, 8))
    ordered = sorted(state.sources, key=lambda x: numbers[x.id])
    rows = [header_row(["Ref", "Title", "Website", "Tier", "Read"], s)]
    for src in ordered:
        rows.append([f'<a href="#ref{numbers[src.id]}" color="{LINK}">[{numbers[src.id]}]</a>', escape(src.title[:110] or src.url),
                     escape(src.domain), str(src.tier), "abstract" if src.abstract_only else "full text"])
    story.append(styled_table(rows, [1.1 * cm, 8.9 * cm, 3.7 * cm, 1.0 * cm, 1.9 * cm], s))
    return story


def references_section(state, numbers, s):
    story = [Paragraph("7. References", s["h1"])]
    accessed = datetime.fromisoformat(state.started).strftime("%d %B %Y")
    for src in sorted(state.sources, key=lambda x: numbers[x.id]):
        n = numbers[src.id]
        url = escape(src.url, {'"': "&quot;"})
        shown = escape(src.url if len(src.url) < 95 else src.url[:92] + "...")
        date = f", {escape(src.date)}" if src.date else ""
        author = f"{escape(src.author)}. " if src.author and len(src.author) < 80 else ""
        story.append(Paragraph(
            f'<a name="ref{n}"/>[{n}]&nbsp; {author}<b>{escape(src.title or src.domain)}</b>. <i>{escape(src.domain)}</i>{date}. '
            f'Tier {src.tier}, weight {src.weight}. <a href="{url}" color="{LINK}">{shown}</a> (accessed {accessed})', s["ref"]))
    return story


def appendix_section(state, numbers, s):
    story = [PageBreak(), Paragraph("Appendix A. Claim verification table", s["h1"]),
             Paragraph("Every group of matching claims found by the Verifier, with the sources that support or dispute it.", s["small"])]
    rows = [header_row(["Claim", "Label", "Supported by", "Disputed by", "Sites"], s)]
    for g in state.groups:
        claim = g.main_claim if len(g.main_claim) < 260 else g.main_claim[:257] + "..."
        rows.append([f"<b>{g.question_id}</b> {escape(claim)}", badge(g.confidence), ref_links(g.supporting_sources, numbers),
                     ref_links(g.contradicting_sources, numbers), str(g.independent_sites)])
    story.append(styled_table(rows, [8.9 * cm, 2.6 * cm, 2.5 * cm, 1.8 * cm, 0.8 * cm], s))
    return story


def build_pdf(state, path):
    if state.report is None:
        raise ValueError("The report has not been written yet")
    s = make_styles()
    numbers = writer.reference_numbers(state)
    report = state.report

    toc = TableOfContents()
    toc.levelStyles = [s["toc1"], s["toc2"]]
    story = title_page(state, s)
    story += [Paragraph("Contents", s["title"]), toc, PageBreak()]

    story.append(Paragraph("1. Executive summary", s["h1"]))
    for finding in report.summary:
        story.append(Paragraph(f"{badge(finding.confidence)}&nbsp; {rich(finding.text, numbers)}", s["bullet"], bulletText="•"))
    if not report.summary:
        story.append(Paragraph("No finding passed the cross-validation rules, see the appendix for the unchecked claims.", s["body"]))

    story += method_section(state, s)

    story.append(Paragraph("3. Findings", s["h1"]))
    for i, section in enumerate(report.sections, start=1):
        story.append(Paragraph(f"3.{i}&nbsp; {escape(section.heading)}", s["h2"]))
        for p in section.paragraphs:
            story.append(Paragraph(rich(p, numbers), s["body"]))

    story.append(Paragraph("4. Contested points", s["h1"]))
    for point in report.contested:
        story.append(Paragraph(rich(point, numbers), s["bullet"], bulletText="•"))
    if not report.contested:
        story.append(Paragraph("No credible source disagreed with a finding in this run.", s["body"]))

    story += source_quality_section(state, numbers, s)

    story.append(Paragraph("6. Limitations", s["h1"]))
    for note in report.limitations:
        story.append(Paragraph(escape(note), s["bullet"], bulletText="•"))

    story += references_section(state, numbers, s)
    story += appendix_section(state, numbers, s)

    document = ReportDocument(path, report.title)
    document.multiBuild(story)
    return Path(path)
