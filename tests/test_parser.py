from sec_rag.ingestion.parser import html_to_text, split_into_items

FILLER = "Lorem ipsum dolor sit amet. " * 20

HTML = f"""
<html><body>
<p>Table of Contents</p>
<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 7. MD&amp;A</p>
<p>Item 1. Business</p><p>We make widgets. {FILLER}</p>
<p>Item 1A. Risk Factors</p><p>Supply chain risk. {FILLER}</p>
<p>Item 7. Management's Discussion</p><p>Revenue grew 10%. {FILLER}</p>
</body></html>
"""


def test_html_to_text_strips_tags():
    text = html_to_text("<p>Hello&nbsp;<b>world</b></p><script>x()</script>")
    assert "Hello" in text and "world" in text
    assert "x()" not in text


def test_split_skips_table_of_contents():
    sections = split_into_items(html_to_text(HTML))
    by_item = {s.item: s for s in sections}
    assert list(by_item) == ["1", "1A", "7"]
    assert "We make widgets" in by_item["1"].text
    assert "Supply chain risk" in by_item["1A"].text
    assert by_item["1A"].title == "Risk Factors"


def test_split_falls_back_to_full_text():
    sections = split_into_items("No item headings in here at all.")
    assert len(sections) == 1 and sections[0].item == "FULL"
