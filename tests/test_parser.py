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


def test_two_digit_items_are_not_read_as_item_1():
    text = html_to_text(
        f"<p>Item 1. Business</p><p>Widgets. {FILLER}</p>"
        f"<p>Item 10. Directors</p><p>Board info. {FILLER}</p>"
        f"<p>Item 15. Exhibits</p><p>Exhibit list. {FILLER}</p>"
    )
    by_item = {s.item: s for s in split_into_items(text)}
    assert list(by_item) == ["1", "10", "15"]
    assert "Widgets" in by_item["1"].text


def test_running_page_headers_are_ignored():
    # Some filers (e.g. Microsoft) print "Item N" at the top of every page.
    text = html_to_text(
        f"<p>Item 1</p><p>ITEM 1. BUSINESS</p><p>Widgets. {FILLER}</p>"
        f"<p>Item 1</p><p>More business. {FILLER}</p>"
        f"<p>Item 1A</p><p>ITEM 1A. RISK FACTORS</p><p>Supply risk. {FILLER}</p>"
        f"<p>Item 1A</p><p>More risks. {FILLER}</p>"
    )
    by_item = {s.item: s for s in split_into_items(text)}
    assert list(by_item) == ["1", "1A"]
    assert "Widgets" in by_item["1"].text and "More business" in by_item["1"].text
    assert "Supply risk" in by_item["1A"].text and "More risks" in by_item["1A"].text


def test_split_falls_back_to_full_text():
    sections = split_into_items("No item headings in here at all.")
    assert len(sections) == 1 and sections[0].item == "FULL"


LEASE_TABLE = """
<table>
<tr><td></td><td>Operating Leases</td><td></td><td>Finance Leases</td><td></td><td>Total</td></tr>
<tr><td>Less: Imputed interest</td><td>(</td><td>1,984</td><td>)</td>
    <td>(</td><td>376</td><td>)</td><td>(</td><td>2,360</td><td>)</td></tr>
<tr><td>Total lease liabilities</td><td>$</td><td>11,534</td><td></td>
    <td>$</td><td>896</td><td></td><td>$</td><td>12,430</td></tr>
<tr><td></td><td></td><td></td></tr>
</table>
"""


def test_tables_keep_one_row_per_line_with_cells_glued():
    text = html_to_text(f"<p>Lease maturities:</p>{LEASE_TABLE}<p>After the table.</p>")
    assert text == (
        "Lease maturities:\n\n"
        "Operating Leases | Finance Leases | Total\n"
        "Less: Imputed interest | (1,984) | (376) | (2,360)\n"
        "Total lease liabilities | $11,534 | $896 | $12,430\n\n"
        "After the table."
    )


def test_inline_tags_do_not_break_sentences():
    # Inline XBRL wraps each number in its own tag.
    html = ('<div><span>The lease term was </span><ix:nonNumeric name="x">10.3</ix:nonNumeric>'
            '<span> years and the rate was </span><ix:nonFraction name="y">3.1</ix:nonFraction>'
            '<span>%.</span></div><div>Next paragraph.</div>')
    assert html_to_text(html) == (
        "The lease term was 10.3 years and the rate was 3.1%.\n\nNext paragraph."
    )


def test_item_headings_inside_tables_are_still_found():
    text = html_to_text(
        f"<table><tr><td>Item 1A.</td><td>Risk Factors</td></tr></table><p>Supply risk. {FILLER}</p>"
        f"<table><tr><td>Item 7.</td><td>MD&amp;A</td></tr></table><p>Revenue grew. {FILLER}</p>"
    )
    by_item = {s.item: s for s in split_into_items(text)}
    assert list(by_item) == ["1A", "7"]
    assert "Supply risk" in by_item["1A"].text
