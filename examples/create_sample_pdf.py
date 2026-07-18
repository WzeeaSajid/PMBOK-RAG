"""Generate a tiny original fixture, not a copy of the PMBOK Guide."""
from pathlib import Path

import fitz

if __name__ == '__main__':
    path = Path(__file__).with_name('sample.pdf')
    with fitz.open() as document:
        for title, text in [
            ('Project Scope', 'Project scope describes the work required to deliver the agreed product.'),
            ('Cost Performance Index',
             'The cost performance index (CPI) equals earned value divided by actual cost.'),
            ('Risk Register', 'A risk register records identified risks, their owners, and planned responses.'),
        ]:
            page = document.new_page()
            page.insert_text((50, 70), title, fontsize=16)
            page.insert_textbox(fitz.Rect(50, 100, 550, 200), text, fontsize=11)
        document.save(path)
    print(path)
