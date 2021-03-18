from PyPDF2 import PdfFileMerger
from pathlib import Path

def merge_pdfs(pdfs: list, output_path: Path):
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(str(output_path))