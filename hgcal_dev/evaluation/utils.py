from PyPDF2 import PdfFileMerger
from pathlib import Path
import sys
sys.path.append('/home/alexj/hgcal-dev/hgcal_dev/evaluation/glasbey')
from glasbey import Glasbey
import numpy as np
import logging

def merge_pdfs(pdfs: list, output_path: Path):
    merger = PdfFileMerger()
    for pdf in pdfs:
        merger.append(pdf)
    merger.write(str(output_path))

def get_palette(instances: np.array):
    logging.info('Generating color palette...')
    gb = Glasbey(base_palette=[(255, 0, 0)], lightness_range=(0, 90))
    size = np.unique(instances).shape[0]
    p = gb.generate_palette(size=size)
    rgb = gb.convert_palette_to_rgb(p)
    logging.info('Generation complete!')
    return [f'rgb{c[0],c[1],c[2]}' for c in rgb]

def main():
    print(get_palette(np.array([0, 1, 2, 0, 3, 0, 0, 4, 0, 1, 1, 2])))

if __name__ == '__main__':
    main()