"""PDF companion from inspected Artifact Tool renders; editable master remains PPTX."""
import argparse
from pathlib import Path
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--images',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
    c=Canvas(str(a.output),pagesize=(960,540))
    c.setTitle('Applebees Fase 2: control y explicación de descuentos')
    for i in range(1,9):
        source=a.images/f'slide-{i}.png'
        if not source.is_file():raise FileNotFoundError(source)
        c.drawImage(ImageReader(str(source)),0,0,width=960,height=540)
        c.showPage()
    c.save()


if __name__=='__main__':main()
