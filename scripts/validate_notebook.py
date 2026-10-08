"""Execute the notebook without changing its checked-in, output-free copy."""
import argparse
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", required=True, help="Ruta o glob de los checks")
    parser.add_argument("--raw", help="Carpeta de catálogos; opcional")
    parser.add_argument("--output", default="outputs/validation/sample")
    parser.add_argument("--price-mode", default="unresolved")
    parser.add_argument("--discount-mode", default="unresolved")
    parser.add_argument("--gratuity-mode", default="unresolved")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = output/"kernel"
    runtime.mkdir(exist_ok=True)
    plot_cache = output/"matplotlib"
    plot_cache.mkdir(exist_ok=True)
    os.environ.update(APPLEBEES_CHECKS=str(Path(args.checks).absolute()), APPLEBEES_OUTPUT=str(output/"eda"),
        APPLEBEES_PRICE_MODE=args.price_mode, APPLEBEES_DISCOUNT_MODE=args.discount_mode,
        APPLEBEES_GRATUITY_MODE=args.gratuity_mode, JUPYTER_RUNTIME_DIR=str(runtime), MPLCONFIGDIR=str(plot_cache))
    if args.raw:
        os.environ["APPLEBEES_RAW"] = str(Path(args.raw).resolve())
    nb = nbformat.read(root/"notebooks"/"eda_descuentos.ipynb", as_version=4)
    nbformat.validate(nb)
    for index,cell in enumerate(nb.cells):
        if cell.cell_type=="code":
            compile(cell.source, f"cell-{index}", "exec")
    km = KernelManager(kernel_name="python3", connection_file=str(runtime/"connection.json"))
    km.kernel_spec.argv = [sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
    client = NotebookClient(nb, km=km, timeout=300, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        if client.kc is not None:
            client.kc.stop_channels()
    executed = output/"eda_descuentos_ejecutado.ipynb"
    nbformat.write(nb,executed)
    errors = [out for c in nb.cells if c.cell_type=="code" for out in c.outputs if out.output_type=="error"]
    if errors:
        raise RuntimeError("Hay celdas con errores")
    summary = {"code_cells_executed": sum(c.cell_type=="code" for c in nb.cells), "errors": len(errors),
        "executed_notebook": str(executed)}
    (output/"validation.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
