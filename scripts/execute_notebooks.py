"""
Deterministic In-Process Notebook Executor for NiftyAlpha
=========================================================
Executes Jupyter notebooks (.ipynb) cell-by-cell in-process without 
ZeroMQ, ipykernel, or localhost socket dependencies. Captures stdout, 
stderr, rich DataFrame display tables, and high-resolution Matplotlib/SHAP 
plots as base64 PNGs directly into the notebook JSON specification.
"""
import ast
import base64
import io
import json
import os
import sys
import uuid
from pathlib import Path

# Ensure headless matplotlib backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

NOTEBOOK_FILES = [
    NOTEBOOKS_DIR / "01_eda_and_data.ipynb",
    NOTEBOOKS_DIR / "02_model_experiments.ipynb",
    NOTEBOOKS_DIR / "03_trading_simulation.ipynb",
]


class InProcessNotebookExecutor:
    """Executes code cells in-process, capturing rich display outputs and stdout."""

    def __init__(self, notebook_path: Path):
        self.notebook_path = notebook_path
        self.namespace = {
            "__name__": "__main__",
            "pd": pd,
            "np": np,
            "plt": plt,
        }
        self.cell_outputs = []
        self.execution_count = 0

    def _hook_environment(self):
        """Hooks plt.show and display() to capture outputs."""
        def mock_show(*args, **kwargs):
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
                buf.seek(0)
                b64_img = base64.b64encode(buf.read()).decode("utf-8")
                self.cell_outputs.append({
                    "output_type": "display_data",
                    "data": {
                        "image/png": b64_img,
                        "text/plain": f"<Figure size {fig.get_size_inches()} with axes>"
                    },
                    "metadata": {}
                })
                plt.close(fig)

        def mock_display(*args):
            for obj in args:
                if isinstance(obj, pd.DataFrame):
                    self.cell_outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "text/plain": obj.to_string(),
                            "text/html": obj.to_html(classes="dataframe table table-striped")
                        },
                        "metadata": {}
                    })
                elif isinstance(obj, pd.Series):
                    self.cell_outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "text/plain": obj.to_string()
                        },
                        "metadata": {}
                    })
                else:
                    self.cell_outputs.append({
                        "output_type": "display_data",
                        "data": {
                            "text/plain": str(obj)
                        },
                        "metadata": {}
                    })

        plt.show = mock_show
        self.namespace["plt"] = plt
        self.namespace["display"] = mock_display

    def execute_cell(self, source_code: str) -> list:
        """Executes a single code cell and returns a list of nbformat output objects."""
        self.cell_outputs = []
        self.execution_count += 1
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        try:
            parsed = ast.parse(source_code)
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                last_expr = parsed.body.pop()
                if parsed.body:
                    exec(compile(parsed, filename="<nb_cell>", mode="exec"), self.namespace)
                result = eval(
                    compile(ast.Expression(last_expr.value), filename="<nb_cell>", mode="eval"),
                    self.namespace
                )
                if result is not None:
                    if isinstance(result, pd.DataFrame):
                        self.cell_outputs.append({
                            "output_type": "execute_result",
                            "execution_count": self.execution_count,
                            "data": {
                                "text/plain": result.to_string(),
                                "text/html": result.to_html(classes="dataframe table table-striped")
                            },
                            "metadata": {}
                        })
                    else:
                        self.cell_outputs.append({
                            "output_type": "execute_result",
                            "execution_count": self.execution_count,
                            "data": {
                                "text/plain": repr(result)
                            },
                            "metadata": {}
                        })
            else:
                exec(compile(parsed, filename="<nb_cell>", mode="exec"), self.namespace)
                
            # Flush any pending matplotlib figures
            if plt.get_fignums():
                plt.show()

        except Exception as err:
            err_msg = f"{type(err).__name__}: {str(err)}"
            print(err_msg, file=stderr_capture)
            self.cell_outputs.append({
                "output_type": "error",
                "ename": type(err).__name__,
                "evalue": str(err),
                "traceback": [err_msg]
            })
            raise err
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Process captured stdout
        stdout_text = stdout_capture.getvalue()
        if stdout_text:
            lines = stdout_text.splitlines(keepends=True)
            self.cell_outputs.insert(0, {
                "output_type": "stream",
                "name": "stdout",
                "text": lines
            })

        # Process captured stderr
        stderr_text = stderr_capture.getvalue()
        if stderr_text:
            lines = stderr_text.splitlines(keepends=True)
            self.cell_outputs.append({
                "output_type": "stream",
                "name": "stderr",
                "text": lines
            })

        return self.cell_outputs

    def run(self):
        """Executes all code cells in the notebook and writes back updated JSON."""
        print(f"\n=======================================================")
        print(f" In-Process Execution: {self.notebook_path.name}")
        print(f"=======================================================")
        
        with open(self.notebook_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        orig_cwd = os.getcwd()
        # Switch working directory to notebooks/ so relative paths like ../data resolve
        os.chdir(self.notebook_path.parent)
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        self._hook_environment()

        try:
            for idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    source = cell.get("source", "")
                    if isinstance(source, list):
                        source = "".join(source)
                    
                    if not source.strip():
                        cell["outputs"] = []
                        cell["execution_count"] = None
                        continue
                        
                    print(f"  [Cell {idx+1}/{len(nb['cells'])}] Executing code cell...")
                    outputs = self.execute_cell(source)
                    cell["outputs"] = outputs
                    cell["execution_count"] = self.execution_count
                    if "id" not in cell:
                        cell["id"] = str(uuid.uuid4())[:8]

            print(f"[SUCCESS] All code cells executed in {self.notebook_path.name}")

        finally:
            os.chdir(orig_cwd)

        with open(self.notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)

        print(f"[SAVED] Persisted cell outputs and figures to {self.notebook_path.name}")


def main():
    targets = [Path(sys.argv[1])] if len(sys.argv) > 1 else NOTEBOOK_FILES
    for nb_path in targets:
        if not nb_path.exists():
            print(f"[ERROR] Notebook not found: {nb_path}")
            sys.exit(1)
        executor = InProcessNotebookExecutor(nb_path)
        executor.run()
    print("\n>>> All research notebooks successfully executed and persisted.")


if __name__ == "__main__":
    main()
