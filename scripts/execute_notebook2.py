import asyncio
import sys

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbclient import NotebookClient

notebook_path = r"c:\Users\Vaishnavi\Downloads\6ab10eb3b23ba_student_resource\notebooks\02_preprocessing_validation.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
print("Executing notebook 02 cells...")
client.execute()

with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully executed and updated notebook 02 with full outputs!")
