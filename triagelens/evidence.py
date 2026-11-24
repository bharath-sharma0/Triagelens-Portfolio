"""Read output evidence, retaining cell references and displayed precision."""
import json
from io import StringIO
import pandas as pd
from .config import ROOT

def outputs():
    return json.loads((ROOT/'evidence/outputs.json').read_text())

def tables(cell):
    result=[]
    for path in sorted((ROOT/'evidence').glob(f'cell_{cell}_*.html')):
        for df in pd.read_html(StringIO(path.read_text()),index_col=0):
            result.append((path.name,df))
    return result
