"""Validated UTF-8 query uploads and optional original CLINC150 dataset."""
import io
import json
import pandas as pd

MAX_BYTES = 2_000_000
MAX_CHARS = 10_000
MAX_ROWS = 200

def validate_query(query):
    if not isinstance(query, str) or not query.strip():
        raise ValueError('Enter a non-empty text query.')
    if len(query) > MAX_CHARS or '\x00' in query:
        raise ValueError('Queries must have at most 10,000 characters and contain no null bytes.')
    return query  # Do not strip/lowercase/clean outside the fitted vectorizer.

def parse_upload(raw, filename):
    if len(raw) > MAX_BYTES:
        raise ValueError('Upload limit is 2 MB.')
    try:
        text = raw.decode('utf-8-sig')
        suffix = filename.lower().rsplit('.',1)[-1]
        if suffix == 'txt':
            queries = [text]
        elif suffix == 'csv':
            df = pd.read_csv(io.StringIO(text), keep_default_na=False)
            if 'text' not in df.columns:
                raise ValueError('CSV must include a text column.')
            queries = df['text'].tolist()
        elif suffix == 'json':
            rows = json.loads(text)
            if not isinstance(rows,list):
                raise ValueError('Query JSON must be a list of strings or objects with a text field.')
            queries = [r.get('text') if isinstance(r,dict) else r for r in rows]
        else:
            raise ValueError('Supported query files: .txt, .csv, .json.')
        if not 1 <= len(queries) <= MAX_ROWS:
            raise ValueError('Upload between 1 and 200 queries.')
        return [validate_query(q) for q in queries]
    except (UnicodeError, json.JSONDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Unable to read file. Use UTF-8 and the documented query schema.') from exc

def load_dataset(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data,dict):
        raise ValueError('Dataset must be a CLINC150 split dictionary.')
    expected = {'train':15000,'val':3000,'test':4500,'oos_train':100,'oos_val':100,'oos_test':1000}
    for split,size in expected.items():
        rows=data.get(split)
        if not isinstance(rows,list) or len(rows)!=size:
            raise ValueError(f'{split} must contain {size} records for this research configuration.')
        for row in rows:
            if not isinstance(row,list) or len(row)!=2 or not isinstance(row[1],str):
                raise ValueError(f'Invalid [text, intent] record in {split}.')
            validate_query(row[0])
            if split.startswith('oos') and row[1]!='oos':
                raise ValueError(f'{split} labels must be oos.')
    labels={r[1] for r in data['train']}
    if len(labels)!=150 or 'oos' in labels:
        raise ValueError('Expected 150 known training intents.')
    if any(r[1] not in labels for s in ('val','test') for r in data[s]):
        raise ValueError('Unknown intent label in known validation/test data.')
    return data
