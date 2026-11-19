"""Write CSV tables in row batches while preserving an existing destination on error."""
from .model_archive import atomic_output
from .work_progress import report


def write_csv(table, path):
    with atomic_output(path) as tmp, tmp.open('w', encoding='utf-8-sig', newline='') as out:
        size = max(1, 100000 // max(1,len(table.columns)))
        for start in range(0, max(1,len(table)), size):
            end = min(start+size, len(table))
            table.iloc[start:end].to_csv(out, index=False, header=start==0)
            report(end, len(table), 'Строки CSV')
