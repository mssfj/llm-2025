from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Process the JSONL file in chunks to avoid loading everything into memory.
INPUT_PATH = Path("dataset/openmathinstruct-2_structured/openmathinstruct-2_structured-1000.jsonl")
OUTPUT_PATH = Path("dataset/openmathinstruct-2_structured/data/train-00000-of-00001.parquet")
CHUNK_SIZE = 100


def main() -> None:
    # Start fresh so reruns don't append to an existing file.
    OUTPUT_PATH.unlink(missing_ok=True)

    writer = None
    rows_written = 0
    total_rows = sum(1 for _ in INPUT_PATH.open())
    if total_rows == 0:
        print("No data found to write.")
        return
    print(f"Processing {total_rows} rows from {INPUT_PATH}...")

    with pd.read_json(INPUT_PATH, lines=True, chunksize=CHUNK_SIZE) as reader:
        for chunk in reader:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PATH, table.schema)
            writer.write_table(table)
            rows_written += len(chunk)
            pct = rows_written / total_rows
            print(f"\rProgress: {rows_written}/{total_rows} rows ({pct:.1%})", end="", flush=True)

    if writer is not None:
        writer.close()
    print(f"\nWrote {rows_written} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
