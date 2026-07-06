from pathlib import Path
import bz2
import json
import pandas as pd


def load_data() -> str:

    texts = []

    for file in Path("data").rglob("*"):

        if file.suffix == ".txt":
            texts.append(file.read_text(encoding="utf-8"))

        elif file.suffix == ".parquet":
            df = pd.read_parquet(file)
            texts.extend(df["text"].dropna().tolist())

        elif file.suffix == ".bz2":
            with bz2.open(file, "rt", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)

                    body = obj.get("body")

                    if body and body not in ("[deleted]", "[removed]"):
                        texts.append(body)

    return "\n\n".join(texts)
