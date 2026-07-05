


from pathlib import Path


def load_data() -> str :

    texts = []
    for file in Path("data").iterdir():
        if file.suffix == ".txt":
            texts.append(file.read_text(encoding="utf-8"))

        elif file.suffix == ".parquet":
            df = pd.read_parquet(file)
            texts.extend(df["text"].dropna().tolist())

    text = "\n\n".join(texts)
    return text


