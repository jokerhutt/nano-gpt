
## Running the Model


Clone Repo
```
git clone https://github.com/jokerhutt/nano-gpt.git
cd nano-gpt
uv sync
mkdir checkpoints/jgpt

git clone https://huggingface.co/Jokerhut/jgpt-reddit checkpoints/jgpt

uv run src/inference/generate.py
```
