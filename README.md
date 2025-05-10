# attribute

Reproduction of Anthropic's [Attribution Graphs](https://transformer-circuits.pub/2025/attribution-graphs/methods.html), heavily inspired by https://github.com/jacobdunefsky/transcoder_circuits.

```
uv venv --seed
uv sync
uv run python -m attribute
cd attribution-graphs-frontend
uv run python serve.py 9999
ngrok http 9999
```
