# Blog Writing Agent

A LangGraph orchestrator–worker agent that turns a single topic string into a finished technical blog post in Markdown.

## How it works

```
START → orchestrator → (fan-out via Send) → worker ×N → reducer → END
```

- **orchestrator** — asks the LLM for a `Plan` (structured output via Pydantic): a blog title, audience, tone, and 5–7 `Task` sections, each with a goal, 3–5 concrete bullets, a target word count, and a section type.
- **worker** — one parallel invocation per section, writing that section as Markdown against its goal, bullets, and word target. Results are merged with `operator.add` on the `sections` state key.
- **reducer** — joins the sections under an H1 title and writes the post to a `.md` file.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install langgraph langchain langchain-openai python-dotenv pydantic
```

Copy `.env.example` to `.env` and add your [OpenRouter](https://openrouter.ai) key:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
```

Then run [`basic_bwa.ipynb`](basic_bwa.ipynb) top to bottom:

```python
out = app.invoke({"topic": "Write a blog on Self Attention", "sections": []})
```

## Note on the free tier

The free Nvidia endpoint intermittently returns HTTP 200 with an error body and no `choices`, which the OpenAI SDK surfaces as a confusing `TypeError: 'NoneType' object is not iterable`. It hits the structured-output call in `orchestrator` roughly half the time. Both LLM call sites are wrapped in `.with_retry(**RETRY)` to ride it out. Note that `ChatOpenAI(max_retries=...)` does *not* help, since it only retries non-2xx responses.

## Sample output

See the generated posts in the repo root.
