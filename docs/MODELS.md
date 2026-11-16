# Models, providers and costs

`--model <provider>:<model>` picks a provider from `config/providers.json`. A name without `:` is an Anthropic
model. The name splits on the first `:` only (`local:qwen2.5vl:7b` is model `qwen2.5vl:7b` of provider `local`).

| Provider | Key variable | Cost | `--effort` |
|---|---|---|---|
| (none), `anthropic` | `ANTHROPIC_API_KEY` | price table in `src/chd_client/models/anthropic.py` | `output_config.effort` |
| `openrouter` | `OPENROUTER_API_KEY` | reported by OpenRouter | `reasoning.effort` |
| `openai` | `OPENAI_API_KEY` | estimated from the prices you add | not sent unless configured |
| `local` | `CHD_LOCAL_API_KEY` (optional) | reported by the gateway (0.0) | not sent |

Keys come from your shell or from `.env` in the CHD-Bench folder (`KEY=value` lines, gitignored). `./run.sh` and the
batch runner read `.env`; `uv run run-episode` alone does not.

## Prices

**Check prices before a paid run.** Anthropic prices are in `PRICES` in `src/chd_client/models/anthropic.py`, with
the date they were checked in `PRICES_SOURCE`. A model without a price is refused before the run starts.

A run never invents a cost. A provider either reports the cost of each reply (`"source": "reported"`) or has a price
for each model (`"source": "estimated"`, computed from the reply's token counts). A reply without a cost stops the
run (`cost_unknown`) before any of its tool calls run.

## Adding a provider

Add an entry to `config/providers.local.json` (gitignored; same format as `config/providers.json`; an entry there
replaces the one of the same name). For example, an OpenAI-compatible vendor:

```json
{
  "providers": {
    "together": {
      "kind": "openai_compatible",
      "base_url": "https://api.together.xyz/v1",
      "api_key_env": "TOGETHER_API_KEY",
      "effort": null,
      "cost": {"source": "estimated",
               "prices": {"<model>": {"input_per_mtok": 0.0, "output_per_mtok": 0.0}}}
    }
  }
}
```

Replace `<model>` and the prices with the vendor's values, then use `--model 'together:<model>'`. Other fields:

- `base_url_env`: a variable that holds the URL (it wins over `base_url`).
- `api_key_env: null` (no key) and `api_key_optional`.
- `headers` (static headers, recorded in the run; credential headers are refused here) and `headers_env` (headers
  whose values come from variables, never recorded).
- `stream`, `extra_body` (extra request fields), `effort` (`null` or `"reasoning_effort"`).
- `max_tokens_param` (`"max_completion_tokens"` for OpenAI) and `max_output_tokens` (`{"<model>": <tokens>}`).
- For Gemini: `"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/"`,
  `"api_key_env": "GEMINI_API_KEY"`. Tool-call fields such as thought signatures are sent back as received.

`--providers <file>` or `$CHD_PROVIDERS` selects another registry file. An unknown key or provider is an error.

## Models you host yourself

A model you run yourself goes through `chd-gateway`, an OpenAI-compatible endpoint in this repository. It supports
the built-in Transformers engine, vLLM, and attaching to a server you already run (e.g. Ollama), on this machine or
as SLURM jobs. Terminal 1:

```bash
uv sync --frozen --inexact --extra transformers
uv run chd-gateway serve --profile config/gateway/qwen2.5-vl-3b.transformers.local.yaml --port 8098 \
  --workdir data/gateway
```

Terminal 2 (the model name is `local:` plus the profile's `name`):

```bash
export CHD_LOCAL_BASE_URL=http://127.0.0.1:8098/v1
./run.sh --imagechd 1001 --labels gt --model local:qwen2.5-vl-3b
```

Everything else (profiles, remote access, SLURM): [GATEWAY.md](GATEWAY.md).
