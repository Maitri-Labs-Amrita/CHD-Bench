# chd-gateway

Run any vision-language model behind one OpenAI-compatible endpoint, for agent harnesses that send long multi-turn
episodes with images and tool calls. The gateway starts model workers where you say (this machine, a SLURM
cluster, or a server you already run), keeps each episode on one worker so the engine's prompt cache stays warm, and
labels every reply with what served it.

```
agent harness ──► chd-gateway (CPU) ──► launcher: local │ slurm │ attach ──► engine: vllm │ transformers │ custom
                   /v1/chat/completions
```

Hosted APIs (OpenRouter, Anthropic, OpenAI) are not proxied: call them from your harness directly. The gateway is
for models you run yourself.

## Quick start

Terminal 1, in the CHD-Bench folder (`./setup.sh` has run): the gateway, in the foreground.

```bash
uv sync --frozen --inexact --extra transformers   # the built-in engine's packages, into this folder's .venv
uv run chd-gateway check config/gateway/qwen2.5-vl-3b.transformers.local.yaml   # validates the profile only
uv run chd-gateway serve --profile config/gateway/qwen2.5-vl-3b.transformers.local.yaml --port 8098 \
  --workdir data/gateway
```

Where the engine's packages must be: this profile sets neither `engine.python` nor `launcher.worker_python`, so the
worker runs the engine with the gateway's own Python. `uv run chd-gateway` in the CHD-Bench folder uses
`CHD-Bench/.venv/bin/python`, so the `transformers` extra (torch, torchvision, transformers, pillow, accelerate) must
be in `CHD-Bench/.venv`: the first line installs it there. Check it with
`uv run --no-sync python -c "import torch, torchvision, transformers"`. Without torchvision the worker ends before it
is ready: Transformers 5.x loads Qwen2.5-VL's video processor in `AutoProcessor`, and that processor needs
torchvision. To use another environment, set `engine.python` in the profile to its Python instead.

Terminal 2, in the CHD-Bench folder, on the same host: one CHD episode with this model. The model name is
`local:` plus the profile's `name` (`qwen2.5-vl-3b`). `./run.sh` starts Slicer, runs the episode and stops Slicer.

```bash
export CHD_LOCAL_BASE_URL=http://127.0.0.1:8098/v1
CASE=1001   # a case id from: uv run python -m chd_client.task.imagechd list
./run.sh --imagechd "$CASE" --labels gt --model local:qwen2.5-vl-3b
```

This profile takes at most 16 images per request (`capabilities.max_images`), and one tool call returns up to 4
frames. The episode reads the cap from the gateway (`GET /v1/models`, below) before its first model call, and each
request then holds only the newest 16 images; older ones become a text placeholder. The cap counts images, so a turn
with several tool calls cannot pass it. `--max-images N` sets a smaller cap (a larger one is refused before the run).

8098 is the default `--port`. The gateway listens on 127.0.0.1 only: an episode on another host needs `--host` and
`--token-env` (see [Security](#security)), with the same token in the episode's `CHD_LOCAL_API_KEY`. With SLURM
workers, `--workdir` must be on a file system that the compute nodes share (see [Files](#files)). `--inexact` keeps the
packages that are already installed. `./setup.sh` and `slurm/submit_parallel.sh` also sync with `--inexact`, so they
keep the extra. A plain `uv sync` (exact) removes it: then run the first line again before you serve.

On a SLURM cluster: this profile has `launcher: {kind: local}`, so its model worker runs on the gateway's own host.
Either run both terminals inside one GPU allocation (terminal 1: `srun --partition=<gpu partition> --gres=gpu:1 ...
--pty bash`, then the gateway; terminal 2: `srun --jobid=<that job id> --overlap --pty bash`, then the episode), or use
a `launcher: {kind: slurm}` profile: the gateway then runs on a CPU host and submits a GPU job per worker
(`config/gateway/qwen2.5-vl-7b.vllm.slurm.yaml` is an example; set its `sbatch` values for your cluster).

For long episodes use the `vllm` engine: it keeps a prefix cache, so a turn does not reprocess the whole
conversation (the built-in Transformers engine does). See [The vllm engine](#the-vllm-engine). A vLLM server (or
Ollama, LM Studio, ...) that you run yourself is used with `launcher: {kind: attach, urls: [...]}`
(`config/gateway/ollama-qwen2.5vl.attach.yaml`): the gateway then neither starts nor stops it.

The client uses `http://127.0.0.1:8098/v1` as its OpenAI base URL and the profile's `name` as the model. The first
request starts a worker; it waits while the model loads.

## Profiles

One YAML file per model. Unknown keys are errors. Examples are in `config/gateway/`.

| Section | What it sets |
|---|---|
| `name`, `revision`, `upstream_model` | the model name clients send; the weights revision; the name the worker expects |
| `engine` | `vllm` (vLLM in `engine.python`'s environment, prefix cache), `transformers` (built in, any OS, no prefix cache) or `custom` (any OpenAI-compatible server command) |
| `launcher` | `local` (subprocesses, `gpus` to pin), `slurm` (`sbatch` lines, `setup` lines), `attach` (`urls` of running servers) |
| `scheduler` | `episodes_per_worker` (or `auto`: the worker reports it), `lease_s`, `request_timeout_s` |
| `interventions` | `tool_call_repair`, `forced_tool_call`: off by default (see below) |
| `cost` | `zero_api_charge`, `estimated` (with prices) or `unknown`; required for `attach` |
| `capabilities` | `images`, `tools`, `max_images`, `max_output_tokens`, `remote_images` |
| `engine.request_defaults` | request fields the gateway adds to every request to this model (the request's own keys win; one level deep for mappings; `x_gateway.request_defaults` records what was added). For settings vLLM reads only from the request, e.g. a model card's thinking budget: `request_defaults: {thinking_token_budget: 17408}` (needs `--reasoning-parser`). `model`, `messages`, `tools`, `stream`, `n` are refused |

## The vllm engine

A worker runs `<engine.python> -m vllm.entrypoints.openai.api_server --model <engine.model> --served-model-name
<name> [--revision <revision>] [--enable-auto-tool-choice --tool-call-parser <engine.tool_parser>] <engine.args>`.
Install vLLM in its own environment (its CUDA build must match the machine) and set `engine.python` to that
environment's Python. `engine.tool_parser` must match the model's tool-call format (vLLM's list:
`vllm/tool_parsers/__init__.py`). Put `--reasoning-parser`, `--max-model-len`, `--trust-remote-code` and other vLLM
flags in `engine.args`.
`uv run chd-gateway check <profile>` prints the command and the names of the environment variables.

The worker sets these variables for vLLM. A value in the profile's `engine.env` replaces each of them:

| Variable | Value | Why |
|---|---|---|
| `PATH` | the folder of `engine.python` first | as if that environment were activated: vLLM's dependencies start some tools by name (FlashInfer runs `ninja`) |
| `VLLM_USE_FLASHINFER_SAMPLER` | `0` | vLLM's sampler otherwise uses a FlashInfer kernel that FlashInfer compiles at start with `nvcc` and `ninja`; a node without a CUDA toolkit then fails (`FileNotFoundError: ... 'ninja'`). `0` selects vLLM's own sampler. |
| `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` | `$TMPDIR/chdgw-<worker id>/torchinductor`, `.../triton` (`/tmp` when `TMPDIR` is not set) | one compile cache per worker on the node's own disk. Two workers that share `/tmp/torchinductor_<user>` and compile at the same time break each other's kernels (`Cubin file saved by TritonBundler not found`). Each worker compiles its kernels once; the folders stay after the worker ends. |
| `VLLM_API_KEY` | the pool's worker key | vLLM then requires the key on `/v1` routes (not on `/health`) |

vLLM's own compile cache (`~/.cache/vllm`, or `VLLM_CACHE_ROOT`) stays shared.

### Maximum output and context

A profile's `capabilities.max_output_tokens` is the output cap per turn for that model. The gateway reports it in
`GET /v1/models`, and a `local:` run without `--max-tokens` sends it as `max_tokens` (see [Protocol](#protocol)).
Leave room for the episode prompt: `max_output_tokens` plus the longest prompt must fit in `--max-model-len`.
When it does not, vLLM lowers `max_tokens` to what is left without a message; a reply cut there ends with
`finish_reason: length` and fewer output tokens than requested, and CHD-Bench marks its transcript line
`max_tokens_lowered`.

## Protocol

Standard OpenAI chat completions (no streaming, `n` = 1). Optional request headers:

| Header | Meaning |
|---|---|
| `X-Episode-Id` | the conversation this request belongs to: requests of one episode go to the same worker |
| `X-Request-Id` | one id per turn, the same on every retry: a retry gets the reply of an attempt that succeeded or is still running (one reply per turn); after a failed or timed-out attempt it runs again |
| `X-Turn-Index` | turn number, for the request log |

Without `X-Episode-Id` requests are placed statelessly (any free worker). Every reply has:

- `usage.cost` and `usage.cost_source` (`reported`, `zero_api_charge`, `estimated`, `unknown`; `unknown` has
  `cost: null`, so a client with a budget stops);
- `x_gateway`: gateway version, profile name and sha256, weights revision, engine, worker, wait and inference times,
  attempts, and the interventions applied (with the original text).

A worker that cannot start (its engine ends before it is ready, for example because a package is missing, or it is
not ready after `startup_timeout_s`) fails the requests that wait for it at once, when no other worker is ready or
starting. The reply is HTTP 503 with error code `worker_start_failed`, the header `x-should-retry: false`, and a
message that gives the worker log path and the last line of that log. After 2 failed starts in a row the gateway
starts no worker of that profile for 300 s, and answers new requests at once with the same error. A worker that gets
ready resets the count. `GET /health` shows `start_failures`, `failing_until` and `last_start_failure` per profile.
CHD-Bench (`--model local:<name>`) does not retry this error: the episode stops with stop reason `runner_exception`,
and the summary's `exception` holds the message. Fix the cause in the log, then restart the gateway (or wait 300 s).
Other 503 replies (`no_worker`: no worker was ready within `request_timeout_s`) are still retried.

`GET /v1/models` lists one entry per profile. Besides the standard fields, each entry has
`x_gateway.capabilities`: `images`, `tools`, `max_images` and `max_output_tokens` (`null` = no cap). A `local:` model
in the harness reads them once before the episode (no model call):

- `max_images`: each request keeps at most this many images. A larger `--max-images` is refused before the run.
- `max_output_tokens` (the model creator's maximum output tokens per turn): without `--max-tokens`, the request's
  `max_tokens` is this value, and the run's `request_config` records `max_tokens_source: "gateway"`. A larger
  `--max-tokens` is refused before the run (./run.sh: before Slicer starts). A `--max-tokens` at or below it is sent
  as given.

A gateway that does not send a field, or a profile without it, leaves the requests as they were (byte-identical).

## Chat templates and tools

Many VLM chat templates do not render `tools`. Qwen2.5-VL's official template is one: with it, inference servers (vLLM,
Transformers) silently drop the tool definitions, and the model never sees them. So:

- the built-in Transformers engine checks its template at start, and refuses requests that offer tools when the
  template would drop them (HTTP 400 with the reason), instead of serving them without tools;
- `chd_client/gateway/templates/` ships fixed templates (`qwen2_5_vl_tools.jinja`: Qwen2.5-VL images + Qwen2.5 tool
  calls). Use one with `--chat-template builtin:qwen2_5_vl_tools` (Transformers engine), or
  `--chat-template {templates}/qwen2_5_vl_tools.jinja` (vLLM, in `engine.args`; a custom engine's command takes the
  same placeholder). `{templates}` is replaced with that folder.

Check a new model's template before you trust its tool calls.

## Engine limits that change a request

The built-in Transformers engine refuses what it cannot honour (`tool_choice` other than `auto`/`none`,
`response_format`, `logprobs`, `parallel_tool_calls: false`) with HTTP 400. It caps `max_tokens` at its
`--max-new-tokens` and records the cap in `x_engine.max_tokens` (a capped reply that runs out ends with
`finish_reason: length`). vLLM refuses a `max_tokens` larger than `--max-model-len`; when the prompt plus
`max_tokens` is longer than `--max-model-len` it lowers `max_tokens` to what is left, without a message (see
[Maximum output and context](#maximum-output-and-context)).

## Interventions

Small open models often write tool calls badly. The gateway can help, but help changes what a benchmark measures, so
every intervention is off by default, versioned, and recorded in each reply it changed:

- `tool_call_repair`: parse calls left in the text, repairing missing tags and miscounted braces;
- `forced_tool_call`: when tools were offered and the reply has no call, continue the reply once after the call
  prefix (needs an engine that can continue a partial reply: vLLM, the built-in engine, or one with
  `supports_prefill: true`).

A custom engine that changes replies itself should say so in `engine.declared_interventions`; the list goes into
every reply. Report results with interventions as assisted, apart from baselines.

## Files

`--workdir` (default `~/.cache/chd-gateway/<host>-<port>`): `state.json` (workers and episode leases, so a restarted
gateway takes its workers back), `requests.jsonl` (timings and counts, never message text or images),
`gateway.url`, and per profile `pools/<name>/` (frozen profile copy, worker URL files, worker logs, SLURM job
scripts). For SLURM the workdir must be on a file system that the compute nodes share.

## Security

- The gateway listens on 127.0.0.1 by default. Another host needs `--token-env` (or `--allow-no-auth`).
- Workers get a random key per pool (`CHD_GATEWAY_WORKER_KEY`; vLLM and the built-in engine check it). A custom
  engine that does not check it listens without authentication: on SLURM, other users of the cluster network can
  reach it.
- Only `data:` image URLs are accepted unless `capabilities.remote_images` is on.
- Engine logs may contain prompts. Keep the workdir private (it is created with mode 700).

## Limits

One gateway process (no replicas). No streaming. Tool-call formats: `hermes` (Qwen, Hermes families). No tenant
accounts or quotas. Requests larger than `--max-body-mb` (512 MB) are refused. A worker leaves the pool only when its
launcher confirms it ended: a stop that cannot be confirmed (e.g. `scancel` fails) is retried, and survives a gateway
restart in `state.json`. Workers of a previous run that the restarted gateway does not serve (another profile, another version of it, or
another launcher) are stopped by their own handle type. Local workers are found by a tag in their environment
(`CHD_GATEWAY_WORKER_ID`), so an engine that outlived its wrapper is still stopped (Linux); a local worker of another
host is reported as unknown, never as gone. The heartbeat (`gateway.alive`) is rewritten every tick, on its own task;
workers read its content, not its mtime, because NFS clients on other nodes may never see an mtime-only change.
