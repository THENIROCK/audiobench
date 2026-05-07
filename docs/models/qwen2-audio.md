# Running `qwen2-audio-7b`

`Qwen/Qwen2-Audio-7B-Instruct` is ~14 GB in fp16 and wants a GPU with at least 16 GB VRAM. Most laptops cannot run it locally; an Apple M3 with 16 GB unified memory technically can, but each probe takes minutes once the system starts swapping. The adapter has two backends:

- **local** — HuggingFace `transformers`. Picks device + dtype automatically (CUDA fp16 → MPS fp16 → CPU fp32). See [Local setup](#local-setup) below.
- **api** — POSTs each probe to `AUDIOBENCH_QWEN_ENDPOINT` over multipart form data. The local `transformers` import never happens. This is the path to use if you don't have a GPU.

If you set `AUDIOBENCH_QWEN_ENDPOINT`, the adapter takes the API path automatically — there is no separate model name.

## Endpoint contract

The adapter source of truth is `_answer_api` in [`src/audiobench/models/qwen2_audio.py`](https://github.com/THENIROCK/audiobench/blob/main/src/audiobench/models/qwen2_audio.py). Anything you stand up at `AUDIOBENCH_QWEN_ENDPOINT` must speak this contract:

**Request** — `POST {AUDIOBENCH_QWEN_ENDPOINT}` with `Content-Type: multipart/form-data` and exactly two fields:

| field | type | content |
|---|---|---|
| `prompt` | text | The full prompt string. The adapter prepends `"Listen to the audio and answer with only the word yes or no. "` to each rendered probe (e.g. `"Do you hear a siren?"`) and sends the whole thing as one string. |
| `audio` | file | `mix.wav`, `Content-Type: audio/wav`. A WAV blob written by `soundfile.write(..., format="WAV")` from a float32 numpy array at 16 kHz for `ab/sound-id`. |

**Response** — plain text body. The adapter `.strip()`s it and feeds it to `audiobench.probes.parse_yes_no`, which is permissive: any of `yes / y / yeah / yep / true / present / positive / 1` at the start counts as yes; the matching no-list counts as no. A short `yes` or `no` is the cleanest thing to return.

**Timeout** — 120 s per request, hardcoded in the adapter. If your server is slower than that, requests will fail. Pre-warm the endpoint before kicking off a benchmark run.

**Calling pattern** — one HTTP request per probe, no batching. Demo-fast issues ~120 sequential requests, so put the endpoint on a low-RTT network and keep the model warm.

## Easiest setup: Modal

[Modal](https://modal.com) is the lowest-friction path if you don't own a GPU. You write one Python file, run `modal deploy`, and get a stable HTTPS URL backed by an A10G. The free tier covers a few full benchmark runs ($30/month in credits; an A10G is roughly $0.04/min and the demo-fast profile finishes in ~5 min once warm).

### 1. Sign up + install Modal

```bash
pip install modal
modal token new
```

`modal token new` opens a browser for sign-up/login.

Modal asks for a payment method; the monthly credits make this free for benchmarking purposes.

### 2. Drop this file somewhere outside iCloud Drive

Save as `qwen_modal.py`. Anywhere works as long as it's not under `~/Documents` or another iCloud-sync'd location (otherwise iCloud may evict files mid-run, see [iCloud caveat](#icloud-caveat) below).

```python
# qwen_modal.py
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "transformers>=4.45",
        "soundfile",
        "librosa",
        "numpy",
        "fastapi[standard]",
    )
)

app = modal.App("audiobench-qwen2", image=image)


@app.cls(gpu="A10G", scaledown_window=300, image=image)
class QwenServer:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.model = (
            Qwen2AudioForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-Audio-7B-Instruct", torch_dtype=torch.float16
            )
            .to("cuda")
            .eval()
        )

    @modal.asgi_app()
    def web(self):
        import io
        import numpy as np
        import soundfile as sf
        from fastapi import FastAPI, File, Form, UploadFile
        from fastapi.responses import PlainTextResponse

        api = FastAPI()

        @api.post("/", response_class=PlainTextResponse)
        async def infer(prompt: str = Form(...), audio: UploadFile = File(...)):
            raw = await audio.read()
            wav, sr = sf.read(io.BytesIO(raw))
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            wav = wav.astype(np.float32)

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "audio.wav"},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            inputs = self.processor(
                text=text, audio=wav, sampling_rate=sr,
                return_tensors="pt", padding=True,
            )
            inputs = {
                k: v.to("cuda") if hasattr(v, "to") else v
                for k, v in inputs.items()
            }
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=8)
            prompt_len = inputs["input_ids"].shape[1]
            return self.processor.batch_decode(
                out[:, prompt_len:], skip_special_tokens=True
            )[0].strip()

        return api
```

### 3. Deploy

```bash
modal deploy qwen_modal.py
```

You'll see a URL like:

```
https://<your-username>--audiobench-qwen2-qwenserver-web.modal.run
```

That's the endpoint. It stays up across runs.

### 4. Pre-warm + smoke-test the contract

The first request triggers a cold start (~60–120 s while Modal pulls the model from HuggingFace into the container). Hit the endpoint once with `curl` before pointing audiobench at it so the first audiobench probe doesn't blow the 120 s adapter timeout:

```bash
ENDPOINT="https://<your-username>--audiobench-qwen2-qwenserver-web.modal.run"
```

Create a 1-second silent WAV:

```bash
python -c "
import soundfile as sf, numpy as np
sf.write('/tmp/silence.wav', np.zeros(16000, dtype=np.float32), 16000)
"
```

Send one request to warm the endpoint and verify the contract:

```bash
curl -sS -X POST "$ENDPOINT" \
  -F 'prompt=Listen to the audio and answer with only the word yes or no. Do you hear a siren?' \
  -F 'audio=@/tmp/silence.wav;type=audio/wav'
```

Once warm, calls usually take a couple of seconds. The exact answer can vary (`yes` or `no`), and either is fine for a contract smoke test.

### 5. Run the benchmark

```bash
export AUDIOBENCH_QWEN_ENDPOINT="$ENDPOINT"
```

Start with a tiny run first (one solo mixture, 3 probes):

```bash
audiobench run ab/sound-id --model qwen2-audio-7b \
  --pack demo --conditions solo --limit 1 \
  --output results/qwen2-modal-smoke.json --pretty-json
```

Then run the full `demo-fast` profile (~30 mixtures, ~120 probes):

```bash
audiobench run ab/sound-id --model qwen2-audio-7b --profile demo-fast \
  --output results/qwen2-modal-demo-fast.json
```

The container stays warm for 300 s after the last request (`scaledown_window=300`); when the benchmark sits idle it auto-stops and you stop paying. Bring it down explicitly when you're done:

```bash
modal app stop audiobench-qwen2
```

## Alternative: Google Colab + Cloudflare Tunnel

Free, no credit card, runs in a browser tab. The free tier gives you a T4 with 15 GB VRAM — just enough for Qwen2-Audio-7B in fp16. The trade-off is that the session disconnects if you close the tab or sit idle for ~90 minutes, so you have to start your benchmark within a few minutes of starting the notebook.

In a fresh Colab notebook with a GPU runtime selected (Runtime → Change runtime type → T4 GPU):

Install dependencies:

```python
!pip install -q transformers soundfile librosa fastapi uvicorn python-multipart
```

Load the model and start the FastAPI server:

```python
import io, threading, numpy as np, soundfile as sf, torch
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import PlainTextResponse
import uvicorn, nest_asyncio
nest_asyncio.apply()

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", torch_dtype=torch.float16
).to("cuda").eval()

app = FastAPI()

@app.post("/", response_class=PlainTextResponse)
async def infer(prompt: str = Form(...), audio: UploadFile = File(...)):
    raw = await audio.read()
    wav, sr = sf.read(io.BytesIO(raw))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    conversation = [{"role": "user", "content": [
        {"type": "audio", "audio_url": "audio.wav"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=wav, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8)
    prompt_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)[0].strip()

threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning"),
    daemon=True,
).start()
```

Start a free public HTTPS tunnel:

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8000 2>&1 | head -20
```

Take the `https://*.trycloudflare.com` URL Cloudflared prints and use it as `AUDIOBENCH_QWEN_ENDPOINT` in your local terminal.

## Local setup

Loading the 7B model needs `torch` and a GPU (CUDA or Apple MPS). On the laptop side:

```bash
pip install -e .
pip install transformers torch soundfile
audiobench run ab/sound-id --model qwen2-audio-7b --pack demo --conditions solo --limit 1
```

The adapter picks device + dtype automatically:

| host | device | dtype |
|---|---|---|
| CUDA available | `cuda` | `float16` |
| Apple MPS available | `mps` | `float16` |
| Otherwise | `cpu` | `float32` |

Override with env vars when needed:

```bash
AUDIOBENCH_QWEN_DEVICE=cpu AUDIOBENCH_QWEN_DTYPE=float32 \
  audiobench run ab/sound-id --model qwen2-audio-7b ...
```

`AUDIOBENCH_QWEN_DEVICE` accepts `cuda`, `mps`, `cpu`. `AUDIOBENCH_QWEN_DTYPE` accepts `float16` / `bfloat16` / `float32` (and the obvious aliases).

### Apple Silicon notes

It runs, but it's slow. On an M3 with 16 GB unified memory, fp16 weights barely fit alongside the OS and the IDE. Expect:

- model load: 5–10 minutes from a warm HuggingFace cache
- per-probe inference: 30–150 seconds depending on memory pressure
- a full `--profile demo-fast` (~120 probes): hours, not minutes

If you go this route, quit Cursor / Chrome / anything else memory-hungry first, and start with `--limit 1 --conditions solo` to get a feel for the timing before committing to a longer run.

### iCloud caveat

If your project lives under `~/Documents` (or any iCloud Drive folder), macOS's File Provider will evict `.venv` files under memory pressure (`compressed,dataless` xattr). Fresh Python processes hang on `read()` of evicted `.pyc` files while iCloud tries to fetch them back, which looks like a 0% CPU hang at startup.

The fix is to move the project (or at least `.venv` and the HuggingFace cache) out of iCloud:

```bash
mkdir -p ~/code
cp -R ~/Documents/audiobench ~/code/
cd ~/code/audiobench
rm -rf .venv && python3 -m venv .venv
source .venv/bin/activate
pip install -e . transformers torch soundfile
```

You can verify the venv is fully materialized with:

```bash
find .venv -type f -flags +dataless | wc -l   # should print 0
```

You want this command to print `0`.

## Picking between options

| You have… | Use |
|---|---|
| A CUDA GPU box | local mode |
| No GPU but want results soon | Modal (~10 min setup, $30/mo free credits) |
| No GPU and don't want to sign up for anything | Colab + Cloudflared (free, but session-bound) |
| Apple Silicon and patience | local mode with the iCloud workaround above |
| Just want to verify the adapter wires up correctly | a fake server that always replies `yes` (see below) |

### Wiring smoke test (no GPU required)

You can confirm the API path runs end-to-end without involving Qwen2-Audio at all:

```bash
python -c "
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        self.send_response(200); self.send_header('Content-Type','text/plain'); self.end_headers()
        self.wfile.write(b'yes')
HTTPServer(('127.0.0.1', 8765), H).serve_forever()
" &
export AUDIOBENCH_QWEN_ENDPOINT="http://127.0.0.1:8765"
audiobench run ab/sound-id --model qwen2-audio-7b --pack demo --conditions solo --limit 1 \
  --output /tmp/yes-bot.json
```

Every probe gets `yes`, so recall is perfect and FPR is 1.0. Useless as a benchmark, but it proves the multipart contract, the adapter, and the suite runner all line up before you spend money on Modal.
