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