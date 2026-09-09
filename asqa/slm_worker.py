#!/usr/bin/env python3
"""Isolated worker process for the small language model.

Run as ``python -m asqa.slm_worker``: reads one JSON request on stdin and writes
one JSON response on stdout.

This exists for a concrete reason.  Loading PyTorch into the same interpreter
that has already run XGBoost segfaults on this platform (exit 139), on both MPS
and CPU, and ``KMP_DUPLICATE_LIB_OK`` does not help -- the boosted-tree runtime
and the tensor runtime bring incompatible native threading libraries into one
address space.  Rather than give up either component, the language model runs in
its own interpreter.

The isolation pays for itself twice over: the challenge asks for per-component
resource cost, and a separate process makes the model's memory and latency
directly measurable instead of tangled with the classifier's.

Request:   {"question": str, "menu": str, "max_new_tokens": int}
Response:  {"ok": bool, "payload": {...}} or {"ok": false, "error": str}
"""

from __future__ import annotations

import json
import re
import sys

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = """You are a sensor analyst. You are given numbered intervals from a wearable recording, each with measured accelerometer and gyroscope statistics.

Answer the user's question using ONLY these intervals. You must not invent times.

Reply with a JSON object and nothing else:
{"answer": "<short direct answer>", "event": "<the behaviour in plain words>", "intervals": [<interval numbers you are citing>], "explanation": "<why those measurements support your answer>"}

Guidance for reading the measurements:
- energy near 0.00g means the body is still; above 0.1g means active motion
- cadence 1.5-2.2Hz with high energy is typical of walking; 2.5-3.5Hz of running
- steady moderate energy with low impact and sustained rotation suggests cycling
- tilt describes the device orientation, which changes between lying and upright"""

# A 0.5B model needs to be shown the shape of the answer, not just told it.
# Without this exemplar it reliably emits a JSON *array* and then degenerates
# into repeating one phrase.
EXAMPLE_USER = """Intervals:
[1] 0-600s (600s, classified sitting) energy=0.0100g cadence=0.8Hz tilt=30deg rotation=0.020rad/s
[2] 660-900s (240s, classified walking) energy=0.3500g cadence=1.8Hz tilt=70deg rotation=1.000rad/s

Question: Was the user physically active at any point?"""

EXAMPLE_ASSISTANT = (
    '{"answer": "Yes", "event": "Sustained walking", "intervals": [2], '
    '"explanation": "Interval 2 shows body-acceleration energy of 0.35g at a steady 1.8Hz '
    'cadence with 1.0rad/s of rotation, the signature of continuous gait, whereas interval 1 '
    'is near-motionless at 0.01g."}'
)

RETRY_SUFFIX = (
    "\n\nReply with ONE JSON object only. It must start with { and end with }. "
    "Do not return a list. Do not repeat yourself."
)


def generate(question: str, menu: str, max_new_tokens: int = 320) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    else:
        device, dtype = "cpu", torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype).to(device).eval()

    def run(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": EXAMPLE_USER},
            {"role": "assistant", "content": EXAMPLE_ASSISTANT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy, so an answer is reproducible
                repetition_penalty=1.15,  # this model otherwise loops on one phrase
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(generated[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

    prompt = f"Intervals:\n{menu}\n\nQuestion: {question}"
    last = ""
    for attempt in range(2):
        completion = run(prompt if attempt == 0 else prompt + RETRY_SUFFIX)
        last = completion
        match = re.search(r"\{.*\}", completion, re.DOTALL)
        if match:
            try:
                # strict=False tolerates literal newlines inside strings, which a
                # small model emits routinely in a multi-line explanation.
                return json.loads(match.group(), strict=False)
            except json.JSONDecodeError:
                continue
    raise ValueError(f"model returned no usable JSON object: {last[:200]!r}")


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        payload = generate(
            request["question"], request["menu"], request.get("max_new_tokens", 320)
        )
        sys.stdout.write(json.dumps({"ok": True, "payload": payload}))
    except Exception as exc:  # noqa: BLE001 - reported to the parent, never raised
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
