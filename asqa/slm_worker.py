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
import os
import re
import sys

# Overridable so a larger model can be measured without editing code:
#     ASQA_SLM_MODEL=Qwen/Qwen2.5-3B-Instruct python tools/parse_strategies.py
# Measured on tools/parse_strategies.py, ten deliberately unusual phrasings:
#
#     0.5B   1/10 correct, stayed inside the vocabulary  3/10
#     3B     9/10 correct, stayed inside the vocabulary 10/10
#
# The 0.5B model does not follow the closed-vocabulary instruction -- it echoes
# the questioner's word in the vocabulary's shape ("wandering", "kipping"), and
# constraining the decoding to fix that costs accuracy rather than buying it.
# The 3B model simply obeys. It costs 22.6s and 10.0 GB against 11.0s and 2.6 GB,
# paid only when the rules cannot place a question, which is rare.
MODEL_ID = os.environ.get("ASQA_SLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")

# ── Parse mode ───────────────────────────────────────────────────────────────
#
# The model's better job. Rather than asking a 0.5B model what the person did,
# ask it only what the *question* is asking for, and let the timeline answer.
# It emits an operation and class names from a closed vocabulary -- never a
# number, never a timestamp -- so a misparse produces the wrong operation
# (visible, checkable) rather than a fabricated interval.

PARSE_PROMPT = """You identify which physical activities a question is about.

Reply with ONE JSON object and nothing else:
{"operation": "...", "activities": [...], "group": "..." or null}

operation may ONLY be one of:
  identification  - what was the person doing
  verification    - did/was the person doing it, yes or no
  duration        - how long, how much time
  count           - how many times, how often
  comparison      - more time doing X or Y
  temporal        - when did it start, begin, first happen

activities may ONLY contain these exact words:
  lying_down, sitting, standing_in_place, standing_and_moving, walking, running, bicycling

group may ONLY be one of these, or null:
  resting   - the question means lying down or sitting
  active    - the question means exertion: tiring, strenuous, exercise, workout
  wheeled   - the question means a bicycle: pedalling, cycling, pushbike
  standing  - the question means standing

Use group when the question describes a kind of behaviour; use activities when it
names one directly. Map informal words and typos onto the list: jog and sprint are
running, pushbike and cycling are bicycling, kip and lie-in are lying_down.
Never invent a word that is not on the lists."""

PARSE_EXAMPLES = [
    ("is user doing anything tiring?",
     '{"operation": "verification", "activities": [], "group": "active"}'),
    ("how mcuh time did he spend lyin down",
     '{"operation": "duration", "activities": ["lying_down"], "group": null}'),
    ("when did she first start to jog",
     '{"operation": "temporal", "activities": ["running"], "group": null}'),
    ("did the bloke ever get on his pushbike",
     '{"operation": "verification", "activities": ["bicycling"], "group": null}'),
    ("how offen did he go for a wander",
     '{"operation": "count", "activities": ["walking"], "group": null}'),
]

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


def _load():
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
    return torch, model, tokenizer


def _extract_json(text: str) -> dict | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(), strict=False)
    except json.JSONDecodeError:
        return None


# ── Choosing, rather than generating ─────────────────────────────────────────
#
# Asking a 0.5B model to "reply with a word from this list" was measured staying
# inside the list 3 times in 10: it echoed the questioner's word in the shape of
# the vocabulary ("wandering", "kipping", "legging_it"), having learned the
# snake_case format but not the membership rule.
#
# So the model is never asked to *write* a class name. It is shown the classes
# as lettered options and only one token is read back -- the letter. Scoring the
# letters against each other cannot produce a word outside the list, because no
# word is generated at all. The vocabulary stops being an instruction the model
# may ignore and becomes a property of the decoding.

CHOICES = [
    ("A", "lying_down", "lying down, in bed, asleep, having a kip or a lie-in"),
    ("B", "sitting", "sitting, seated, sat down, perched"),
    ("C", "standing_in_place", "standing still in one spot"),
    ("D", "standing_and_moving", "standing while shuffling or moving about"),
    ("E", "walking", "walking, strolling, wandering, ambling, hiking, on foot"),
    ("F", "running", "running, jogging, sprinting, legging it"),
    ("G", "bicycling", "cycling, bicycling, pedalling, on a bike or pushbike"),
    ("H", "__none__", "none of these, or the question is not about one activity"),
]

CHOOSE_PROMPT = """Which activity is this question about? Reply with one letter only.

""" + "\n".join(f"{letter}. {desc}" for letter, _, desc in CHOICES)


def choose(question: str) -> dict:
    """Pick one class by scoring letters -- never by writing a word."""
    torch, model, tokenizer = _load()
    messages = [
        {"role": "system", "content": CHOOSE_PROMPT},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]

    # Only the eight letters compete; every other token is out of the running.
    scored = []
    for letter, activity, _ in CHOICES:
        ids = {tokenizer.encode(form, add_special_tokens=False)[0] for form in (letter, " " + letter)}
        scored.append((max(float(logits[i]) for i in ids), letter, activity))
    scored.sort(reverse=True)

    best, runner_up = scored[0], scored[1]
    probs = torch.softmax(torch.tensor([s for s, _, _ in scored]), dim=0)
    return {
        "activity": None if best[2] == "__none__" else best[2],
        "letter": best[1],
        "confidence": round(float(probs[0]), 3),
        "runner_up": runner_up[2],
    }


def parse(question: str, max_new_tokens: int = 120) -> dict:
    """Turn a question into a structured request. Raises if unusable."""
    torch, model, tokenizer = _load()
    messages = [{"role": "system", "content": PARSE_PROMPT}]
    for user, assistant in PARSE_EXAMPLES:
        messages += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    messages.append({"role": "user", "content": question})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.15, pad_token_id=tokenizer.eos_token_id,
        )
    completion = tokenizer.decode(generated[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    payload = _extract_json(completion)
    if payload is None:
        raise ValueError(f"parser returned no JSON object: {completion[:200]!r}")
    return payload


def generate(question: str, menu: str, max_new_tokens: int = 320) -> dict:
    torch, model, tokenizer = _load()

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
        if request.get("mode") == "choose":
            payload = choose(request["question"])
        elif request.get("mode") == "parse":
            payload = parse(request["question"], request.get("max_new_tokens", 120))
        else:
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
