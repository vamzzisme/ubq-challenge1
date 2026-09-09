#!/usr/bin/env python3
"""Small Language Model (SLM) engine for Open-World Activity Reasoning (Task 4)."""

import json
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


_model = None
_tokenizer = None

def get_slm() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is None:
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading SLM: {model_id} onto {device} for Open-World Reasoning...")
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        ).to(device)
        _model.eval()
    return _model, _tokenizer


def extract_top_segments(timeline: dict[str, Any], top_n: int = 20) -> list[dict[str, Any]]:
    """Extract the longest segments to fit in the SLM context window."""
    segments = timeline.get("segments", [])
    # Sort by observed duration, descending
    sorted_segs = sorted(segments, key=lambda s: s.get("observed_duration_s", 0), reverse=True)
    return sorted_segs[:top_n]


def slm_open_world_reasoning(question: str, timeline: dict[str, Any]) -> dict[str, Any]:
    """Fallback engine for Task 4 open-world reasoning questions."""
    recording_start = timeline.get("recording_start_time_s", 0.0)
    top_segments = extract_top_segments(timeline, top_n=15)
    
    # Compress context
    lines = []
    for i, seg in enumerate(top_segments):
        start = seg.get("relative_start_s", 0)
        end = seg.get("relative_end_s", 0)
        activity = seg.get("activity", "Unknown")
        modality = seg.get("sensor_modality", "")
        channels = ", ".join(seg.get("sensor_channels", []))
        features = seg.get("evidence_features", [])
        feats_str = ", ".join(f"{f['name']}={f['value']}" for f in features[:4])
        lines.append(f"Segment: {start:.0f} to {end:.0f} seconds | Base Prediction: {activity} | Modality: {modality} ({channels}) | Signal Features: {feats_str}")
        
    context = "\n".join(lines)
    
    system_prompt = (
        "You are an expert sensor data analyst answering complex open-world queries about a user's activity. "
        "You are given a subset of the most significant timeline segments. The 'Base Prediction' is from a simple classifier, but the question asks about untrained behaviors (e.g. 'rest', 'strenuous', 'wheeled movement'). "
        "You MUST infer the answer by analyzing the 'Signal Features' (e.g. low variance for rest, cyclic/oscillating features for wheels).\n"
        "Your response MUST be EXACTLY a JSON object matching this schema:\n"
        "{\n"
        '  "Answer": "Concise direct answer",\n'
        '  "Activity/Event": "The primary activity or event inferred",\n'
        '  "Evidence": {\n'
        '    "Timestamp(s)": "<start> to <end> seconds from start",\n'
        '    "Sensor Modality": "accelerometer, gyroscope",\n'
        '    "Sensor Channel(s)": "Acc X, Acc Y, Acc Z, Gyro X, Gyro Y, Gyro Z"\n'
        "  },\n"
        '  "Explanation": "Detailed reasoning explaining how the specific signal features (e.g. variance, frequency, amplitude) justify your answer."\n'
        "}\n"
        "Do NOT output any other text or markdown, only the JSON block."
    )
    
    model, tokenizer = get_slm()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Timeline Segments:\n{context}\n\nQuestion: {question}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, 
            max_new_tokens=512, 
            temperature=0.1,
            do_sample=False
        )
        
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Parse JSON
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if not match:
        raise ValueError(f"Failed to parse JSON from SLM response:\n{response}")
        
    raw = json.loads(match.group())
    evidence = raw.get("Evidence", {})
    
    intervals = []
    ts_str = str(evidence.get("Timestamp(s)", ""))
    for range_match in re.finditer(r"(\d+(?:\.\d+)?)\s*to\s*(\d+(?:\.\d+)?)", ts_str):
        intervals.append({
            "start_s": float(range_match.group(1)),
            "end_s": float(range_match.group(2))
        })
        
    return {
        "time_base": "seconds from the first observed recording window",
        "answer": str(raw.get("Answer", "N/A")),
        "activity_event": str(raw.get("Activity/Event", "N/A")),
        "evidence": {
            "timestamps": ts_str,
            "modality": str(evidence.get("Sensor Modality", "N/A")),
            "channels": str(evidence.get("Sensor Channel(s)", "N/A")),
            "intervals": intervals
        },
        "explanation": str(raw.get("Explanation", "N/A"))
    }
