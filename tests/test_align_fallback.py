"""Alignment must not silently vanish when whisper misdetects the language.

Real failure: a code-switched meeting was detected as 'jw' (Javanese), which
has no wav2vec2 align model; load_align_model raised, the pipeline swallowed
it, and the whole file lost word-level timing — downstream, the rename UI
degraded to 30s multi-speaker text blobs whose audio didn't match the quote.
"""
import pytest

import src.transcribe as t


def test_unknown_language_falls_back_to_english(monkeypatch):
    calls = []

    def fake_load(language_code, device):
        calls.append(language_code)
        if language_code != "en":
            raise ValueError(f"No default align-model for language: {language_code}")
        return "MODEL", {"language": "en"}

    monkeypatch.setattr(t.whisperx, "load_align_model", fake_load)
    model, meta = t._load_align_model_with_fallback("jw", "cuda")
    assert calls == ["jw", "en"]
    assert model == "MODEL"


def test_known_language_loads_directly(monkeypatch):
    calls = []

    def fake_load(language_code, device):
        calls.append(language_code)
        return "MODEL", {"language": language_code}

    monkeypatch.setattr(t.whisperx, "load_align_model", fake_load)
    model, meta = t._load_align_model_with_fallback("hi", "cuda")
    assert calls == ["hi"]


def test_english_failure_is_not_retried(monkeypatch):
    def fake_load(language_code, device):
        raise RuntimeError("download failed")

    monkeypatch.setattr(t.whisperx, "load_align_model", fake_load)
    with pytest.raises(RuntimeError):
        t._load_align_model_with_fallback("en", "cuda")
