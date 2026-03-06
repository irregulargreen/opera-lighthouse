"""
ollama_setup.py — helper script to check what Ollama model is best for your hardware
and pull the optimal one.

Run: python ollama_setup.py

Tier 1 screener needs:
  - Fast: <3s response target
  - Reasonable instruction following
  - Good enough to detect obvious mispricings

Model recommendations by VRAM:
  4GB VRAM:  phi3:mini (3.8B) — fastest, decent quality
  8GB VRAM:  llama3.1:8b    — good balance (DEFAULT)
  16GB VRAM: mistral:7b or llama3.1:8b (can run quantised 13B)
  24GB VRAM: llama3.1:latest (70B Q4) — near frontier quality, free
  CPU only:  phi3:mini — only viable option under 60s response
"""
from __future__ import annotations

import subprocess
import sys

import httpx


RECOMMENDED_MODELS = [
    # (name, vram_gb, quality, speed, description)
    ("phi3:mini",       2,  "good",      "very fast",  "3.8B — best for CPU or 4GB VRAM"),
    ("llama3.1:8b",     6,  "good",      "fast",       "8B — default, best balance"),
    ("mistral:7b",      6,  "good",      "fast",       "7B — good for reasoning"),
    ("llama3.1:70b",    40, "very good", "slow",       "70B Q4 — near GPT-4o quality, free"),
    ("gemma2:9b",       7,  "good",      "fast",       "9B — strong instruction following"),
]


def check_ollama() -> bool:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_installed_models() -> list[str]:
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


def benchmark_model(model: str) -> float:
    """Quick latency test on a simple prediction market prompt."""
    import time
    prompt = '{"probability": 0.65, "confidence": 0.7}'
    try:
        start = time.time()
        resp = httpx.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Respond only with JSON: {\"probability\": 0.XX, \"confidence\": 0.XX}"},
                    {"role": "user", "content": "Will it rain tomorrow? Current price: 40%"}
                ],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 50},
            },
            timeout=60,
        )
        elapsed = time.time() - start
        content = resp.json().get("message", {}).get("content", "")
        # Check it returned parseable JSON
        import json, re
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            json.loads(match.group())
            return elapsed
        return 999.0  # Failed to return valid JSON
    except Exception as e:
        print(f"  Error: {e}")
        return 999.0


def main():
    print("=" * 60)
    print("Polymarket Bot — Ollama Setup Assistant")
    print("=" * 60)

    if not check_ollama():
        print("\n❌ Ollama is NOT running.")
        print("   Install: https://ollama.ai")
        print("   Start:   ollama serve")
        sys.exit(1)

    print("\n✅ Ollama is running\n")
    installed = get_installed_models()

    if installed:
        print(f"Installed models: {', '.join(installed)}\n")
    else:
        print("No models installed yet.\n")

    # Model recommendations
    print("Recommended models for Tier 1 screening:\n")
    print(f"  {'Model':<20} {'VRAM':<8} {'Quality':<12} {'Speed':<12} {'Notes'}")
    print("  " + "-" * 65)
    for name, vram, quality, speed, desc in RECOMMENDED_MODELS:
        marker = "✅ installed" if any(name in m for m in installed) else ""
        print(f"  {name:<20} {str(vram)+'GB':<8} {quality:<12} {speed:<12} {desc} {marker}")

    print()

    # Benchmark installed models
    if installed:
        print("Benchmarking installed models (latency on a test prompt)...")
        results = []
        for model in installed[:5]:  # Limit to first 5
            print(f"  Testing {model}...", end=" ", flush=True)
            latency = benchmark_model(model)
            if latency < 999:
                print(f"{latency:.1f}s")
                results.append((model, latency))
            else:
                print("failed (didn't return valid JSON)")

        if results:
            best = sorted(results, key=lambda x: x[1])[0]
            print(f"\n  🏆 Best for Tier 1 screening: {best[0]} ({best[1]:.1f}s)")
            print(f"     Add to .env: OLLAMA_MODEL={best[0]}")

    # Suggest pull command
    if not installed or not any("llama3.1:8b" in m or "phi3" in m for m in installed):
        print("\nRecommended: pull llama3.1:8b (default) or phi3:mini (fastest)")
        print("  ollama pull llama3.1:8b")
        print("  ollama pull phi3:mini")

    print("\nFor 24GB VRAM (near-GPT-4o quality, completely free):")
    print("  ollama pull llama3.1:70b")
    print("  Then set OLLAMA_MODEL=llama3.1:70b in .env")
    print()


if __name__ == "__main__":
    main()