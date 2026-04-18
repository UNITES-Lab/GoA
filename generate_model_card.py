"""
Generate a model card entry for endpoint.py from a HuggingFace model.

Usage:
    python generate_model_card.py \
        --model_id "Qwen/Qwen2.5-7B-Instruct" \
        --name "qwen" \
        --url "http://localhost:8000/v1/completions" \
        --domain "general"

    # Use a custom LLM endpoint to generate the summary (optional):
    python generate_model_card.py \
        --model_id "ContactDoctor/Bio-Medical-Llama-3-8B" \
        --name "biomedical_llama" \
        --url "http://localhost:8003/v1/completions" \
        --domain "biomedical" \
        --llm_model "Qwen/Qwen2.5-7B-Instruct" \
        --llm_endpoint "http://localhost:8000/v1/completions"

Output: prints a Python dict entry ready to paste into endpoint.py.
"""

import argparse
import json
import requests
from huggingface_hub import hf_hub_download, HfApi


def fetch_readme(model_id):
    """Fetch the README.md content from a HuggingFace model repo."""
    try:
        readme_path = hf_hub_download(repo_id=model_id, filename="README.md")
        with open(readme_path, "r") as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not fetch README for {model_id}: {e}")
        return None


def fetch_model_info(model_id):
    """Fetch basic model info from HuggingFace API as fallback."""
    try:
        api = HfApi()
        info = api.model_info(model_id)
        parts = []
        if info.tags:
            parts.append(f"Tags: {', '.join(info.tags)}")
        if info.pipeline_tag:
            parts.append(f"Pipeline: {info.pipeline_tag}")
        if info.library_name:
            parts.append(f"Library: {info.library_name}")
        return "\n".join(parts) if parts else None
    except Exception:
        return None


EXTRACTION_PROMPT = """You are given the README file of a language model:

{readme}

Please extract and summarize the model's key characteristics clearly and concisely in the following structured format:

1. **Domain**: The primary domain or application area the model is designed for (e.g., general-purpose, biomedical, finance, coding, math, etc.).

2. **Task Specialization**: Describe the task types the model is designed for or excels at. Be as specific as possible, including the domain context of each task (e.g., biomedical question answering, clinical decision support, financial sentiment classification, code generation). Do not include performance metrics, benchmark names, or evaluation results.

3. **Parameter Size**: The number of parameters in the model (approximate if not explicitly stated).

4. **Special Features**: Any distinguishing aspects such as fine-tuning datasets (if applicable).

Your summary will later be used to compare multiple models for selection purposes. Return your answer in bullet-point format, using the exact field names shown above. Keep it concise but specific enough for model comparison.

Answer:"""


def generate_model_card_via_llm(readme_text, llm_model, llm_endpoint):
    """Use an LLM to extract model card info from the README."""
    from transformers import AutoTokenizer

    readme_truncated = readme_text[:6000]

    prompt_text = EXTRACTION_PROMPT.format(readme=readme_truncated)

    if llm_model in ['THUDM/glm-4-9b-chat', 'internlm/internlm3-8b-instruct', 'LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct']:
        tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_model)

    msg = [{"role": "user", "content": prompt_text}]

    if tokenizer.chat_template is None:
        formatted = ""
        formatted += f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        formatted += f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
        formatted += "<|im_start|>assistant\n"
    else:
        formatted = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    res = requests.post(
        llm_endpoint,
        json={
            "model": llm_model,
            "max_tokens": 512,
            "temperature": 0.3,
            "prompt": formatted,
            "seed": 0
        },
        timeout=120
    )
    res.raise_for_status()
    output = res.json()["choices"][0]["text"].strip()
    return output


def generate_model_card_simple(model_id):
    """Generate a minimal model card without LLM (from HF metadata only)."""
    api = HfApi()
    try:
        info = api.model_info(model_id)
    except Exception as e:
        return f"Model: {model_id} (could not fetch info: {e})"

    parts = []
    parts.append(f"- **Domain**: Unknown (please fill in)")

    tags = info.tags or []
    if tags:
        parts.append(f"- **Task Specialization**: Tags: {', '.join(tags[:10])}")
    else:
        parts.append(f"- **Task Specialization**: Unknown (please fill in)")

    safetensors = info.safetensors
    if safetensors and hasattr(safetensors, 'total'):
        param_b = safetensors.total / 1e9
        parts.append(f"- **Parameter Size**: {param_b:.2f}B")
    else:
        parts.append(f"- **Parameter Size**: Unknown")

    parts.append(f"- **Special Features**: (please fill in)")

    return "\n".join(parts)


def format_endpoint_entry(name, model_id, url, domain, model_card, max_tokens=4096):
    """Format a Python dict entry for endpoint.py."""
    model_card_escaped = model_card.replace("'", "\\'")
    model_card_oneline = model_card_escaped.replace("\n", "\\n")

    entry = f'''    "{name}": {{
        "url": "{url}",
        "model_id": "{model_id}",
        "max_tokens": {max_tokens},
        "domain": "{domain}",
        "model_card": '{model_card_oneline}'
    }}'''
    return entry


def main():
    parser = argparse.ArgumentParser(description="Generate a model card entry for endpoint.py")
    parser.add_argument("--model_id", required=True, help="HuggingFace model ID (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--name", required=True, help="Short name for endpoint.py key (e.g., 'qwen')")
    parser.add_argument("--url", required=True, help="vLLM endpoint URL (e.g., http://localhost:8000/v1/completions)")
    parser.add_argument("--domain", required=True, help="Domain label (e.g., general, code, math, biomedical, finance, legal)")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for the model")
    parser.add_argument("--llm_model", default=None, help="LLM model ID to use for extraction (optional, uses HF metadata if not set)")
    parser.add_argument("--llm_endpoint", default=None, help="LLM endpoint URL for extraction")
    args = parser.parse_args()

    print(f"Fetching README for {args.model_id}...")
    readme = fetch_readme(args.model_id)

    if readme and args.llm_model and args.llm_endpoint:
        print(f"Generating model card via LLM ({args.llm_model})...")
        model_card = generate_model_card_via_llm(readme, args.llm_model, args.llm_endpoint)
    elif readme:
        print("No LLM endpoint specified. Generating from HF metadata only.")
        print("Tip: pass --llm_model and --llm_endpoint for better results.")
        model_card = generate_model_card_simple(args.model_id)
    else:
        print("Could not fetch README. Generating from HF metadata only.")
        model_card = generate_model_card_simple(args.model_id)

    entry = format_endpoint_entry(args.name, args.model_id, args.url, args.domain, model_card, args.max_tokens)

    print("\n" + "=" * 70)
    print("Add the following entry to endpoint.py (inside model_endpoint_dict):")
    print("=" * 70 + "\n")
    print(entry)
    print("\n" + "=" * 70)

    print(f"\nGenerated model card:\n{model_card}")


if __name__ == "__main__":
    main()
