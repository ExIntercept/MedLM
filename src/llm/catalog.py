"""Medical models an admin can switch the gateway to from the UI.

Same four models as the cross-model benchmark (scripts/cross_model_local.json),
served by the local Ollama. The provider says how each is called: the ones
whose Ollama chat template is reliable use /v1/chat/completions, OpenBioLLM
uses raw completions with our own Llama 3 template.
"""

MEDICAL_MODELS = [
    {"model": "hf.co/unsloth/medgemma-4b-it-GGUF:Q4_K_M", "label": "MedGemma 4B",
     "provider": "openai_chat", "note": "Google, instruction-tuned, has vision"},
    {"model": "hf.co/mradermacher/Llama3-OpenBioLLM-8B-GGUF:Q4_K_M", "label": "OpenBioLLM 8B",
     "provider": "completions", "note": "Llama 3 fine-tuned on biomedical text"},
    {"model": "hf.co/MaziyarPanahi/BioMistral-7B-GGUF:Q4_K_M", "label": "BioMistral 7B",
     "provider": "openai_chat", "note": "Mistral further trained on PubMed Central"},
    {"model": "hf.co/bartowski/JSL-MedLlama-3-8B-v2.0-GGUF:Q4_K_M", "label": "JSL-MedLlama 3 8B",
     "provider": "openai_chat", "note": "John Snow Labs medical Llama 3"},
]


def find(model: str):
    return next((m for m in MEDICAL_MODELS if m["model"] == model), None)
