"""
list_models.py
Lists all locally available Ollama models.
"""

import ollama

def list_available_models():
    try:
        response = ollama.list()
        # Modern ollama-python returns ListResponse object with .models attribute
        if hasattr(response, "models"):
            models_list = response.models
        elif isinstance(response, dict):
            models_list = response.get("models", [])
        else:
            models_list = []

        print("Available Ollama models:\n")
        if not models_list:
            print("  No models found. Run: ollama pull llama3.2")
            return

        for model in models_list:
            if hasattr(model, "model"):
                name = model.model or getattr(model, "name", "Unknown")
                size = getattr(model, "size", 0) or 0
            elif isinstance(model, dict):
                name = model.get("name") or model.get("model", "Unknown")
                size = model.get("size", 0)
            else:
                name = str(model)
                size = 0

            size_gb = round(size / (1024 ** 3), 2) if size else "N/A"
            print(f"  • {name}  ({size_gb} GB)")

    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("Make sure Ollama is running (`ollama serve`)")

if __name__ == "__main__":
    list_available_models()