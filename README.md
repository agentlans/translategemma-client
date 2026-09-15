# TranslateGemma Client 🌍🤖

An optimized, production-ready Python wrapper class for **TranslateGemma** using [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python). It provides robust support for both **synchronous** and **asynchronous** text and multimodal image translation out of the box, with full cross-version compatibility (`Python >= 3.8`).

## Features

- ⚡ **Sync & Async Support**: Seamlessly switch between synchronous methods and asynchronous execution (`async`/`await`) off the main thread using `asyncio.to_thread`.
- 🖼️ **Multimodal Translation**: Translate raw text or extract and translate text from images (via URLs or local paths).
- 🔒 **Strict Type Safety**: Comprehensive type hints using `TypedDict` and conditional `typing_extensions` backports for legacy Python versions.
- 🧹 **Context Management**: Supports Python context managers (`with` and `async with`) for safe resource handling.
- 🛡️ **Defensive Parsing**: Safe response extraction with clear, informative error handling for unexpected model outputs.

## Installation

Clone the repository and install the package along with its dependencies:

```bash
# Install directly from GitHub
pip install git+https://github.com/agentlans/translategemma-client.git
```

Or install locally after cloning

```bash
pip install .
```

Or install dependencies manually via `requirements.txt`:

```bash
pip install -r requirements.txt

```

## Quickstart Guide

### 1. Synchronous Usage

```python
from translategemma_client import TranslateGemmaClient

# Initialize the client with your GGUF model and mmproj paths
with TranslateGemmaClient(
    model_path="translategemma-12b-it.Q4_K_M.gguf",
    mmproj_path="gemma-3-12b-it-mmproj-f16.gguf",
    n_gpu_layers=-1 # Offload all layers to GPU
) as client:
    
    # Translate text from English to Spanish
    translation = client.translate_text(
        text="Hello, how are you?",
        source_lang="en",
        target_lang="es"
    )
    print(f"Translation: {translation}")

    # Translate text found inside an image
    image_translation = client.translate_image(
        image_url_or_path="path/to/sign.jpg",
        source_lang="fr",
        target_lang="en"
    )
    print(f"Image Translation: {image_translation}")

```

### 2. Asynchronous Usage

```python
import asyncio
from translategemma_client import TranslateGemmaClient

async def main():
    client = TranslateGemmaClient(
        model_path="translategemma-12b-it.Q4_K_M.gguf",
        mmproj_path="gemma-3-12b-it-mmproj-f16.gguf"
    )

    # Asynchronous text translation
    result = await client.atranslate_text(
        text="Bonjour le monde!",
        source_lang="fr",
        target_lang="de"
    )
    print(f"Async Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())

```

## API Reference

### `TranslateGemmaClient`

| Method | Type | Description |
| --- | --- | --- |
| `translate_text(...)` | Synchronous | Translates raw text from source language code to target language code. |
| `atranslate_text(...)` | Asynchronous | Asynchronous counterpart for text translation. |
| `translate_image(...)` | Synchronous | Extracts text from an image (path/URL) and translates it. |
| `atranslate_image(...)` | Asynchronous | Asynchronous counterpart for image translation. |

#### Parameters for Translation Methods:

* `text` / `image_url_or_path` (`str`): The payload to translate.
* `source_lang` (`str`): ISO language code of the source (e.g., `"en"`).
* `target_lang` (`str`): ISO language code of the target (e.g., `"es"`).
* `temperature` (`float`, default `0.0`): Sampling temperature.
* `**kwargs`: Additional keyword arguments passed directly into `llama_cpp.Llama.create_chat_completion`.

## Requirements

* Python `3.8` or higher.
* `llama-cpp-python >= 0.2.90`
* `typing-extensions >= 4.0.0` (automatically installed on Python < 3.11)

## License

Distributed under the **MIT License**. See `pyproject.toml` for more details.

