import asyncio
import sys
from typing import Any, Union
from llama_cpp import Llama

if sys.version_info >= (3, 11):
    from typing import Literal, TypedDict
else:
    from typing_extensions import Literal, TypedDict


class TextContentBlock(TypedDict):
    type: Literal["text"]
    source_lang_code: str
    target_lang_code: str
    text: str


class ImageContentBlock(TypedDict):
    type: Literal["image"]
    source_lang_code: str
    target_lang_code: str
    url: str


ContentBlock = Union[TextContentBlock, ImageContentBlock]


class TranslateGemmaClient:
    """An optimized, reusable wrapper class for TranslateGemma using llama-cpp-python

    supporting both synchronous and asynchronous text/image translation.
    """

    def __init__(
        self,
        model_path: str = "translategemma-12b-it.Q4_K_M.gguf",
        mmproj_path: str | None = "gemma-3-12b-it-mmproj-f16.gguf",
        n_gpu_layers: int = -1,
        n_ctx: int = 8192,
        **kwargs: Any,
    ) -> None:
        """Initializes the Llama model with the provided configurations."""
        self.llm = Llama(
            model_path=model_path,
            mmproj_path=mmproj_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            **kwargs,
        )

    def __enter__(self) -> "TranslateGemmaClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if hasattr(self, "llm"):
            del self.llm

    async def __aenter__(self) -> "TranslateGemmaClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)

    def _execute_translation(
        self, content_block: ContentBlock, temperature: float = 0.0, **kwargs: Any
    ) -> str:
        """Private helper to handle chat completion execution and safe response parsing."""
        messages = [{"role": "user", "content": [content_block]}]
        
        response = self.llm.create_chat_completion(
            messages=messages, temperature=temperature, **kwargs
        )
        
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Unexpected response structure from Llama model: {response}") from e

    async def _aexecute_translation(
        self, content_block: ContentBlock, temperature: float = 0.0, **kwargs: Any
    ) -> str:
        """Asynchronous wrapper for executing translation off the main thread."""
        return await asyncio.to_thread(
            self._execute_translation, content_block, temperature=temperature, **kwargs
        )

    def translate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        temperature: float = 0.0, 
        **kwargs: Any
    ) -> str:
        """Translates raw text from a source language to a target language (Synchronous)."""
        block: TextContentBlock = {
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        }
        return self._execute_translation(block, temperature=temperature, **kwargs)

    async def atranslate_text(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        temperature: float = 0.0, 
        **kwargs: Any
    ) -> str:
        """Translates raw text from a source language to a target language (Asynchronous)."""
        block: TextContentBlock = {
            "type": "text",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "text": text,
        }
        return await self._aexecute_translation(block, temperature=temperature, **kwargs)

    def translate_image(
        self,
        image_url_or_path: str,
        source_lang: str,
        target_lang: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Extracts text from an image and translates it into the target language (Synchronous)."""
        block: ImageContentBlock = {
            "type": "image",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "url": image_url_or_path,
        }
        return self._execute_translation(block, temperature=temperature, **kwargs)

    async def atranslate_image(
        self,
        image_url_or_path: str,
        source_lang: str,
        target_lang: str,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Extracts text from an image and translates it into the target language (Asynchronous)."""
        block: ImageContentBlock = {
            "type": "image",
            "source_lang_code": source_lang,
            "target_lang_code": target_lang,
            "url": image_url_or_path,
        }
        return await self._aexecute_translation(block, temperature=temperature, **kwargs)

