from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - 依赖缺失时插件仍可加载
    OpenAI = None


@dataclass
class LLMImageResult:
    text: str
    images: list[bytes]


class LLMService:
    def __init__(self, config: dict[str, Any] | Any):
        self._client = None
        self.reload_config(config)

    def reload_config(self, config: dict[str, Any] | Any) -> None:
        self.api_key = str(config.get("api_key", "")).strip()
        self.api_base = str(config.get("api_base", "")).strip()
        self.prefix = str(config.get("prefix", "openai")).strip()
        self.model = str(config.get("model", "gemini-3.1-flash-image-preview")).strip()

    def _build_model_name(self, model: str | None = None) -> str:
        target = (model or self.model).strip()
        if not target:
            raise ValueError("模型名为空，请在插件配置中设置 model。")
        if "/" in target:
            return target
        if self.prefix:
            return f"{self.prefix}/{target}"
        return target

    def _ensure_client(self):
        if OpenAI is None:
            raise RuntimeError("缺少 openai 依赖，请先安装 openai。")
        if not self.api_key:
            raise ValueError("未配置 api_key，请在插件配置中填写。")
        if self._client is None:
            client_kwargs = {"api_key": self.api_key}
            if self.api_base:
                client_kwargs["base_url"] = self.api_base
            self._client = OpenAI(**client_kwargs)
        return self._client

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        model: str | None = None,
    ) -> LLMImageResult:
        return self._chat_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            model=model,
            base64_image=None,
        )

    def edit_image(
        self,
        prompt: str,
        base64_image: str,
        aspect_ratio: str = "1:1",
        model: str | None = None,
    ) -> LLMImageResult:
        if not base64_image:
            raise ValueError("编辑图片时缺少输入图片。")
        return self._chat_image(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            model=model,
            base64_image=base64_image,
        )

    def _chat_image(
        self,
        prompt: str,
        aspect_ratio: str,
        model: str | None,
        base64_image: str | None,
    ) -> LLMImageResult:
        if not prompt.strip():
            raise ValueError("prompt 不能为空。")

        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        if base64_image:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                }
            )

        client = self._ensure_client()
        response = client.chat.completions.create(
            model=self._build_model_name(model),
            messages=[
                {"role": "system", "content": f"aspect_ratio={aspect_ratio}"},
                {"role": "user", "content": user_content},
            ],
            modalities=["text", "image"],
        )
        return self._parse_response(response)

    def _parse_response(self, response: Any) -> LLMImageResult:
        parts = self._extract_parts(response)
        texts: list[str] = []
        images: list[bytes] = []

        for part in parts:
            item = part.model_dump() if hasattr(part, "model_dump") else part
            if not isinstance(item, dict):
                continue

            text = item.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text.strip())

            inline_data = item.get("inline_data")
            if hasattr(inline_data, "model_dump"):
                inline_data = inline_data.model_dump()
            if isinstance(inline_data, dict):
                encoded = inline_data.get("data") or inline_data.get("b64_json")
                if isinstance(encoded, str) and encoded:
                    try:
                        images.append(base64.b64decode(encoded))
                    except Exception:
                        continue

        if not images:
            raise RuntimeError("未收到有效图片数据，请稍后重试或更换模型。")

        return LLMImageResult(text="\n".join(texts), images=images)

    @staticmethod
    def _extract_parts(response: Any) -> list[Any]:
        try:
            message = response.choices[0].message
        except Exception:
            return []

        parts = getattr(message, "multi_mod_content", None)
        if isinstance(parts, list) and parts:
            return parts

        content = getattr(message, "content", None)
        if isinstance(content, list):
            return content
        if isinstance(content, str) and content.strip():
            return [{"text": content}]
        return []
