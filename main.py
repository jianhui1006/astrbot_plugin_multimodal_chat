import asyncio
import os
import re
import tempfile
import traceback

from astrbot.api import AstrBotConfig
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register

try:
    from .src.llm.service import LLMService
except Exception:
    from src.llm.service import LLMService


@register("multimodal_chat", "DLEE", "多模态生图与编辑插件", "1.1.0")
class MultimodalChat(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.llm_service = LLMService(config)

    @staticmethod
    def _extract_prompt(message: str, command_name: str) -> str:
        text = (message or "").strip()
        text = re.sub(rf"^/?{re.escape(command_name)}(?:\s+|$)", "", text, count=1)
        return text.strip()

    @staticmethod
    def _extract_aspect_ratio(prompt: str) -> tuple[str, str]:
        text = (prompt or "").strip()
        match = re.search(
            r"(?:^|\s)(?:--ar|--aspect|--aspect-ratio)\s+([0-9]{1,2}:[0-9]{1,2})(?=\s|$)",
            text,
        )
        if not match:
            return text, "1:1"

        aspect_ratio = match.group(1)
        clean_prompt = f"{text[:match.start()]} {text[match.end():]}".strip()
        clean_prompt = re.sub(r"\s+", " ", clean_prompt)
        return clean_prompt, aspect_ratio

    @staticmethod
    def _save_temp_image(image_bytes: bytes, file_prefix: str) -> str:
        fd, path = tempfile.mkstemp(prefix=file_prefix, suffix=".png")
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        return path

    @staticmethod
    def _find_first_image(event: AstrMessageEvent) -> Image | None:
        for comp in event.get_messages():
            if isinstance(comp, Image):
                return comp
        return None

    async def initialize(self):
        logger.info("astrbot_plugin_multimodal_chat 初始化完成")

    @filter.command("edit")
    async def edit_image(self, event: AstrMessageEvent):
        """编辑用户发送的图片并返回结果。"""
        prompt = self._extract_prompt(event.message_str, "edit")
        prompt, aspect_ratio = self._extract_aspect_ratio(prompt)
        if not prompt:
            yield event.plain_result("用法：/edit 提示词 [--ar 16:9]（并附带一张图片）")
            return

        image_comp = self._find_first_image(event)
        if image_comp is None:
            yield event.plain_result("请在 /edit 指令消息中附带一张图片。")
            return

        try:
            input_image_base64 = await image_comp.convert_to_base64()
            result = await asyncio.to_thread(
                self.llm_service.edit_image,
                prompt,
                input_image_base64,
                aspect_ratio,
            )
        except Exception as e:
            logger.error(f"编辑图片失败: {e}")
            logger.error(traceback.format_exc())
            yield event.plain_result(f"编辑失败：{e}")
            return

        if result.text:
            yield event.plain_result(result.text)

        for idx, image_bytes in enumerate(result.images, start=1):
            try:
                image_path = await asyncio.to_thread(
                    self._save_temp_image, image_bytes, f"astrbot_edit_{idx}_"
                )
            except Exception as e:
                logger.error(f"保存编辑结果图片失败: {e}")
                logger.error(traceback.format_exc())
                yield event.plain_result(f"编辑成功，但保存图片失败：{e}")
                continue
            yield event.image_result(image_path)

    @filter.command("gen")
    async def generate_image(self, event: AstrMessageEvent):
        """根据提示词生成图片并返回结果。"""
        prompt = self._extract_prompt(event.message_str, "gen")
        prompt, aspect_ratio = self._extract_aspect_ratio(prompt)
        if not prompt:
            yield event.plain_result("用法：/gen 提示词 [--ar 16:9]")
            return

        try:
            result = await asyncio.to_thread(
                self.llm_service.generate_image,
                prompt,
                aspect_ratio,
            )
        except Exception as e:
            logger.error(f"生成图片失败: {e}")
            logger.error(traceback.format_exc())
            yield event.plain_result(f"生图失败：{e}")
            return

        if result.text:
            yield event.plain_result(result.text)

        for idx, image_bytes in enumerate(result.images, start=1):
            try:
                image_path = await asyncio.to_thread(
                    self._save_temp_image, image_bytes, f"astrbot_gen_{idx}_"
                )
            except Exception as e:
                logger.error(f"保存生成图片失败: {e}")
                logger.error(traceback.format_exc())
                yield event.plain_result(f"生图成功，但保存图片失败：{e}")
                continue
            yield event.image_result(image_path)

    async def terminate(self):
        logger.info("astrbot_plugin_multimodal_chat 已卸载")
