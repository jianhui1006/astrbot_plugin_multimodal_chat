import asyncio
import os
import re
import tempfile
import time
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

    @staticmethod
    def _event_trace(event: AstrMessageEvent) -> str:
        return (
            f"user_id={event.get_sender_id()} "
            f"user_name={event.get_sender_name()} "
            f"session_id={event.get_session_id()} "
            f"group_id={event.get_group_id()} "
            f"platform={event.get_platform_name()}"
        )

    async def initialize(self):
        logger.info("astrbot_plugin_multimodal_chat 初始化完成")

    @filter.command("edit")
    async def edit_image(self, event: AstrMessageEvent):
        """编辑用户发送的图片并返回结果。"""
        trace = self._event_trace(event)
        started_at = time.perf_counter()
        logger.info(f"[/edit] 收到请求 | {trace} | raw_input={event.message_str!r}")

        logger.info(f"[/edit] 开始解析指令参数 | {trace}")
        prompt = self._extract_prompt(event.message_str, "edit")
        prompt, aspect_ratio = self._extract_aspect_ratio(prompt)
        logger.info(
            f"[/edit] 参数解析完成 | {trace} | prompt_len={len(prompt)} | aspect_ratio={aspect_ratio}"
        )

        if not prompt:
            logger.warning(f"[/edit] 缺少 prompt | {trace}")
            yield event.plain_result("用法：/edit 提示词 [--ar 16:9]（并附带一张图片）")
            return

        image_comp = self._find_first_image(event)
        if image_comp is None:
            logger.warning(f"[/edit] 未检测到图片消息 | {trace}")
            yield event.plain_result("请在 /edit 指令消息中附带一张图片。")
            return

        try:
            logger.info(f"[/edit] 开始处理输入图片 | {trace}")
            input_image_base64 = await image_comp.convert_to_base64()
            logger.info(
                f"[/edit] 输入图片准备完成 | {trace} | image_b64_len={len(input_image_base64)} | model={self.llm_service.model}"
            )
            logger.info(f"[/edit] 开始调用图像编辑模型 | {trace}")
            result = await asyncio.to_thread(
                self.llm_service.edit_image,
                prompt,
                input_image_base64,
                aspect_ratio,
            )
            cost = time.perf_counter() - started_at
            logger.info(
                f"[/edit] 模型返回成功 | {trace} | image_count={len(result.images)} | text_len={len(result.text)} | cost_sec={cost:.3f}"
            )
        except Exception as e:
            logger.error(f"[/edit] 处理失败 | {trace} | err={e}")
            logger.error(traceback.format_exc())
            yield event.plain_result(f"编辑失败：{e}")
            return

        if result.text:
            logger.info(f"[/edit] 开始发送文本结果 | {trace} | text_len={len(result.text)}")
            yield event.plain_result(result.text)

        for idx, image_bytes in enumerate(result.images, start=1):
            try:
                logger.info(f"[/edit] 开始落盘第{idx}张结果图 | {trace}")
                image_path = await asyncio.to_thread(
                    self._save_temp_image, image_bytes, f"astrbot_edit_{idx}_"
                )
                logger.info(
                    f"[/edit] 结果图片已落盘 | {trace} | index={idx} | bytes={len(image_bytes)} | path={image_path}"
                )
            except Exception as e:
                logger.error(f"[/edit] 保存编辑结果失败 | {trace} | index={idx} | err={e}")
                logger.error(traceback.format_exc())
                yield event.plain_result(f"编辑成功，但保存图片失败：{e}")
                continue
            logger.info(f"[/edit] 开始发送第{idx}张结果图 | {trace}")
            yield event.image_result(image_path)
        logger.info(f"[/edit] 请求处理结束 | {trace}")

    @filter.command("gen")
    async def generate_image(self, event: AstrMessageEvent):
        """根据提示词生成图片并返回结果。"""
        trace = self._event_trace(event)
        started_at = time.perf_counter()
        logger.info(f"[/gen] 收到请求 | {trace} | raw_input={event.message_str!r}")

        logger.info(f"[/gen] 开始解析指令参数 | {trace}")
        prompt = self._extract_prompt(event.message_str, "gen")
        prompt, aspect_ratio = self._extract_aspect_ratio(prompt)
        logger.info(
            f"[/gen] 参数解析完成 | {trace} | prompt_len={len(prompt)} | aspect_ratio={aspect_ratio}"
        )

        if not prompt:
            logger.warning(f"[/gen] 缺少 prompt | {trace}")
            yield event.plain_result("用法：/gen 提示词 [--ar 16:9]")
            return

        try:
            logger.info(f"[/gen] 开始调用生图模型 | {trace} | model={self.llm_service.model}")
            result = await asyncio.to_thread(
                self.llm_service.generate_image,
                prompt,
                aspect_ratio,
            )
            cost = time.perf_counter() - started_at
            logger.info(
                f"[/gen] 模型返回成功 | {trace} | image_count={len(result.images)} | text_len={len(result.text)} | cost_sec={cost:.3f}"
            )
        except Exception as e:
            logger.error(f"[/gen] 处理失败 | {trace} | err={e}")
            logger.error(traceback.format_exc())
            yield event.plain_result(f"生图失败：{e}")
            return

        if result.text:
            logger.info(f"[/gen] 开始发送文本结果 | {trace} | text_len={len(result.text)}")
            yield event.plain_result(result.text)

        for idx, image_bytes in enumerate(result.images, start=1):
            try:
                logger.info(f"[/gen] 开始落盘第{idx}张结果图 | {trace}")
                image_path = await asyncio.to_thread(
                    self._save_temp_image, image_bytes, f"astrbot_gen_{idx}_"
                )
                logger.info(
                    f"[/gen] 结果图片已落盘 | {trace} | index={idx} | bytes={len(image_bytes)} | path={image_path}"
                )
            except Exception as e:
                logger.error(f"[/gen] 保存生成结果失败 | {trace} | index={idx} | err={e}")
                logger.error(traceback.format_exc())
                yield event.plain_result(f"生图成功，但保存图片失败：{e}")
                continue
            logger.info(f"[/gen] 开始发送第{idx}张结果图 | {trace}")
            yield event.image_result(image_path)
        logger.info(f"[/gen] 请求处理结束 | {trace}")

    async def terminate(self):
        logger.info("astrbot_plugin_multimodal_chat 已卸载")
