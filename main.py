import asyncio
import os
import re
import tempfile
import time
import traceback

from astrbot.api import AstrBotConfig
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Plain
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
        self._log_current_config(stage="__init__")

    @staticmethod
    def _extract_prompt(message: str, command_names: list[str]) -> str:
        text = (message or "").strip()
        escaped_names = [re.escape(name) for name in command_names if name]
        escaped_names.sort(key=len, reverse=True)
        pattern = rf"^/?(?:{'|'.join(escaped_names)})(?:\s+|$)"
        text = re.sub(pattern, "", text, count=1)
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

    @staticmethod
    def _mask_key(raw: str) -> str:
        value = (raw or "").strip()
        if not value:
            return "<empty>"
        if len(value) <= 8:
            return value[0] + "***" + value[-1]
        return f"{value[:4]}***{value[-4:]}"

    def _config_snapshot(self) -> str:
        api_base = str(self.config.get("api_base", "")).strip()
        api_key = str(self.config.get("api_key", "")).strip()
        prefix = str(self.config.get("prefix", "")).strip()
        model = str(self.config.get("model", "")).strip()
        return (
            f"api_base={api_base or '<empty>'} "
            f"api_key={self._mask_key(api_key)} "
            f"prefix={prefix or '<empty>'} "
            f"model={model or '<empty>'}"
        )

    def _log_current_config(self, stage: str) -> None:
        logger.info(f"[config] stage={stage} | {self._config_snapshot()}")

    def _refresh_runtime_config(self, trace: str, command_name: str) -> None:
        self.llm_service.reload_config(self.config)
        self._log_current_config(stage=f"{command_name}.runtime")
        api_key = str(self.config.get("api_key", "")).strip()
        if not api_key:
            logger.warning(f"[{command_name}] 检测到 api_key 为空，调用会失败 | {trace}")

    def _log_runtime_diagnostics(self) -> None:
        try:
            global_cfg = self.context.get_config()
            wake_prefix = str(global_cfg.get("provider_wake_prefix", "")).strip()
        except Exception:
            wake_prefix = "<unknown>"
        logger.info(
            "[diag] plugin_loaded | "
            f"file={__file__} | "
            "commands=生成/gen/genimg, 编辑/edit/editimg | "
            f"provider_wake_prefix={wake_prefix or '<empty>'}",
        )
        if wake_prefix and wake_prefix != "/":
            logger.warning(
                "[diag] 当前 provider_wake_prefix 不是 '/'，标准 command 过滤器会受此影响。"
            )

    @staticmethod
    def _extract_message_text(event: AstrMessageEvent) -> str:
        plain_components = [
            comp.text for comp in event.get_messages() if isinstance(comp, Plain) and comp.text
        ]
        if plain_components:
            return " ".join(plain_components).strip()
        return (event.message_str or "").strip()

    async def initialize(self):
        logger.info("astrbot_plugin_multimodal_chat 初始化完成")
        self._log_current_config(stage="initialize")
        self._log_runtime_diagnostics()

    @filter.event_message_type(filter.EventMessageType.ALL, priority=5)
    async def fallback_message_router(self, event: AstrMessageEvent):
        """兜底命令路由：当标准 command 过滤链未命中时，仍能处理 /生成 与 /编辑。"""
        if event.get_extra("mmc_routed", False):
            return

        text = self._extract_message_text(event)
        if not text.startswith("/"):
            return

        trace = self._event_trace(event)
        if re.match(r"^/(?:生成|gen|genimg)(?:\s|$)", text):
            event.set_extra("mmc_routed", True)
            logger.info(f"[fallback] 命中兜底生图路由 | {trace} | raw_input={text!r}")
            async for result in self.generate_image(event):
                yield result
            event.stop_event()
            return

        if re.match(r"^/(?:编辑|edit|editimg)(?:\s|$)", text):
            event.set_extra("mmc_routed", True)
            logger.info(f"[fallback] 命中兜底编辑路由 | {trace} | raw_input={text!r}")
            async for result in self.edit_image(event):
                yield result
            event.stop_event()
            return

    @filter.command("编辑", alias={"edit", "editimg"})
    async def edit_image(self, event: AstrMessageEvent):
        """编辑用户发送的图片并返回结果。"""
        event.set_extra("mmc_routed", True)
        trace = self._event_trace(event)
        started_at = time.perf_counter()
        logger.info(f"[/edit] 收到请求 | {trace} | raw_input={event.message_str!r}")
        self._refresh_runtime_config(trace, "/edit")

        logger.info(f"[/edit] 开始解析指令参数 | {trace}")
        prompt = self._extract_prompt(event.message_str, ["编辑", "edit", "editimg"])
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
                f"[/edit] 输入图片准备完成 | {trace} | image_b64_len={len(input_image_base64)} | model={self.llm_service._build_model_name()}"
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

    @filter.command("生成", alias={"gen", "genimg"})
    async def generate_image(self, event: AstrMessageEvent):
        """根据提示词生成图片并返回结果。"""
        event.set_extra("mmc_routed", True)
        trace = self._event_trace(event)
        started_at = time.perf_counter()
        logger.info(f"[/gen] 收到请求 | {trace} | raw_input={event.message_str!r}")
        self._refresh_runtime_config(trace, "/gen")

        logger.info(f"[/gen] 开始解析指令参数 | {trace}")
        prompt = self._extract_prompt(event.message_str, ["生成", "gen", "genimg"])
        prompt, aspect_ratio = self._extract_aspect_ratio(prompt)
        logger.info(
            f"[/gen] 参数解析完成 | {trace} | prompt_len={len(prompt)} | aspect_ratio={aspect_ratio}"
        )

        if not prompt:
            logger.warning(f"[/gen] 缺少 prompt | {trace}")
            yield event.plain_result("用法：/gen 提示词 [--ar 16:9]")
            return

        try:
            logger.info(
                f"[/gen] 开始调用生图模型 | {trace} | model={self.llm_service._build_model_name()}"
            )
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
