"""Telegram bot that produces the daily Tử Vi video and makes videos on demand.

Run with ``python -m app.bot.telegram_bot`` (see start-bot.sh).
"""

import asyncio
import os
import uuid
from datetime import datetime, time
from zoneinfo import ZoneInfo

from loguru import logger
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.config import config
from app.services.tuvi import pipeline, sources

TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_UPLOAD_LIMIT = 50 * 1024 * 1024
MAX_REGEN_ENTRIES = 50

HELP_TEXT = (
    "Bot tạo video Tử Vi hằng ngày\n\n"
    "/homnay - tạo ngay video của hôm nay từ các nguồn\n"
    "/taovideo <chủ đề | link | nội dung> - tạo video theo yêu cầu\n"
    "/nguon - xem danh sách nguồn tin\n"
    "/trangthai - trạng thái bot và video gần nhất\n"
    "/chatid - xem chat id của bạn\n\n"
    "Bài trên Facebook: copy nội dung bài rồi gửi thẳng cho bot (hoặc dùng /taovideo)."
)


def _tuvi_cfg() -> dict:
    return config.tuvi


def get_token() -> str:
    return os.getenv("TUVI_TELEGRAM_BOT_TOKEN") or str(_tuvi_cfg().get("telegram_bot_token", "") or "")


def get_allowed_chat_ids() -> set[int]:
    raw = os.getenv("TUVI_TELEGRAM_CHAT_IDS")
    values = raw.split(",") if raw else _tuvi_cfg().get("telegram_allowed_chat_ids", [])
    chat_ids = set()
    for value in values:
        try:
            chat_ids.add(int(str(value).strip()))
        except ValueError:
            logger.warning(f"invalid telegram chat id: {value}")
    return chat_ids


def get_daily_time() -> time:
    tz = ZoneInfo(_tuvi_cfg().get("timezone", "Asia/Ho_Chi_Minh"))
    hour, minute = str(_tuvi_cfg().get("daily_time", "06:30")).split(":", 1)
    return time(int(hour), int(minute), tzinfo=tz)


def truncate_caption(text: str, limit: int = TELEGRAM_CAPTION_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class BotState:
    def __init__(self):
        self.running_since: datetime | None = None
        self.running_label = ""
        self.last_result: pipeline.DailyVideoResult | None = None
        self.last_error = ""
        self.last_finished: datetime | None = None
        self.regen_items: dict[str, sources.SourceItem] = {}

    def remember(self, item: sources.SourceItem) -> str:
        key = uuid.uuid4().hex[:10]
        self.regen_items[key] = item
        while len(self.regen_items) > MAX_REGEN_ENTRIES:
            self.regen_items.pop(next(iter(self.regen_items)))
        return key


STATE = BotState()


def _is_allowed(update: Update) -> bool:
    chat = update.effective_chat
    return bool(chat and chat.id in get_allowed_chat_ids())


async def _deny(update: Update) -> None:
    chat_id = update.effective_chat.id if update.effective_chat else "?"
    if update.effective_message:
        await update.effective_message.reply_text(
            f"Chat này chưa được cấp quyền. Thêm chat id {chat_id} vào "
            "[tuvi].telegram_allowed_chat_ids trong config.toml rồi khởi động lại bot."
        )


def _result_keyboard(regen_key: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("Tạo lại", callback_data=f"regen:{regen_key}"),
            InlineKeyboardButton("Chủ đề khác", callback_data="daily"),
        ]]
    )


async def _run_generation(application: Application, chat_ids: list[int], label: str, job) -> None:
    """Run a blocking pipeline job in a worker thread and report to ``chat_ids``."""
    bot = application.bot
    if STATE.running_since:
        for chat_id in chat_ids:
            await bot.send_message(chat_id, f"Đang bận tạo video: {STATE.running_label}. Thử lại sau nhé.")
        return

    STATE.running_since = datetime.now()
    STATE.running_label = label
    status_messages = []
    for chat_id in chat_ids:
        status_messages.append(await bot.send_message(chat_id, f"Bắt đầu: {label}"))

    loop = asyncio.get_running_loop()

    async def _edit(text: str) -> None:
        for message in status_messages:
            try:
                await message.edit_text(text)
            except TelegramError:
                pass

    def progress(text: str) -> None:
        logger.info(f"tuvi bot progress: {text}")
        asyncio.run_coroutine_threadsafe(_edit(text), loop)

    try:
        result: pipeline.DailyVideoResult = await asyncio.to_thread(job, progress)
    except Exception as exc:
        logger.exception(f"tuvi video generation failed: {exc}")
        STATE.last_error = str(exc)
        await _edit(f"Tạo video thất bại: {exc}")
        return
    finally:
        STATE.running_since = None
        STATE.running_label = ""
        STATE.last_finished = datetime.now()

    STATE.last_result = result
    STATE.last_error = ""
    regen_key = STATE.remember(result.source_item)
    caption = result.post_text
    if result.source_url:
        caption += f"\n\nNguồn tham khảo: {result.source_url}"
    caption = truncate_caption(caption)

    await _edit(f"Xong: {result.title}")
    size = os.path.getsize(result.video_path)
    for chat_id in chat_ids:
        try:
            if size > TELEGRAM_UPLOAD_LIMIT:
                await bot.send_message(
                    chat_id,
                    f"Video lớn hơn 50MB nên không gửi qua Telegram được.\nFile: {result.video_path}\n\n{caption}",
                    reply_markup=_result_keyboard(regen_key),
                )
                continue
            with open(result.video_path, "rb") as video_file:
                await bot.send_video(
                    chat_id,
                    video=video_file,
                    caption=caption,
                    supports_streaming=True,
                    width=1080,
                    height=1920,
                    reply_markup=_result_keyboard(regen_key),
                    read_timeout=300,
                    write_timeout=300,
                )
        except TelegramError as exc:
            logger.error(f"send video to {chat_id} failed: {exc}")
            await bot.send_message(chat_id, f"Gửi video thất bại ({exc}). File: {result.video_path}")


def _schedule(context: ContextTypes.DEFAULT_TYPE, chat_ids: list[int], label: str, job) -> None:
    context.application.create_task(_run_generation(context.application, chat_ids, label, job))


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _deny(update)
        return
    daily = get_daily_time()
    await update.effective_message.reply_text(
        f"{HELP_TEXT}\n\nVideo tự động mỗi ngày lúc {daily:%H:%M}."
    )


async def cmd_chat_id(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(f"Chat id: {update.effective_chat.id}")


async def cmd_today(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _deny(update)
        return
    _schedule(
        context,
        [update.effective_chat.id],
        "video Tử Vi hôm nay",
        lambda progress: pipeline.generate_daily_video(progress=progress),
    )


async def cmd_make_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _deny(update)
        return
    message = update.effective_message
    text = (message.text or "").split(maxsplit=1)
    topic = text[1].strip() if len(text) > 1 else ""
    if not topic:
        await message.reply_text(
            "Dùng: /taovideo <chủ đề | link bài viết | nội dung bài>\n"
            "Ví dụ: /taovideo Sao Thất Sát ở cung Mệnh"
        )
        return
    _schedule(
        context,
        [update.effective_chat.id],
        f"video: {topic[:60]}",
        lambda progress: pipeline.generate_daily_video(topic, progress=progress),
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Plain or forwarded text (e.g. a copied Facebook post) becomes a video topic."""
    if not _is_allowed(update):
        await _deny(update)
        return
    topic = (update.effective_message.text or "").strip()
    if len(topic) < 4:
        return
    _schedule(
        context,
        [update.effective_chat.id],
        f"video: {topic[:60]}",
        lambda progress: pipeline.generate_daily_video(topic, progress=progress),
    )


async def cmd_sources(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _deny(update)
        return
    urls = sources.get_sources()
    listing = "\n".join(f"- {url}" for url in urls) or "(chưa cấu hình nguồn)"
    await update.effective_message.reply_text(
        f"Nguồn tin ([tuvi].sources trong config.toml):\n{listing}",
        disable_web_page_preview=True,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_allowed(update):
        await _deny(update)
        return
    lines = [f"Lịch hằng ngày: {get_daily_time():%H:%M} ({_tuvi_cfg().get('timezone', 'Asia/Ho_Chi_Minh')})"]
    jobs = context.job_queue.get_jobs_by_name("tuvi-daily") if context.job_queue else []
    if jobs and jobs[0].next_t:
        lines.append(f"Lần chạy tới: {jobs[0].next_t.astimezone(get_daily_time().tzinfo):%d/%m %H:%M}")
    lines.append(f"Giọng đọc: {pipeline.resolve_voice_name()}")
    lines.append(f"LLM: {config.app.get('llm_provider', '')}")
    if STATE.running_since:
        lines.append(f"Đang chạy: {STATE.running_label} (từ {STATE.running_since:%H:%M:%S})")
    if STATE.last_result:
        lines.append(f"Video gần nhất: {STATE.last_result.title}\n{STATE.last_result.video_path}")
    if STATE.last_error:
        lines.append(f"Lỗi gần nhất: {STATE.last_error}")
    await update.effective_message.reply_text("\n".join(lines))


async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if not _is_allowed(update):
        return
    chat_id = update.effective_chat.id
    data = query.data or ""
    if data == "daily":
        _schedule(
            context,
            [chat_id],
            "video với chủ đề khác",
            lambda progress: pipeline.generate_daily_video(progress=progress),
        )
        return
    if data.startswith("regen:"):
        item = STATE.regen_items.get(data.split(":", 1)[1])
        if item is None:
            await context.bot.send_message(chat_id, "Không còn dữ liệu để tạo lại, hãy dùng /taovideo.")
            return
        _schedule(
            context,
            [chat_id],
            f"tạo lại: {item.title[:60]}",
            lambda progress: pipeline.generate_video(item, progress=progress),
        )


async def daily_job(context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_ids = sorted(get_allowed_chat_ids())
    if not chat_ids:
        logger.warning("tuvi daily job skipped: no telegram_allowed_chat_ids configured")
        return
    await _run_generation(
        context.application,
        chat_ids,
        "video Tử Vi hằng ngày",
        lambda progress: pipeline.generate_daily_video(progress=progress),
    )


async def _post_init(application: Application) -> None:
    await application.bot.set_my_commands(
        [
            ("homnay", "Tạo video Tử Vi hôm nay"),
            ("taovideo", "Tạo video theo chủ đề/link/nội dung"),
            ("nguon", "Danh sách nguồn tin"),
            ("trangthai", "Trạng thái bot"),
            ("chatid", "Xem chat id"),
        ]
    )


def build_application(token: str) -> Application:
    application = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .concurrent_updates(True)
        .build()
    )
    application.add_handler(CommandHandler(["start", "help"], cmd_start))
    application.add_handler(CommandHandler("chatid", cmd_chat_id))
    application.add_handler(CommandHandler("homnay", cmd_today))
    application.add_handler(CommandHandler("taovideo", cmd_make_video))
    application.add_handler(CommandHandler("nguon", cmd_sources))
    application.add_handler(CommandHandler("trangthai", cmd_status))
    application.add_handler(CallbackQueryHandler(on_button))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    application.job_queue.run_daily(daily_job, time=get_daily_time(), name="tuvi-daily")
    return application


def main() -> None:
    token = get_token()
    if not token:
        raise SystemExit(
            "Thiếu Telegram bot token: điền [tuvi].telegram_bot_token trong config.toml "
            "hoặc đặt biến môi trường TUVI_TELEGRAM_BOT_TOKEN."
        )
    if not get_allowed_chat_ids():
        logger.warning(
            "telegram_allowed_chat_ids is empty: send /chatid to the bot, add the id to "
            "config.toml and restart"
        )
    logger.info(f"tuvi bot started, daily video at {get_daily_time():%H:%M}")
    build_application(token).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
