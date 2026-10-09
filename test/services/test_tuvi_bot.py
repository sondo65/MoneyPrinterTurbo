import pytest

pytest.importorskip("telegram")

from app.bot import telegram_bot  # noqa: E402
from app.config import config  # noqa: E402
from app.services.tuvi.sources import SourceItem  # noqa: E402


def test_allowed_chat_ids_from_config_and_env(monkeypatch):
    monkeypatch.delenv("TUVI_TELEGRAM_CHAT_IDS", raising=False)
    monkeypatch.setattr(config, "tuvi", {"telegram_allowed_chat_ids": [123, "-456", "bad"]})
    assert telegram_bot.get_allowed_chat_ids() == {123, -456}

    monkeypatch.setenv("TUVI_TELEGRAM_CHAT_IDS", "7, 8")
    assert telegram_bot.get_allowed_chat_ids() == {7, 8}


def test_daily_time_uses_configured_timezone(monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"daily_time": "07:15", "timezone": "Asia/Ho_Chi_Minh"})
    daily = telegram_bot.get_daily_time()

    assert (daily.hour, daily.minute) == (7, 15)
    assert str(daily.tzinfo) == "Asia/Ho_Chi_Minh"


def test_truncate_caption():
    assert telegram_bot.truncate_caption("ngắn") == "ngắn"
    long_caption = telegram_bot.truncate_caption("a" * 2000)
    assert len(long_caption) == telegram_bot.TELEGRAM_CAPTION_LIMIT
    assert long_caption.endswith("…")


def test_regen_memory_is_bounded():
    state = telegram_bot.BotState()
    keys = [state.remember(SourceItem(title=str(i), text="")) for i in range(60)]

    assert len(state.regen_items) == telegram_bot.MAX_REGEN_ENTRIES
    assert keys[0] not in state.regen_items
    assert state.regen_items[keys[-1]].title == "59"
    assert all(len(f"regen:{key}".encode()) <= 64 for key in keys)


def test_build_application_registers_daily_job(monkeypatch):
    monkeypatch.setattr(config, "tuvi", {"daily_time": "06:30"})
    application = telegram_bot.build_application("123456:TEST")

    assert [job.name for job in application.job_queue.jobs()] == ["tuvi-daily"]
