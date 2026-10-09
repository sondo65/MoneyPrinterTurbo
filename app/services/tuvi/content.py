import json
import re
from dataclasses import asdict, dataclass, field

from loguru import logger

from app.config import config
from app.services import llm
from app.services.tuvi.sources import SourceItem

ZODIAC_KEYS = (
    "ty", "suu", "dan", "mao", "thin", "ti", "ngo", "mui", "than", "dau", "tuat", "hoi",
)
MIN_SCENES = 3
MAX_SCENES = 10
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """Bạn là biên kịch video ngắn (TikTok/Reels/Shorts) cho kênh của ứng dụng "{app_name}".
Phong cách: người kể chuyện trầm ấm, gần gũi, hơi huyền bí, tiếng Việt tự nhiên có dấu đầy đủ.

QUY TẮC NỘI DUNG
- Video dài 75-90 giây khi đọc thành tiếng (tổng lời đọc khoảng 230-290 từ), mỗi cảnh 30-40 từ.
- Viết lại hoàn toàn bằng lời của bạn; KHÔNG sao chép nguyên văn câu nào từ tư liệu, không nhắc tên trang nguồn.
- Không phán đoán tuyệt đối, không hù dọa, không mê tín cực đoan; dùng cách nói "theo tử vi cổ", "có xu hướng", "nên lưu ý".
- Không hứa hẹn tiền bạc, sức khỏe, không đưa lời khuyên y tế/tài chính cụ thể.
- Câu mở đầu (hook) phải gây tò mò trong 3 giây đầu.
- Cảnh cuối là lời kêu gọi nhẹ nhàng: "{cta_text}".
- Lời đọc chỉ gồm chữ đọc được: không emoji, không ký hiệu, không hashtag, viết số bằng chữ khi đọc tự nhiên hơn.
- Viết như đang kể chuyện thành tiếng, có nhịp ngắt nghỉ rõ ràng vì giọng đọc máy chỉ ngừng ở dấu câu:
  + Câu ngắn, 6-14 từ mỗi câu; kết thúc bằng dấu chấm, dấu chấm hỏi hoặc dấu chấm than.
  + Đặt dấu phẩy ở chỗ người kể tự nhiên sẽ ngừng lấy hơi: sau trạng ngữ ("Theo tử vi cổ,"), trước từ nối ("nhưng", "và rồi"), giữa các vế liệt kê.
  + Thỉnh thoảng dùng câu hỏi tu từ để tạo điểm nhấn. Không dùng dấu chấm phẩy, dấu ba chấm, dấu ngoặc.

QUY TẮC HÌNH ẢNH (tranh minh họa cổ phong Việt Nam, vẽ tay trên nền giấy dó)
- MỌI cảnh đều có tranh minh họa riêng, kể cả cảnh cuối. Cảnh 1 (hook) cần image_prompt thật cuốn hút để giữ chân người xem.
- Cảnh cuối (CTA): chủ thể nằm ở nửa trên khung hình, một phần ba phía dưới để trống (hệ thống đặt logo app vào đó).
- "image_prompt" viết bằng tiếng Anh, mô tả MỘT hình ảnh đơn giản, mang tính ẩn dụ cho lời đọc: người Việt xưa mặc áo the, áo dài, khăn xếp, khăn vấn;
  hoặc đồ vật cổ như cán cân, bình phong, ô giấy, đèn dầu, chén trà, sách cổ, la bàn phong thủy, cửa song gỗ, con thuyền, cây đa.
- Chủ thể đặt giữa khung hình, xung quanh nhiều khoảng trống. Không mô tả chữ, không mô tả nền, không nói phong cách vẽ (hệ thống tự thêm).
- Không dùng hình ảnh đáng sợ, máu me, ma quỷ.

ĐẦU RA: CHỈ một JSON object hợp lệ, không markdown, theo đúng cấu trúc:
{{
  "title": "tiêu đề video ngắn gọn, tối đa 60 ký tự",
  "hook": "câu mở đầu gây tò mò",
  "scenes": [
    {{
      "heading": "tiêu đề ngắn của cảnh, tối đa 40 ký tự",
      "narration": "lời đọc của cảnh này, 3-4 câu ngắn có dấu phẩy, dấu chấm ngắt nhịp",
      "image_prompt": "mô tả tranh minh họa bằng TIẾNG ANH (xem quy tắc hình ảnh)",
      "zodiac": "một trong {zodiac_keys} nếu cảnh nói về con giáp đó, ngược lại để chuỗi rỗng"
    }}
  ],
  "cta": "câu kêu gọi cuối video",
  "caption": "mô tả đăng kèm video, 1-3 câu",
  "hashtags": ["#tuvi", "..."]
}}
Số cảnh: đúng {scene_count} cảnh (cảnh đầu chứa hook, cảnh cuối là CTA)."""

USER_PROMPT = """Ngày hôm nay: {today}
Chủ đề: {title}

Tư liệu tham khảo (chỉ dùng để lấy ý, phải viết lại):
\"\"\"
{text}
\"\"\"

Hãy viết kịch bản video theo đúng định dạng JSON đã yêu cầu."""


DEFAULT_IMAGE_PROMPT = (
    "an old Vietnamese astrologer in a dark ao the and khan xep studying a horoscope scroll "
    "by a small oil lamp, calm and mysterious"
)


@dataclass
class Scene:
    heading: str
    text: str
    narration: str
    zodiac: str = ""
    image_prompt: str = ""


@dataclass
class VideoContent:
    title: str
    hook: str
    scenes: list[Scene]
    cta: str
    caption: str
    hashtags: list[str] = field(default_factory=list)

    @property
    def narration(self) -> str:
        return "\n".join(scene.narration for scene in self.scenes)

    @property
    def post_text(self) -> str:
        tags = " ".join(self.hashtags)
        return f"{self.title}\n\n{self.caption}\n\n{tags}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


class ContentError(ValueError):
    pass


def _clean(value, limit: int = 0) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if limit and len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _clean_narration(value) -> str:
    text = _clean(value)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[*_`~>|\[\]{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_zodiac(value) -> str:
    key = _clean(value).lower()
    return key if key in ZODIAC_KEYS else ""


def _normalize_hashtag(value) -> str:
    tag = re.sub(r"[\s#]+", "", str(value or ""))
    return f"#{tag}" if tag else ""


def extract_json(response: str) -> dict:
    text = (response or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        data = json.loads(text)
    except ValueError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise ContentError("LLM response contains no JSON object")
        try:
            data = json.loads(match.group(0))
        except ValueError as exc:
            raise ContentError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ContentError("LLM response JSON is not an object")
    return data


def parse_content(data: dict, cta_text: str = "") -> VideoContent:
    raw_scenes = data.get("scenes")
    if not isinstance(raw_scenes, list):
        raise ContentError("scenes must be a list")

    scenes = []
    for raw in raw_scenes:
        if not isinstance(raw, dict):
            continue
        narration = _clean_narration(raw.get("narration"))
        if not narration:
            continue
        scenes.append(
            Scene(
                heading=_clean(raw.get("heading"), 48),
                text=_clean(raw.get("text"), 170),
                narration=narration,
                zodiac=_normalize_zodiac(raw.get("zodiac")),
                image_prompt=_clean(raw.get("image_prompt"), 600) or DEFAULT_IMAGE_PROMPT,
            )
        )

    if len(scenes) < MIN_SCENES:
        raise ContentError(f"need at least {MIN_SCENES} scenes with narration, got {len(scenes)}")
    scenes = scenes[:MAX_SCENES]

    title = _clean(data.get("title"), 80)
    if not title:
        raise ContentError("title is empty")

    cta = _clean(data.get("cta")) or cta_text
    hashtags = [tag for tag in map(_normalize_hashtag, data.get("hashtags") or []) if tag]
    if "#tuvi" not in [tag.lower() for tag in hashtags]:
        hashtags.insert(0, "#tuvi")

    return VideoContent(
        title=title,
        hook=_clean(data.get("hook")) or scenes[0].narration,
        scenes=scenes,
        cta=cta,
        caption=_clean(data.get("caption")) or title,
        hashtags=hashtags[:10],
    )


def build_prompt(item: SourceItem, today: str, scene_count: int, app_name: str, cta_text: str) -> str:
    system = SYSTEM_PROMPT.format(
        app_name=app_name,
        cta_text=cta_text,
        zodiac_keys=", ".join(ZODIAC_KEYS),
        scene_count=scene_count,
    )
    user = USER_PROMPT.format(
        today=today,
        title=item.title,
        text=(item.text or "(không có tư liệu, hãy dựa trên kiến thức tử vi phổ thông)")[:6000],
    )
    return f"{system}\n\n{user}"


def generate_content(item: SourceItem, today: str, generate=None) -> VideoContent:
    """Ask the configured LLM (DeepSeek) for a scene-by-scene script."""
    tuvi_cfg = config.tuvi
    scene_count = max(MIN_SCENES, min(MAX_SCENES, int(tuvi_cfg.get("scene_count", 8))))
    app_name = tuvi_cfg.get("app_name", "Tử Vi AI")
    cta_text = tuvi_cfg.get("cta_text", f"Tải app {app_name} để xem lá số của riêng bạn")
    prompt = build_prompt(item, today, scene_count, app_name, cta_text)
    generate = generate or llm._generate_response

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = generate(prompt)
        if isinstance(response, str) and response.startswith("Error:"):
            # Provider/config errors (missing key, auth) will not fix themselves on retry.
            raise ContentError(response.removeprefix("Error:").strip())
        try:
            return parse_content(extract_json(response), cta_text)
        except ContentError as exc:
            last_error = exc
            logger.warning(f"tuvi content attempt {attempt} invalid: {exc}")
    raise ContentError(f"LLM did not return a valid script: {last_error}")
