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

Người xem là người Việt 20-40 tuổi, tò mò về tử vi nhưng ngại sách vở. Họ ở lại xem khi thấy mình trong câu chuyện, và chia sẻ khi nhận ra điều chưa từng nghĩ tới.

QUY TẮC NỘI DUNG (quan trọng nhất)
- Video dài 75-90 giây khi đọc thành tiếng (tổng lời đọc khoảng 230-290 từ), mỗi cảnh khoảng 30-40 từ.
- MỘT MẠCH DUY NHẤT: hook đặt ra một câu hỏi hoặc một điều lạ; các cảnh giữa lần lượt trả lời nó từng lớp; cảnh gần cuối chốt lại bằng một nhận ra mà người xem mang về; cảnh cuối là lời kêu gọi. Cảnh sau PHẢI nối từ cảnh trước bằng chính ý đang nói ("Và vì thế...", "Nhưng có một điều ngược lại...", "Thử nghĩ xem..."), không được liệt kê rời rạc như các gạch đầu dòng.
- CHIỀU SÂU: mỗi cảnh mang MỘT ý cụ thể và mới: ý nghĩa gốc của tên cung hay sao, lĩnh vực cung đó quản, mối liên hệ với cung khác, cách nó hiện ra trong đời sống hằng ngày, một hiểu lầm phổ biến, hoặc một tình huống đời thường để người xem tự soi mình. Không nói chung chung kiểu "cung này rất quan trọng", "mỗi người một khác". Một ý chỉ được nói một lần.
- ĐỘ CHÍNH XÁC: chỉ nêu kiến thức tử vi bạn chắc chắn đúng theo sách tử vi phổ thông (tên cung, tên sao, thứ tự cung). Không bịa tên sao, luật hay con số; nếu không chắc thì nói ở mức khái quát.
- Dùng DÀN Ý trong tư liệu (nếu có) làm xương sống: bám câu hỏi trung tâm, ẩn dụ xuyên suốt, các ý cốt lõi và điều nhận ra cuối video; viết lại bằng lời tự nhiên chứ không chép lại dàn ý.
- Viết lại hoàn toàn bằng lời của bạn; KHÔNG sao chép nguyên văn câu nào từ tư liệu, không nhắc tên trang nguồn.
- Không phán đoán tuyệt đối, không hù dọa, không mê tín cực đoan, không hứa hẹn tiền bạc, sức khỏe, không đưa lời khuyên y tế hay tài chính cụ thể. Lời rào đón chỉ nói MỘT lần duy nhất, ở cảnh gần cuối, bằng giọng tự nhiên ("Lá số là tấm bản đồ, đường đi vẫn là của bạn"); không lặp "mang tính tham khảo", "gợi mở nhẹ nhàng" ở nhiều cảnh.
- Tránh công thức sáo rỗng: "Theo tử vi cổ" tối đa 1 lần cả video; không mở hai cảnh liền nhau bằng cùng một kiểu câu.
- Giọng kể trầm ấm, như người am hiểu kể cho bạn thân nghe, hơi huyền bí. Có thể xưng "mình" hoặc "ta", gọi người xem là "bạn" nhưng đừng lặp "bạn" ở mọi câu.
- Câu mở đầu (hook) phải gây tò mò trong 3 giây đầu. Cảnh cuối là lời kêu gọi nhẹ nhàng: "{cta_text}", chỉ MỘT câu kêu gọi.
- Lời đọc chỉ gồm chữ đọc được: không emoji, không ký hiệu, không hashtag; số và ngày tháng viết bằng chữ khi đọc ("hai mươi mốt tháng chín").

NHỊP ĐỌC (để giọng đọc máy nghe tự nhiên; chỉ chỉnh câu chữ, không được hy sinh ý)
- Giọng đọc máy ngắt nhẹ ở dấu phẩy, ngắt rõ ở dấu chấm, chấm hỏi, chấm than. Không dùng chấm phẩy, ba chấm, gạch ngang, hai chấm, ngoặc vì chúng không tạo nhịp tốt hơn.
- Độ dài câu thay đổi tự nhiên: xen câu rất ngắn (3-6 từ) làm điểm nhấn ở chỗ thật sự cần với câu vừa và câu dài hơn (13-18 từ). Dấu phẩy chỉ đặt ở ranh giới ý, mỗi vế dài ít nhất 3 từ, tối đa 2 dấu phẩy mỗi câu; không đặt phẩy sau một từ lẻ ("Còn,", "Và,").
- Không để quá 3 câu liên tiếp cùng kiểu mở (ví dụ đều bắt đầu bằng "Bạn") hoặc cùng là câu hỏi.
- Từ khóa cần nhấn đặt ở CUỐI vế hoặc cuối câu. Câu hỏi tu từ chỉ dùng khi dẫn vào ý tiếp theo, tối đa 3 lần cả video; dấu chấm than tối đa 1 lần cả video.

HÌNH ẢNH
- Mỗi cảnh có một "image_prompt" tiếng Anh, chỉ là ý tưởng ngắn về vật hoặc cảnh minh họa đúng điều lời đọc đang nói (hệ thống sẽ có bước họa sĩ biên tập lại). Không mô tả chữ, không dùng hình ảnh đáng sợ, máu me, ma quỷ.

ĐẦU RA: CHỈ một JSON object hợp lệ, không markdown, theo đúng cấu trúc:
{{
  "title": "tiêu đề video ngắn gọn, tối đa 60 ký tự",
  "hook": "câu mở đầu gây tò mò",
  "scenes": [
    {{
      "heading": "tiêu đề ngắn của cảnh, tối đa 40 ký tự",
      "narration": "lời đọc của cảnh này (30-40 từ), một ý cụ thể, nối tiếp mạch từ cảnh trước",
      "image_prompt": "ý tưởng hình minh họa bằng TIẾNG ANH, bám sát lời đọc của cảnh",
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
{outline_block}
Hãy viết kịch bản video theo đúng định dạng JSON đã yêu cầu."""


OUTLINE_PROMPT = """Bạn là biên tập viên nội dung tử vi, am hiểu tử vi phổ thông Việt Nam, đang chuẩn bị một video ngắn {scene_count} cảnh cho kênh "{app_name}".
Hãy lên DÀN Ý cho chủ đề dưới đây. CHƯA viết lời đọc. Trả lời bằng văn bản thuần (không JSON, không markdown), tối đa 350 từ, gồm đúng các mục:

1. CÂU HỎI TRUNG TÂM: một câu hỏi đời thường mà người xem thật sự băn khoăn, liên quan đến chủ đề.
2. ẨN DỤ XUYÊN SUỐT: một hình ảnh đời thường duy nhất để giải thích chủ đề (ví dụ ngôi nhà, con đường, cái cây).
3. KIẾN THỨC CỐT LÕI (4-6 ý, mỗi ý một dòng, cụ thể): nghĩa của tên cung hoặc sao, lĩnh vực cung quản, liên hệ với các cung khác, cách biểu hiện trong đời sống. Chỉ ghi điều bạn chắc chắn đúng theo tử vi phổ thông; điều chỉ chắc ở mức chung thì ghi thêm "(khái quát)". Không bịa tên sao hay luật.
4. HIỂU LẦM PHỔ BIẾN: một hiểu lầm người ta hay mắc và sự thật đằng sau.
5. NHẬN RA CUỐI VIDEO: một điều người xem chưa từng nghĩ tới, 1-2 câu, ấm áp, không phán đoán tuyệt đối.
6. MẠCH {scene_count} CẢNH: mỗi cảnh một dòng "Cảnh n: mục đích, ý chính, cầu nối sang cảnh sau". Cảnh 1 là hook, cảnh cuối là lời kêu gọi tải app."""

OUTLINE_USER_PROMPT = """Ngày hôm nay: {today}
Chủ đề: {title}

Tư liệu tham khảo (nếu có):
\"\"\"
{text}
\"\"\""""

OUTLINE_BLOCK = """
DÀN Ý ĐÃ DUYỆT (bám sát mạch này, viết lại bằng lời tự nhiên):
\"\"\"
{outline}
\"\"\"
"""

MAX_OUTLINE_CHARS = 3000

ILLUSTRATOR_PROMPT = """Bạn là đạo diễn hình ảnh cho video tử vi ngắn. Dưới đây là {scene_count} cảnh, mỗi cảnh có lời đọc. Hãy viết MỘT prompt tiếng Anh cho model vẽ tranh cho từng cảnh.

NGUYÊN TẮC
1. Minh họa TRỰC TIẾP điều lời đọc đang nói. Tìm danh từ hoặc khái niệm chính của cảnh (lá số, một cung, ngôi nhà, mảnh đất, ngôi sao, bầu trời đêm, la bàn, dòng sông...) và vẽ chính nó. Nhìn ảnh phải đoán được cảnh đang nói về gì. Không vẽ người chỉ để diễn tả cảm xúc.
2. Cảnh nào lời đọc nói về con người thì vẽ nhân vật Việt Nam xưa như bình thường (áo the, áo dài, khăn xếp, khăn vấn), không hạn chế số cảnh có người; cảnh nói về vật, nơi chốn hay khái niệm thì vẽ vật, cảnh vật hoặc biểu tượng.
3. Gợi ý hình ảnh theo khái niệm:
   - lá số: a square Tu Vi horoscope chart drawn as a ring of twelve small empty cells (four per side, corners shared) around one large empty central square, thin ink lines, the cells hold no writing
   - một cung cụ thể: one cell of that twelve-cell grid softly highlighted with a warm wash while the others stay pale
   - sao, vận mệnh: a constellation of small stars joined by thin lines over a night sky; one bright star above a rooftop
   - nhà cửa: a tile-roofed three-room Vietnamese house, a wooden stilt house, a courtyard gate, a threshold, a bamboo fence
   - đất đai, tài sản: a rice field with a boundary stone, a handful of soil, a plot of land with wooden pegs and red string
   - Mệnh, Thân, lựa chọn: a feng shui compass on a table, two paths in a field, a bronze mirror
   - thời gian, đại vận: moon phases in a row, a river bending, falling leaves, a sundial
   - gia đình, tổ ấm: a tea set with two cups, two bowls of rice, a pair of sandals at a door
   - hiểu lầm, nhận ra: a cracked jar being mended, a door opening onto light, a lantern in the dark
4. Ảnh không được có chữ: đừng chọn bảng hiệu, sách hay cuộn giấy có chữ, chữ Hán; lá số thì vẽ các ô trống (hệ thống tự chặn chữ).
5. Chủ thể phải LỚN và rõ: chiếm khoảng một nửa đến hai phần ba chiều rộng khung, đừng để nó nhỏ li ti giữa khoảng trống; hãy viết "large", "filling half the frame" thay vì "wide empty space". Chỉ mô tả thứ CÓ trong ảnh, KHÔNG viết câu phủ định ("no people", "no writing", "without text"...) vì model vẽ sẽ vẽ đúng thứ bị nhắc tên. Mỗi prompt 20-40 từ: một chủ thể chính đặt giữa khung, xung quanh nhiều khoảng trống. Không nói phong cách vẽ, màu nền hay chất liệu (hệ thống tự thêm).
6. Đa dạng: hai cảnh liền nhau phải khác hẳn chủ thể và bố cục (kể cả vật nhỏ trên nền kín thì chỉ dùng tối đa 1 lần); không lặp lại cùng một vật ở hai cảnh.
7. Cảnh cuối (kêu gọi tải app): chủ thể nằm ở nửa trên khung hình, một phần ba phía dưới để trống. Đừng vẽ điện thoại, màn hình, bàn tay thật hay biểu tượng hiện đại (model vẽ quá giống ảnh chụp và chèn ký hiệu lạ); dùng một hình tượng cổ, ví dụ ngôi sao sáng trên mái nhà hay ngọn đèn trước cửa.

ĐẦU RA: CHỈ một JSON array đúng {scene_count} phần tử theo thứ tự cảnh, không markdown:
[{{"scene": 1, "key_visual": "khái niệm chính của cảnh, tiếng Việt, 2-5 từ", "image_prompt": "..."}}]"""

ILLUSTRATOR_USER_PROMPT = """Chủ đề video: {title}

Các cảnh:
{scenes}"""

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
    key_visual: str = ""


@dataclass
class VideoContent:
    title: str
    hook: str
    scenes: list[Scene]
    cta: str
    caption: str
    hashtags: list[str] = field(default_factory=list)
    outline: str = ""

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
    text = re.sub(r"[*_`~>|\[\]{}()]", "", text)
    # VieNeu pauses no longer on ; : or a dash than on a comma (measured), so fold them into one.
    text = re.sub(r"\s*[;—–]\s*|\s+-\s+|:\s+", ", ", text)
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r",\s*([.!?…])", r"\1", text)
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


def _source_text(item: SourceItem) -> str:
    return (item.text or "(không có tư liệu, hãy dựa trên kiến thức tử vi phổ thông)")[:6000]


def build_outline_prompt(item: SourceItem, today: str, scene_count: int, app_name: str) -> str:
    system = OUTLINE_PROMPT.format(scene_count=scene_count, app_name=app_name)
    user = OUTLINE_USER_PROMPT.format(today=today, title=item.title, text=_source_text(item))
    return f"{system}\n\n{user}"


def build_prompt(
    item: SourceItem, today: str, scene_count: int, app_name: str, cta_text: str, outline: str = ""
) -> str:
    system = SYSTEM_PROMPT.format(
        app_name=app_name,
        cta_text=cta_text,
        zodiac_keys=", ".join(ZODIAC_KEYS),
        scene_count=scene_count,
    )
    user = USER_PROMPT.format(
        today=today,
        title=item.title,
        text=_source_text(item),
        outline_block=OUTLINE_BLOCK.format(outline=outline) if outline else "",
    )
    return f"{system}\n\n{user}"


def generate_outline(item: SourceItem, today: str, scene_count: int, app_name: str, generate) -> str:
    """Step 1: a depth-first outline (central question, metaphor, facts, arc). Never fatal:
    on any failure the script is simply written without it."""
    try:
        response = generate(build_outline_prompt(item, today, scene_count, app_name))
    except Exception as exc:
        logger.warning(f"tuvi outline failed, writing without it: {exc}")
        return ""
    if not isinstance(response, str) or response.startswith("Error:"):
        logger.warning(f"tuvi outline unavailable, writing without it: {str(response)[:120]}")
        return ""
    return re.sub(r"\n{3,}", "\n\n", response).strip()[:MAX_OUTLINE_CHARS]


def build_illustrator_prompt(video_content: "VideoContent") -> str:
    count = len(video_content.scenes)
    system = ILLUSTRATOR_PROMPT.format(scene_count=count)
    scenes = "\n".join(f"Cảnh {i}: {scene.narration}" for i, scene in enumerate(video_content.scenes, start=1))
    return f"{system}\n\n{ILLUSTRATOR_USER_PROMPT.format(title=video_content.title, scenes=scenes)}"


def apply_image_prompts(video_content: "VideoContent", response) -> bool:
    """Copy the illustrator's prompts onto the scenes; all-or-nothing so a bad reply keeps the script's own."""
    if not isinstance(response, str) or response.startswith("Error:"):
        return False
    match = re.search(r"\[.*\]", response, re.S)
    try:
        items = json.loads(match.group(0)) if match else None
    except ValueError:
        return False
    if not isinstance(items, list) or len(items) != len(video_content.scenes):
        return False
    prompts = []
    for item in items:
        prompt = _clean(item.get("image_prompt") if isinstance(item, dict) else "", 600)
        if not prompt:
            return False
        prompts.append((prompt, _clean(item.get("key_visual"), 60)))
    for scene, (prompt, key_visual) in zip(video_content.scenes, prompts):
        scene.image_prompt, scene.key_visual = prompt, key_visual
    return True


def generate_image_prompts(video_content: "VideoContent", generate) -> bool:
    """Step 3: an illustrator pass that draws what each narration actually talks about."""
    try:
        response = generate(build_illustrator_prompt(video_content))
    except Exception as exc:
        logger.warning(f"tuvi illustrator failed, keeping script image prompts: {exc}")
        return False
    if not apply_image_prompts(video_content, response):
        logger.warning("tuvi illustrator reply unusable, keeping script image prompts")
        return False
    return True


def generate_content(item: SourceItem, today: str, generate=None) -> VideoContent:
    """Ask the configured LLM (DeepSeek) for a scene-by-scene script."""
    tuvi_cfg = config.tuvi
    scene_count = max(MIN_SCENES, min(MAX_SCENES, int(tuvi_cfg.get("scene_count", 8))))
    app_name = tuvi_cfg.get("app_name", "Tử Vi AI")
    cta_text = tuvi_cfg.get("cta_text", f"Tải app {app_name} để xem lá số của riêng bạn")
    generate = generate or llm._generate_response
    outline = generate_outline(item, today, scene_count, app_name, generate)
    prompt = build_prompt(item, today, scene_count, app_name, cta_text, outline)

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = generate(prompt)
        if isinstance(response, str) and response.startswith("Error:"):
            # Provider/config errors (missing key, auth) will not fix themselves on retry.
            raise ContentError(response.removeprefix("Error:").strip())
        try:
            video_content = parse_content(extract_json(response), cta_text)
            video_content.outline = outline
            generate_image_prompts(video_content, generate)
            return video_content
        except ContentError as exc:
            last_error = exc
            logger.warning(f"tuvi content attempt {attempt} invalid: {exc}")
    raise ContentError(f"LLM did not return a valid script: {last_error}")
