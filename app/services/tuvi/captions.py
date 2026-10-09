"""Centre-screen karaoke captions (2-3 words, current word in gold) as an ASS script.

Narration is synthesised per scene, so each scene's speech span is known exactly.
Vietnamese is monosyllabic (one written word = one syllable), so spreading the
span across words with extra weight for punctuation pauses tracks speech closely
without a forced aligner.
"""

import re
from dataclasses import dataclass

CAPTION_Y = 960
MAX_WORDS_PER_CHUNK = 3
MAX_CHARS_PER_CHUNK = 18
# Extra time weight (in "syllables") for the pause after a punctuation mark.
PAUSE_WEIGHTS = {",": 0.8, ";": 0.9, ":": 0.9, ".": 1.4, "!": 1.4, "?": 1.4, "…": 1.4}
MAX_HOLD_GAP = 0.6

FONT_NAME = "Noto Serif"
WHITE = "&H00FFFFFF&"
GOLD = "&H0000E1FF&"
GLOW = "&H0019C8FF&"


@dataclass
class Word:
    text: str
    start: float
    end: float


def tokenize(narration: str) -> list[str]:
    return [token for token in re.split(r"\s+", narration or "") if re.search(r"\w", token)]


def _pause_after(token: str) -> float:
    tail = token.rstrip("\"'”’)»")
    return PAUSE_WEIGHTS.get(tail[-1:], 0.0) if tail else 0.0


def time_words(tokens: list[str], start: float, end: float) -> list[Word]:
    """Spread ``tokens`` over ``[start, end]`` by syllable, with pauses after punctuation."""
    if not tokens:
        return []
    weights = [1.0 + _pause_after(token) for token in tokens]
    weights[-1] = 1.0
    unit = max(end - start, 0.01) / sum(weights)
    words = []
    cursor = start
    for token, weight in zip(tokens, weights):
        words.append(Word(token, cursor, cursor + unit))
        cursor += unit * weight
    return words


def display_text(token: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", token).upper()


def chunk_words(words: list[Word]) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        text = " ".join(display_text(w.text) for w in [*current, word])
        if current and (len(current) >= MAX_WORDS_PER_CHUNK or len(text) > MAX_CHARS_PER_CHUNK):
            chunks.append(current)
            current = []
        current.append(word)
        if _pause_after(word.text):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _ass_time(seconds: float) -> str:
    centis = max(0, round(seconds * 100))
    hours, rest = divmod(centis, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _chunk_events(chunk: list[Word], chunk_end: float, y: int) -> list[str]:
    labels = [_escape(display_text(word.text)) for word in chunk]
    events = []
    glow_text = " ".join(labels)
    events.append(
        f"Dialogue: 0,{_ass_time(chunk[0].start)},{_ass_time(chunk_end)},Glow,,0,0,0,,"
        f"{{\\pos(540,{y})\\blur20\\fad(80,80)}}{glow_text}"
    )
    for index, word in enumerate(chunk):
        start = word.start if index else chunk[0].start
        end = chunk[index + 1].start if index + 1 < len(chunk) else chunk_end
        parts = [
            f"{{\\1c{GOLD}}}{label}{{\\1c{WHITE}}}" if i == index else label
            for i, label in enumerate(labels)
        ]
        events.append(
            f"Dialogue: 1,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,"
            f"{{\\pos(540,{y})}}{' '.join(parts)}"
        )
    return events


def build_ass(scene_words: list[tuple[list[Word], int]], font_size: int = 70) -> str:
    """``scene_words`` is a list of (timed words, caption y) per scene."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Glow,{FONT_NAME},{font_size},&HFF000000&,&HFF000000&,{GLOW},&HFF000000&,-1,0,0,0,100,100,2,0,1,12,0,5,40,40,0,1
Style: Caption,{FONT_NAME},{font_size},{WHITE},{WHITE},&H00206A9A&,&H00000000&,-1,0,0,0,100,100,2,0,1,1.5,0,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events: list[str] = []
    all_chunks: list[tuple[list[Word], int]] = []
    for words, y in scene_words:
        all_chunks.extend((chunk, y) for chunk in chunk_words(words))
    for index, (chunk, y) in enumerate(all_chunks):
        chunk_end = chunk[-1].end
        if index + 1 < len(all_chunks):
            next_start = all_chunks[index + 1][0][0].start
            if next_start - chunk_end <= MAX_HOLD_GAP:
                chunk_end = next_start
        events.extend(_chunk_events(chunk, chunk_end, y))
    return header + "\n".join(events) + "\n"
