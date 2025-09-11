# server/app.py

import os
import sys
import hmac
import hashlib
import io
import pathlib
import logging
import re
from typing import List, Optional, Tuple

import requests
from flask import Flask, send_from_directory, request, jsonify
from dotenv import load_dotenv
from PIL import Image
from google.api_core.exceptions import ResourceExhausted
import google.generativeai as genai
import razorpay

# Optional: Replicate for AI image edits
try:
    import replicate
except ImportError:
    replicate = None

# ----------------------------
# 1) Init & configuration
# ----------------------------
load_dotenv()

GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY")
RAZORPAY_KEY_ID     = os.getenv("RAZORPAY_KEY_ID")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET")

# Prices (INR)
PRICE_INR = {
    "caption": 5,
    "image": 10,  # photo + edits tier
}

# Replicate edit models (primary + fallback)
REPLICATE_API_TOKEN         = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_IMG_EDIT_MODEL    = os.getenv("REPLICATE_IMG_EDIT_MODEL", "google/nano-banana")
REPLICATE_IMG_EDIT_FALLBACK = os.getenv("REPLICATE_IMG_EDIT_FALLBACK", "black-forest-labs/flux-redux-schnell")

if not GOOGLE_API_KEY:
    print("[BOOT] WARNING: GOOGLE_API_KEY not set", file=sys.stderr)
if not RAZORPAY_KEY_ID or not RAZORPAY_KEY_SECRET:
    print("[BOOT] WARNING: Razorpay keys not set", file=sys.stderr)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("caption-ai-app")

genai.configure(api_key=GOOGLE_API_KEY)

PUBLIC_DIR  = pathlib.Path(__file__).resolve().parents[1] / "public"
OUTPUT_DIR  = PUBLIC_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder=str(PUBLIC_DIR), static_url_path='')

razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# ----------------------------
# 2) Static home
# ----------------------------
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# ----------------------------
# 3) Frontend config (Key ID + prices)
# ----------------------------
@app.get('/config')
def get_config():
    return jsonify({
        "razorpay_key_id": RAZORPAY_KEY_ID,
        "prices": PRICE_INR,  # expose prices for the UI
    })

# ----------------------------
# 4) Create order (tier-aware & price-validated)
# ----------------------------
@app.post('/create-order')
def create_order():
    try:
        data = request.get_json(force=True, silent=True) or {}

        tier = (data.get('tier') or '').strip().lower()
        client_amount = int(data.get('amount', 0) or 0)

        if tier in PRICE_INR:
            amount_in_inr = PRICE_INR[tier]
        else:
            # Backward-compat: allow amount-only calls if it matches a known price
            if client_amount in PRICE_INR.values():
                amount_in_inr = client_amount
                tier = next((k for k, v in PRICE_INR.items() if v == client_amount), "unknown")
            else:
                return jsonify({"error": "Invalid tier/amount"}), 400

        if amount_in_inr <= 0:
            return jsonify({"error": "Invalid amount"}), 400

        order = razorpay_client.order.create({
            "amount": amount_in_inr * 100,  # paise
            "currency": "INR",
            "payment_capture": 1,
            "notes": {"tier": f"{tier}_{amount_in_inr}_rs"}
        })
        return jsonify({"id": order["id"], "amount": order["amount"], "currency": order["currency"]})
    except Exception:
        app.logger.exception("Error creating order")
        return jsonify({"error": "Could not create order"}), 500

# ----------------------------
# 5) Verify payment signature
# ----------------------------
@app.post('/verify-payment')
def verify_payment():
    try:
        body = request.get_json(force=True, silent=True) or {}
        rp_payment_id = body.get("razorpay_payment_id")
        rp_order_id   = body.get("razorpay_order_id")
        rp_signature  = body.get("razorpay_signature")

        if not (rp_payment_id and rp_order_id and rp_signature):
            return jsonify({"error": "Missing verification fields"}), 400

        payload = f"{rp_order_id}|{rp_payment_id}"
        generated_signature = hmac.new(
            (RAZORPAY_KEY_SECRET or "").encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(generated_signature, rp_signature):
            app.logger.warning("Payment verification failed: Signature mismatch", extra={
                "order_id": rp_order_id, "payment_id": rp_payment_id, "payload_repr": repr(payload)
            })
            return jsonify({"error": "Signature mismatch"}), 400

        return jsonify({"status": "ok"})
    except Exception:
        app.logger.exception("Payment verification error")
        return jsonify({"error": "Verification error"}), 500

# ----------------------------
# 6) Generate content (JSON or multipart)
# ----------------------------
@app.post('/generate-content')
def generate_content():
    try:
        # ----- Image + edits (multipart/form-data) -----
        if 'image' in request.files:
            image_file   = request.files['image']
            description  = (request.form.get('description') or '').strip()
            edits        = (request.form.get('edits') or '').strip()
            platform     = (request.form.get('platform') or 'instagram').strip()
            language     = (request.form.get('language') or 'English').strip()
            style_prompt = (request.form.get('style_prompt') or '').strip()

            if not image_file or image_file.filename == '':
                return jsonify({"error": "Image is required"}), 400

            # Validate image
            try:
                pil = Image.open(image_file.stream).convert("RGB")
            except Exception:
                return jsonify({"error": "Invalid image"}), 400

            # Re-encode to JPEG bytes for captioning context
            jpg_bytes = _downscale_to_jpeg_bytes(pil, max_side=1024, quality=85)

            caption_prompt = _build_caption_prompt(
                description, edits, platform, language, style_prompt, with_image=True
            )

            text_model = _get_model()
            try:
                cap_resp = text_model.generate_content(
                    [caption_prompt, {"mime_type": "image/jpeg", "data": jpg_bytes}]
                )
                captions_raw = _extract_bullets(cap_resp.text)
            except ResourceExhausted:
                # Fallback: text-only if multimodal hits quota
                fallback_prompt = _build_caption_prompt(
                    description, edits, platform, language, style_prompt, with_image=False
                )
                cap_resp = text_model.generate_content(fallback_prompt)
                captions_raw = _extract_bullets(cap_resp.text)

            captions = _postprocess_captions(
                captions_raw, platform, description, language
            )

            # >>> AI IMAGE EDITS (Replicate) <<<
            images_urls: List[str] = []
            notice: Optional[str] = None
            if REPLICATE_API_TOKEN and replicate is not None:
                try:
                    images_urls = _run_img_edit_with_replicate_from_pil(pil, edits or description)
                    images_urls = [_persist_from_url(u) for u in images_urls]
                except Exception:
                    app.logger.exception("Replicate edit failed")
                    notice = "AI image edits temporarily unavailable; showing captions only."
            else:
                notice = "Image edits not configured. Set REPLICATE_API_TOKEN to enable AI edits."

            payload = {"status": "success", "captions": captions, "images": images_urls}
            if notice:
                payload["notice"] = notice
            return jsonify(payload)

        # ----- Caption-only (application/json) -----
        else:
            data         = request.get_json(force=True, silent=True) or {}
            description  = (data.get('description') or '').strip()
            platform     = (data.get('platform') or 'instagram').strip()
            language     = (data.get('language') or 'English').strip()
            style_prompt = (data.get('style_prompt') or '').strip()

            if not description:
                return jsonify({"error": "Description is required"}), 400

            prompt = _build_caption_prompt(
                description, "", platform, language, style_prompt, with_image=False
            )

            model = _get_model()
            resp = model.generate_content(prompt)
            captions_raw = _extract_bullets(resp.text)
            captions = _postprocess_captions(
                captions_raw, platform, description, language
            )
            return jsonify({"status": "success", "captions": captions, "images": []})

    except Exception:
        app.logger.exception("Error generating content")
        return jsonify({"error": "An internal error occurred"}), 500

# ----------------------------
# 7) Helpers
# ----------------------------
def _get_model():
    """Try a few modern fast models, then fall back."""
    for mid in ('gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-1.5-flash-latest'):
        try:
            logger.info(f"[GENAI] Using model: {mid}")
            return genai.GenerativeModel(mid)
        except Exception:
            continue
    logger.info("[GENAI] Falling back to gemini-1.5-flash-latest")
    return genai.GenerativeModel('gemini-1.5-flash-latest')


def _extract_bullets(text: str) -> List[str]:
    """Extract up to 3 lines, stripping leading bullets/nums/quotes."""
    if not text:
        return []
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    clean = []
    for ln in lines:
        ln = re.sub(r'^\s*[\d\.\)\-\*\•\>]+\s*', '', ln)  # strip common bullet markers
        ln = ln.strip('“”"\' ')  # clean surrounding quotes
        if ln:
            clean.append(ln)
        if len(clean) == 3:
            break
    return clean


def _downscale_to_jpeg_bytes(pil_img: Image.Image, max_side: int = 1024, quality: int = 85) -> bytes:
    img = pil_img.convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

# ---------- Platform style presets ----------
def _platform_style(platform: str):
    p = (platform or '').strip().lower()

    # Sensible short-form defaults to keep captions scannable
    base = {
        "name": platform or "Instagram",
        "max_len": 220,  # short, high-retention captions by default
        "bullets": 3,
        "tone": "conversational, specific, benefit-led",
        "emoji_rule": "0–2 relevant emojis; never spam.",
        "hashtag_rule": "0–3 niche, descriptive hashtags only.",
        "cta_rule": "Soft, value-focused CTA.",
        "rules": [
            "Hook in the first ~6 words.",
            "Specific > vague; concrete nouns/verbs.",
            "Avoid clickbait, clichés, and generic hashtags.",
            "At least one caption must include a question."
        ],
    }

    presets = {
        "instagram": dict(base, name="Instagram", max_len=220, tone="engaging, sensory, authentic"),
        "x": dict(base, name="X (Twitter)", max_len=280, tone="witty, sharp, to-the-point",
                  emoji_rule="≤1 emoji", hashtag_rule="≤2 if truly useful"),
        "linkedin": dict(base, name="LinkedIn", max_len=300, tone="professional, insightful, credible",
                         emoji_rule="optional ≤1", hashtag_rule="0–2 industry tags",
                         cta_rule="Invite discussion with a question."),
        "facebook": dict(base, name="Facebook", max_len=300, tone="friendly, community-first"),
        "tiktok": dict(base, name="TikTok", max_len=150, tone="fun, punchy, high-energy"),
        "youtube": dict(base, name="YouTube Shorts", max_len=120, tone="clear, curiosity-driven"),
        "pinterest": dict(base, name="Pinterest", max_len=200, tone="helpful, aspirational, how-to"),
        "threads": dict(base, name="Threads", max_len=280, tone="chill, conversational, witty"),
        "reddit": dict(base, name="Reddit", max_len=300, tone="useful, specific, non-promotional",
                       emoji_rule="avoid", hashtag_rule="none", cta_rule="Ask for experiences"),
    }

    aliases = {
        "ig": "instagram", "twitter": "x", "li": "linkedin",
        "fb": "facebook", "youtube shorts": "youtube", "shorts": "youtube"
    }

    key = aliases.get(p, p)
    return presets.get(key) or base


# ---------- Prompt builder ----------
def _build_caption_prompt(
    description: str,
    edits: str,
    platform: str,
    language: str,
    style_prompt: str,
    with_image: bool
):
    st = _platform_style(platform)
    context_line = (
        "The user has provided an image for context."
        if with_image else
        "No image provided; rely on the description."
    )

    # Clear, testable instructions; forces diversity and compliance
    return f"""
You are CaptionCraft AI, an elite social media strategist and copywriter.

CONTEXT: {context_line}
PLATFORM: {st['name']}
LANGUAGE: {language}

USER INPUT:
- Core description: "{description}"
- Visual notes/edits (if any): "{edits or ''}"
- Style directive (if any): "{style_prompt or ''}"

REQUIREMENTS:
- Generate EXACTLY {st['bullets']} distinct caption options.
- Each caption MUST be under {st['max_len']} characters (strict).
- Tone: {st['tone']}.
- Emojis: {st['emoji_rule']}.
- Hashtags: {st['hashtag_rule']}.
- CTA: {st['cta_rule']}.
- Additional rules:
  - {st['rules'][0]}
  - {st['rules'][1]}
  - {st['rules'][2]}
  - {st['rules'][3]}

DIVERSITY:
- Provide 3 different styles:
  1) Direct & value-first.
  2) Creative/emotional with vivid detail.
  3) Conversational and includes an engaging question.

FORMAT:
- Return ONLY the {st['bullets']} final captions.
- One caption per line.
- No numbering, bullets, labels, or commentary.
""".strip()


# ---------- Post-processing & utilities ----------
def _postprocess_captions(
    captions_raw: List[str],
    platform: str,
    description: str,
    language: str
) -> List[str]:
    """Trim, dedupe, ensure one question, and optionally add smart hashtags within limit."""
    st = _platform_style(platform)
    max_len = st["max_len"]

    # Clean empty lines
    captions = [c.strip() for c in (captions_raw or []) if c and c.strip()]

    # Ensure up to 3 candidates
    captions = captions[:3]
    if not captions:
        captions = _fallback_captions(description, platform, language)

    # Deduplicate (case-insensitive)
    captions = _dedupe(captions)

    # Ensure at least one caption ends with a question if required
    captions = _ensure_question(captions, description, language)

    # Optionally compose 0–3 niche hashtags from description keywords
    hashtags = _compose_hashtags(description, platform, max_tags=3)
    processed = []
    for i, c in enumerate(captions):
        # For variety, add hashtags only to one caption (typically the first),
        # and only if platform allows hashtags
        with_tags = c
        if i == 0 and hashtags and _platform_allows_hashtags(platform):
            trial = f"{c} {' '.join(hashtags)}".strip()
            with_tags = _trim_to_limit(trial, max_len)
        else:
            with_tags = _trim_to_limit(c, max_len)

        processed.append(with_tags)

    # Final safety: enforce length again and re-dedupe
    processed = [_trim_to_limit(x, max_len) for x in processed]
    processed = _dedupe(processed)

    # Guarantee exactly 3 lines (pad with short variants if needed)
    while len(processed) < 3:
        filler = _trim_to_limit(_quick_variant(description, platform, language), max_len)
        if filler not in processed:
            processed.append(filler)
        else:
            processed.append(_trim_to_limit(f"{description[: max(10, len(description)//2)]}?", max_len))

    return processed[:3]


def _platform_allows_hashtags(platform: str) -> bool:
    key = (platform or "").strip().lower()
    if key in {"reddit"}:
        return False
    return True


def _trim_to_limit(text: str, max_len: int) -> str:
    """Trim on word boundary; add ellipsis if truncated in the middle of a word."""
    text = (text or "").strip()
    if len(text) <= max_len:
        return text
    # Try last space within limit
    cut = text.rfind(" ", 0, max_len)
    if cut != -1 and cut >= max_len - 30:
        return text[:cut].rstrip() + "…"
    # Hard cut with ellipsis
    if max_len >= 1:
        return text[: max_len - 1].rstrip() + "…"
    return ""


def _dedupe(lines: List[str]) -> List[str]:
    seen = set()
    out = []
    for ln in lines:
        key = ln.lower().strip()
        if key and key not in seen:
            out.append(ln)
            seen.add(key)
    return out


def _ensure_question(captions: List[str], description: str, language: str) -> List[str]:
    if any(c.strip().endswith("?") or "?" in c for c in captions):
        return captions
    # Add a conversational question to the last caption
    question = _question_for(description, language)
    if captions:
        base = captions[-1].rstrip("?.! ")
        captions[-1] = f"{base}? {question}".strip()
        return captions
    return [f"{_quick_variant(description, 'instagram', language)} {question}".strip()]


def _question_for(description: str, language: str) -> str:
    # Simple multilingual-friendly question add-on (kept short)
    if (language or "").lower().startswith("hi"):
        return "आप क्या सोचते हैं?"
    if (language or "").lower().startswith("te"):
        return "మీ అభిప్రాయం ఏమిటి?"
    if (language or "").lower().startswith("es"):
        return "¿Qué opinas?"
    return "What do you think?"


def _compose_hashtags(description: str, platform: str, max_tags: int = 3) -> List[str]:
    """Create 0–3 niche, descriptive hashtags from description keywords."""
    if not _platform_allows_hashtags(platform):
        return []
    # Extract simple keywords
    words = re.findall(r"[A-Za-z0-9]+", description.lower())
    # Filter generic stopwords
    stops = {"the","a","an","and","or","to","of","in","on","for","with","this","that","is","are","it","at","by","be"}
    kws = [w for w in words if w not in stops and len(w) > 2]
    # Keep unique order
    seen = set()
    kws_unique = []
    for w in kws:
        if w not in seen:
            kws_unique.append(w); seen.add(w)
        if len(kws_unique) >= 6:
            break
    hashtags = [f"#{re.sub(r'[^A-Za-z0-9]', '', w)}" for w in kws_unique[:max_tags]]
    # Avoid empty or numeric-only tags
    hashtags = [ht for ht in hashtags if re.search(r"[A-Za-z]", ht)]
    return hashtags[:max_tags]


def _quick_variant(description: str, platform: str, language: str) -> str:
    # Tiny heuristic alternative if model returns too few lines
    base = description.strip() or "Making something awesome"
    hooks = [
        "Quick tip:", "Behind the shot:", "Real talk:", "Pro move:",
        "Did you know?", "Try this next:", "Hot take:"
    ]
    hook = hooks[hash(base) % len(hooks)]
    return f"{hook} {base}"


def _fallback_captions(description: str, platform: str, language: str) -> List[str]:
    # Graceful fallback; short, safe, varied
    v1 = f"{description[:80].strip()} — save this for later!"
    v2 = f"From idea to reality: {description[:70].strip()}"
    v3 = f"{description[:60].strip()} — thoughts?"
    return [v1, v2, v3]


# ---------- Replicate helpers ----------
def _persist_bytes_to_temp(b: bytes) -> str:
    """Persist raw bytes to a short-lived temp host and return a URL."""
    try:
        files = {"file": ("out.png", b)}
        r = requests.post("https://temp.sh/upload", files=files, timeout=60)
        r.raise_for_status()
        u = r.text.strip()
        return u if u.startswith("http") else ""
    except Exception:
        app.logger.exception("Failed to persist bytes")
        return ""


def _normalize_replicate_outputs(out) -> List[str]:
    """
    Normalize possible Replicate outputs (URLs/bytes/file-like/objects with .url) into URL strings.
    """
    if out is None:
        return []
    if not isinstance(out, (list, tuple)):
        out = [out]

    urls: List[str] = []

    for o in out:
        if isinstance(o, str) and o.startswith("http"):
            urls.append(o); continue

        u = getattr(o, "url", None)
        if isinstance(u, str) and u.startswith("http"):
            urls.append(u); continue

        reader = getattr(o, "read", None)
        if callable(reader):
            try:
                data = reader()
                if isinstance(data, (bytes, bytearray)) and data:
                    u2 = _persist_bytes_to_temp(bytes(data))
                    if u2: urls.append(u2); continue
            except Exception:
                app.logger.exception("Failed reading file-like output")

        if isinstance(o, (bytes, bytearray)):
            u3 = _persist_bytes_to_temp(bytes(o))
            if u3: urls.append(u3); continue

    return urls


def _run_img_edit_with_replicate_from_pil(pil_img, prompt: str) -> List[str]:
    """
    Run Replicate using an in-memory file (no pre-upload).
    Returns a list of HTTPS URLs for edited images.
    """
    if replicate is None or not REPLICATE_API_TOKEN:
        raise RuntimeError("Replicate not configured. Set REPLICATE_API_TOKEN and install the 'replicate' package.")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    buf.name = "source.png"

    # Primary: google/nano-banana
    try:
        out = replicate.run(
            REPLICATE_IMG_EDIT_MODEL,
            input={
                "prompt": prompt or "",
                "image_input": [buf],   # MUST be a list
                "num_outputs": 2,
            },
        )
        urls = _normalize_replicate_outputs(out)
        if urls:
            return urls
    except Exception as e:
        app.logger.warning(f"Primary model failed ({REPLICATE_IMG_EDIT_MODEL}): {e}. Trying fallback...")

    # Fallback: flux-redux-schnell (expects redux_image)
    try:
        if not REPLICATE_IMG_EDIT_FALLBACK:
            return []
        buf.seek(0)
        out = replicate.run(
            REPLICATE_IMG_EDIT_FALLBACK,
            input={
                "prompt": prompt or "",
                "redux_image": buf,
                "strength": 0.6,
                "num_outputs": 2,
            },
        )
        return _normalize_replicate_outputs(out)
    except Exception as e:
        app.logger.warning(f"Fallback model failed ({REPLICATE_IMG_EDIT_FALLBACK}): {e}")
        return []


def _persist_from_url(url: str) -> str:
    """
    (Optional) Re-host output image bytes to a temp file host for short-lived stable links.
    If this fails, return the original URL.
    """
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        image_bytes = r.content
        files = {'file': ('edited_image.png', image_bytes)}
        upload_response = requests.post('https://temp.sh/upload', files=files, timeout=60)
        upload_response.raise_for_status()
        temp_url = upload_response.text.strip()
        if temp_url.startswith("http"):
            return temp_url
        return url
    except Exception:
        app.logger.exception(f"Failed to re-upload image from URL {url}")
        return url


# ----------------------------
# 8) Entry
# ----------------------------
if __name__ == '__main__':
    print(f"[BOOT] Using Razorpay key: {repr((RAZORPAY_KEY_ID or '')[:12])}…", file=sys.stderr)
    print(f"[BOOT] GOOGLE_API_KEY set: {bool(GOOGLE_API_KEY)}", file=sys.stderr)
    if REPLICATE_API_TOKEN:
        print("[BOOT] Replicate enabled for image edits.", file=sys.stderr)
        print(f"[BOOT] Replicate edit model: {REPLICATE_IMG_EDIT_MODEL} (fallback: {REPLICATE_IMG_EDIT_FALLBACK})", file=sys.stderr)
    else:
        print("[BOOT] Replicate not configured. Set REPLICATE_API_TOKEN to enable AI image edits.", file=sys.stderr)

    print("WARNING: This is a development server. Use a production WSGI server in production.", file=sys.stderr)
    app.run(debug=True, port=5000)
