# server/app.py

import os
import sys
import hmac
import hashlib
import io
import pathlib
import logging
from typing import List, Optional

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

# Prices (INR) — change here if you update plans later
PRICE_INR = {
    "caption": 5,
    "image": 10,   # ← updated photo tier price
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
# 4) Create order (now tier-aware & price-validated)
# ----------------------------
@app.post('/create-order')
def create_order():
    try:
        data = request.get_json(force=True, silent=True) or {}

        # Optional 'tier' from client; if provided, we ENFORCE server price.
        tier = (data.get('tier') or '').strip().lower()
        client_amount = int(data.get('amount', 0) or 0)

        if tier in PRICE_INR:
            amount_in_inr = PRICE_INR[tier]  # authoritative price
        else:
            # Backward-compat: allow amount-only calls if it matches a known price
            if client_amount in PRICE_INR.values():
                amount_in_inr = client_amount
                # try to deduce tier for notes
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
            image_file  = request.files['image']
            description = (request.form.get('description') or '').strip()
            edits       = (request.form.get('edits') or '').strip()
            platform    = (request.form.get('platform') or 'instagram').strip()
            language    = (request.form.get('language') or 'English').strip()

            if not image_file or image_file.filename == '':
                return jsonify({"error": "Image is required"}), 400

            # Validate image by decoding with PIL; also convert to RGB for safe downstream handling
            try:
                pil = Image.open(image_file.stream).convert("RGB")
            except Exception:
                return jsonify({"error": "Invalid image"}), 400

            # Re-encode to JPEG bytes (downscaled) for captioning context only
            jpg_bytes = _downscale_to_jpeg_bytes(pil, max_side=1024, quality=85)

            # Build prompt + call Gemini
            caption_prompt = _build_caption_prompt(description, edits, platform, language, with_image=True)
            text_model = _get_model()
            try:
                cap_resp = text_model.generate_content(
                    [caption_prompt, {"mime_type": "image/jpeg", "data": jpg_bytes}]
                )
                captions = _extract_bullets(cap_resp.text)
            except ResourceExhausted:
                # fall back to text-only if the multimodal call hits quota
                fallback_prompt = _build_caption_prompt(description, edits, platform, language, with_image=False)
                cap_resp = text_model.generate_content(fallback_prompt)
                captions = _extract_bullets(cap_resp.text)

            captions = _enforce_length_per_platform(captions, platform)

            # >>> AI IMAGE EDITS (Replicate) <<<
            images_urls: List[str] = []
            notice: Optional[str] = None
            if REPLICATE_API_TOKEN and replicate is not None:
                try:
                    images_urls = _run_img_edit_with_replicate_from_pil(pil, edits or description)
                    # Optional: re-host for longer-lived links
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
            data        = request.get_json(force=True, silent=True) or {}
            description = (data.get('description') or '').strip()
            platform    = (data.get('platform') or 'instagram').strip()
            language    = (data.get('language') or 'English').strip()

            if not description:
                return jsonify({"error": "Description is required"}), 400

            prompt = _build_caption_prompt(description, edits="", platform=platform, language=language, with_image=False)

            model = _get_model()
            resp = model.generate_content(prompt)
            captions = _extract_bullets(resp.text)
            captions = _enforce_length_per_platform(captions, platform)
            return jsonify({"status": "success", "captions": captions, "images": []})

    except Exception:
        app.logger.exception("Error generating content")
        return jsonify({"error": "An internal error occurred"}), 500

# ----------------------------
# 7) Helpers
# ----------------------------
def _get_model():
    for mid in ('gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-1.5-flash-latest'):
        try:
            logger.info(f"[GENAI] Using model: {mid}")
            return genai.GenerativeModel(mid)
        except Exception:
            continue
    logger.info("[GENAI] Falling back to gemini-1.5-flash-latest")
    return genai.GenerativeModel('gemini-1.5-flash-latest')


def _extract_bullets(text: str) -> List[str]:
    if not text:
        return []
    lines = [ln.strip("•-* \t\r") for ln in str(text).strip().splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:3] or ["", "", ""]


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

    base = {
        "name": platform or "Instagram",
        "max_len": 150,
        "bullets": 3,
        "tone": "modern, concise, value-first",
        "rules": [
            "Front-load a hook within ~6 words.",
            "Specific > vague; concrete verbs.",
            "Avoid clickbait and generic hashtags.",
            "At least one caption should contain a question."
        ],
        "emoji_rule": "Use emojis sparingly.",
        "hashtag_rule": "Only niche, relevant tags.",
        "cta_rule": "Soft CTA (save/share/comment)."
    }

    presets = {
        "instagram": dict(base, name="Instagram", max_len=150, tone="trendy, playful, high-energy, Gen Z voice",
                          rules=[
                              "Hook in first ~6 words.",
                              "Use 1–3 relevant emojis; no spam.",
                              "Use 0–3 niche hashtags; avoid generic tags.",
                              "At least one caption should include a question."
                          ],
                          emoji_rule="1–3 emojis", hashtag_rule="0–3 niche tags", cta_rule="Soft CTA"),
        "x": dict(base, name="Twitter", max_len=280, tone="witty, sharp, to-the-point",
                  rules=[
                      "Clarity and punch over fluff.",
                      "Use at most 1 emoji; at most 2 hashtags.",
                      "Keep it skimmable; one idea.",
                      "Include a question."
                  ],
                  emoji_rule="≤1 emoji", hashtag_rule="≤2 if useful", cta_rule="Implicit CTA"),
        "linkedin": dict(base, name="LinkedIn", max_len=300, tone="professional, insightful, credible",
                         rules=[
                             "Lead with a concrete insight.",
                             "No slang; plain English.",
                             "0–2 industry hashtags only.",
                             "Invite discussion with a thoughtful question (exactly one caption)."
                         ],
                         emoji_rule="optional ≤1", hashtag_rule="0–2 industry tags", cta_rule="Invite discussion"),
        "facebook": dict(base, name="Facebook", max_len=220, tone="friendly, community-oriented",
                         rules=[
                             "Lead with clear benefit/emotion.",
                             "0–2 tasteful emojis.",
                             "0–2 relevant hashtags.",
                             "Include a question to invite comments."
                         ],
                         emoji_rule="0–2", hashtag_rule="0–2 relevant", cta_rule="Comment/share prompt"),
        "tiktok": dict(base, name="TikTok", max_len=150, tone="fun, casual, high-energy",
                       rules=[
                           "Hook in the first 5 words.",
                           "Use 1–3 emojis.",
                           "Use 1–3 discovery hashtags.",
                           "Include a question."
                       ],
                       emoji_rule="1–3", hashtag_rule="1–3 discovery tags", cta_rule="Watch/try/comment"),
        "youtube": dict(base, name="YouTube Shorts", max_len=150, tone="clear, curiosity-driven",
                        rules=[
                            "Front-load the hook.",
                            "≤2 emojis; 0–2 topic hashtags.",
                            "Include a question."
                        ],
                        emoji_rule="≤2", hashtag_rule="0–2 topic tags", cta_rule="Watch/save"),
        "pinterest": dict(base, name="Pinterest", max_len=200, tone="helpful, aspirational, how-to",
                          rules=[
                              "Lead with outcome or transformation.",
                              "≤2 emojis.",
                              "1–3 keyworded hashtags.",
                              "Question optional."
                          ],
                          emoji_rule="≤2", hashtag_rule="1–3 keyworded", cta_rule="Save/try it"),
        "threads": dict(base, name="Threads", max_len=280, tone="chill, conversational, witty",
                        rules=[
                            "Sound like a human, not a brand.",
                            "≤2 emojis; hashtags optional (≤2).",
                            "Include a question."
                        ],
                        emoji_rule="≤2", hashtag_rule="≤2 optional", cta_rule="Light conversational prompt"),
        "reddit": dict(base, name="Reddit", max_len=300, tone="useful, specific, non-promotional",
                       rules=[
                           "Lead with the concrete takeaway.",
                           "No hashtags. Emojis only if subreddit culture allows.",
                           "Invite experiences/opinions."
                       ],
                       emoji_rule="avoid", hashtag_rule="none", cta_rule="Ask for experiences"),
    }

    aliases = { "ig": "instagram", "twitter": "x", "li": "linkedin",
                "fb": "facebook", "youtube shorts": "youtube", "shorts": "youtube" }

    key = aliases.get(p, p)
    return presets.get(key) or base

# ---------- Prompt builder ----------
def _build_caption_prompt(description: str, edits: str, platform: str, language: str, with_image: bool):
    st = _platform_style(platform)
    context_line = "You are viewing an image." if with_image else "You are writing without seeing the image."
    return f"""
You are a senior social media strategist. {context_line}
Generate exactly {st['bullets']} caption options in {language} for {st['name']}.
Audience: platform-native users who expect {st['tone']}.

Post context (from user): "{description}"
If relevant, image edits or visual notes: "{edits}"

Platform rules:
- Voice/Tone: {st['tone']}
- Hard length limit: {st['max_len']} characters total per caption.
- Emojis: {st.get('emoji_rule','Use emojis sparingly.')}
- Hashtags: {st.get('hashtag_rule','Only niche, relevant tags.')}
- Calls to action: {st.get('cta_rule','Soft CTA (save/share/comment).')}
- Banned: clickbait cliches, spammy tags, culture-insensitive phrasing.

Formatting:
- Return ONLY {st['bullets']} lines. One caption per line. No numbering, no quotes.
- Each line must be <= {st['max_len']} chars (strict).
- At least one caption should include a question.

Quality checklist (follow silently):
- Front-load a hook in the first ~6 words.
- Specific > vague; benefits > features; concrete nouns/verbs.
- Natural rhythm: short clauses, varied sentence length.

Output: the {st['bullets']} captions.
""".strip()


def _enforce_length_per_platform(captions: List[str], platform: str) -> List[str]:
    st = _platform_style(platform)
    max_len = st["max_len"]
    clean = []
    for c in (captions or [])[:3]:
        c = (c or "").strip()
        if len(c) > max_len:
            c = c[:max_len].rstrip()
        clean.append(c)
    while len(clean) < 3:
        clean.append("")
    return clean[:3]

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
