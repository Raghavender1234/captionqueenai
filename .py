from PIL import Image, ImageDraw, ImageFont

def generate_favicon(path="favicon.ico"):
    size = (32, 32)  # standard favicon size
    img = Image.new("RGBA", size, (255, 255, 255, 0))  # transparent background
    draw = ImageDraw.Draw(img)

    # Background (royal purple for "queen" theme)
    draw.rectangle([0, 0, size[0]-1, size[1]-1], fill=(138, 43, 226))  # BlueViolet

    # Try to load a bold font
    try:
        font = ImageFont.truetype("arialbd.ttf", 14)  # Arial Bold, if available
    except:
        font = ImageFont.load_default()

    # Text: "CQ"
    text = "CQ"

    # Get bounding box instead of textsize (Pillow >=10)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center position
    pos = ((size[0] - text_w) // 2, (size[1] - text_h) // 2)

    # Draw text (gold color for queen vibe)
    draw.text(pos, text, fill=(255, 215, 0), font=font)  # Gold

    img.save(path, format="ICO")
    print(f"✅ Saved favicon at {path}")

generate_favicon()
