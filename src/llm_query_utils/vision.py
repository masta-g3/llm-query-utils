from typing import Optional


def format_vision_messages(
    images: list[str],
    text: str,
    model: str,
    system_message: Optional[str] = None,
) -> list[dict]:
    """Format vision messages based on model provider."""
    if "claude" in model.lower():
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        f"data:image/png;base64,{img}"
                        if not img.startswith("http")
                        else img
                    )
                },
            }
            for img in images
        ] + [{"type": "text", "text": text}]
    else:
        content = [{"type": "text", "text": text}] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        img
                        if img.startswith("http")
                        else f"data:image/png;base64,{img}"
                    )
                },
            }
            for img in images
        ]

    messages = [{"role": "user", "content": content}]
    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})

    return messages
