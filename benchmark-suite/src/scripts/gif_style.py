from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont


NAVY = '#0c2949'
BACKGROUND = '#eef3f9'
CYAN = '#17c8e8'


@lru_cache(maxsize=24)
def font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/' + name, size)


def centered(draw, xy, text, size, fill=NAVY, bold=False):
    draw.text(xy, text, font=font(size, bold), fill=fill, anchor='mt')


def comparison_frame(images, labels, errors, terminated, task, checkpoint, payload, delay, time):
    width, card_width, gap = 1488, 480, 12
    canvas = Image.new('RGB', (width, 610), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    title = 'Reaching controller comparison' if task == 'reach' else 'Tracking controller comparison'
    centered(draw, (width // 2, 12), title, 34, bold=True)
    subtitle = f'{checkpoint.capitalize()} checkpoint • {task.capitalize()} • Payload: +{payload:g} kg • Delay: {delay} ms • Time: {time:.2f} s'
    centered(draw, (width // 2, 58), subtitle, 19, '#405570')
    for i, (image, label, error, stopped) in enumerate(zip(images, labels, errors, terminated)):
        x = gap + i * (card_width + gap)
        draw.rounded_rectangle((x, 100, x + card_width, 598), radius=12, fill='#fbfdff')
        draw.rounded_rectangle((x, 100, x + card_width, 150), radius=12, fill='#193450')
        draw.rectangle((x, 126, x + card_width, 150), fill='#193450')
        centered(draw, (x + card_width // 2, 109), label, 25, 'white', True)
        canvas.paste(image.resize((480, 330), Image.Resampling.LANCZOS), (x, 150))
        draw.line((x + 28, 508, x + 138, 508), fill=CYAN, width=3)
        draw.line((x + 342, 508, x + 452, 508), fill=CYAN, width=3)
        centered(draw, (x + 240, 495), 'Position error', 19)
        color = '#d48a00' if error >= 20 else NAVY
        centered(draw, (x + 240, 526), f'{error:.1f} mm', 40, color, True)
        if stopped:
            centered(draw, (x + 240, 576), 'Episode terminated', 13, '#b34338')
    return canvas
