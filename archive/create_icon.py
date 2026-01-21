#!/usr/bin/env python3
"""
Create an app icon for Mini Quant Fund.
Uses PIL to generate a professional-looking icon.
"""

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("PIL not installed. Installing...")
    import subprocess
    subprocess.run(["pip", "install", "Pillow"])
    from PIL import Image, ImageDraw, ImageFont

import os

def create_icon():
    # Icon size (1024x1024 for high resolution)
    size = 1024
    
    # Create image with gradient background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw rounded rectangle background with gradient effect
    margin = 50
    
    # Dark blue to purple gradient (simulated with multiple rectangles)
    colors = [
        (15, 23, 42),    # Dark navy
        (30, 41, 59),    # Slate
        (51, 65, 85),    # Lighter slate
    ]
    
    # Draw background circle
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill=(15, 23, 42)
    )
    
    # Draw inner gradient rings
    for i, color in enumerate(colors[1:]):
        offset = margin + (i + 1) * 80
        draw.ellipse(
            [offset, offset, size - offset, size - offset],
            fill=color,
            outline=None
        )
    
    # Draw chart-like elements (candlestick pattern)
    chart_colors = {
        'green': (34, 197, 94),   # Green for up
        'red': (239, 68, 68),     # Red for down
        'blue': (59, 130, 246),   # Blue accent
        'gold': (251, 191, 36),   # Gold accent
    }
    
    # Draw stylized chart bars
    bar_width = 60
    bar_gap = 40
    chart_left = 280
    chart_bottom = 700
    
    bars = [
        (180, 'green'),
        (250, 'green'),
        (150, 'red'),
        (320, 'green'),
        (280, 'green'),
        (200, 'red'),
        (350, 'green'),
    ]
    
    for i, (height, color) in enumerate(bars):
        x = chart_left + i * (bar_width + bar_gap)
        y_top = chart_bottom - height
        
        # Draw bar
        draw.rectangle(
            [x, y_top, x + bar_width, chart_bottom],
            fill=chart_colors[color],
            outline=None
        )
        
        # Draw wick for candlestick effect
        wick_x = x + bar_width // 2
        if color == 'green':
            draw.line(
                [(wick_x, y_top - 30), (wick_x, y_top)],
                fill=chart_colors[color],
                width=8
            )
        else:
            draw.line(
                [(wick_x, chart_bottom), (wick_x, chart_bottom + 30)],
                fill=chart_colors[color],
                width=8
            )
    
    # Draw trend line
    points = []
    for i, (height, _) in enumerate(bars):
        x = chart_left + i * (bar_width + bar_gap) + bar_width // 2
        y = chart_bottom - height - 50
        points.append((x, y))
    
    # Smooth trend line
    for i in range(len(points) - 1):
        draw.line(
            [points[i], points[i + 1]],
            fill=chart_colors['gold'],
            width=12
        )
    
    # Draw dots on trend line
    for point in points:
        draw.ellipse(
            [point[0] - 15, point[1] - 15, point[0] + 15, point[1] + 15],
            fill=chart_colors['gold']
        )
    
    # Draw "MQ" text
    try:
        # Try to use a nice font
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 200)
    except:
        font = ImageFont.load_default()
    
    # Draw text shadow
    text = "MQ"
    text_x = 380
    text_y = 180
    draw.text((text_x + 5, text_y + 5), text, fill=(0, 0, 0, 100), font=font)
    draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
    # Draw "FUND" subtitle
    try:
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 80)
    except:
        small_font = ImageFont.load_default()
    
    draw.text((400, 750), "FUND", fill=(148, 163, 184), font=small_font)
    
    # Draw decorative elements - small dots
    for _ in range(20):
        import random
        x = random.randint(margin + 50, size - margin - 50)
        y = random.randint(margin + 50, size - margin - 50)
        r = random.randint(3, 8)
        alpha = random.randint(50, 150)
        draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=(255, 255, 255, alpha)
        )
    
    # Save as PNG
    output_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(output_dir, "MiniQuantFund.app/Contents/Resources/AppIcon.png")
    
    # Create directory if needed
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    
    img.save(png_path, "PNG")
    print(f"✅ Icon saved to: {png_path}")
    
    # Also save a copy to Desktop for easy access
    desktop_path = os.path.expanduser("~/Desktop/MiniQuantFund_Icon.png")
    img.save(desktop_path, "PNG")
    print(f"✅ Icon also saved to: {desktop_path}")
    
    # Create iconset for proper macOS icon
    iconset_dir = os.path.join(output_dir, "MiniQuantFund.app/Contents/Resources/AppIcon.iconset")
    os.makedirs(iconset_dir, exist_ok=True)
    
    # Generate different sizes
    sizes = [16, 32, 64, 128, 256, 512, 1024]
    for s in sizes:
        resized = img.resize((s, s), Image.LANCZOS)
        resized.save(os.path.join(iconset_dir, f"icon_{s}x{s}.png"))
        if s <= 512:
            resized_2x = img.resize((s * 2, s * 2), Image.LANCZOS)
            resized_2x.save(os.path.join(iconset_dir, f"icon_{s}x{s}@2x.png"))
    
    print("✅ Iconset created")
    
    # Convert to icns using iconutil
    icns_path = os.path.join(output_dir, "MiniQuantFund.app/Contents/Resources/AppIcon.icns")
    os.system(f'iconutil -c icns "{iconset_dir}" -o "{icns_path}"')
    print(f"✅ ICNS created: {icns_path}")
    
    return png_path

if __name__ == "__main__":
    create_icon()
