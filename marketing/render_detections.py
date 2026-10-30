"""Render bundled raw detections, not a newly solved or fitted animation."""
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'examples/dragon_roar/det.npz'
rig_path = ROOT / 'assets/dragon/rigged/rig_info.json'
data = np.load(source)
rig = json.loads(rig_path.read_text())
names = list(data['names'])
frame = 48
image = Image.new('RGB', (1600, 1000), '#0d141e')
draw = ImageDraw.Draw(image)
font_path = 'C:/Windows/Fonts/segoeui.ttf'
font = lambda size: ImageFont.truetype(font_path, size)
draw.text((60,35), 'anymove / bundled dragon detections', font=font(38), fill='white')
for x, key, label in zip([80,880], ['front_uv', 'side_uv'], ['Front view', 'Side view']):
    uv = data[key][frame]
    confidence = data[key.replace('_uv', '_conf')][frame]
    draw.rounded_rectangle((x-20,120,x+660,860), 15, fill='#121e2c', outline='#33455b', width=2)
    draw.text((x,90),label,font=font(24),fill='white')
    for bone in rig['bones']:
        if bone['head_joint'] in names and bone['tail_joint'] in names:
            a, b = [names.index(bone[k]) for k in ['head_joint', 'tail_joint']]
            draw.line([(x+float(uv[a,0]),130+float(uv[a,1])),(x+float(uv[b,0]),130+float(uv[b,1]))],fill='#91baff',width=3)
    for point, conf in zip(uv, confidence):
        px,py=x+float(point[0]),130+float(point[1])
        color=(255, int(90+165*float(np.clip(conf,0,1))), 70)
        draw.ellipse((px-4,py-4,px+4,py+4),fill=color)
draw.text((80,900),'Frame 48 of 124 · raw reader output · not a fitted animation',font=font(25),fill='#bbc9dc')
draw.text((80,945),'Stonefletcher fork · upstream: Patch.gg / CanItCode method',font=font(20),fill='#93a5bc')
out = ROOT / 'marketing/refresh'
out.mkdir(parents=True, exist_ok=True)
image.save(out/'detections.png')
(out/'detections-receipt.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)), 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(), 'rig_sha256':hashlib.sha256(rig_path.read_bytes()).hexdigest(), 'frame':frame, 'frames':len(data['front_uv']), 'joints':len(names), 'kind':'bundled raw detections; no model generation, fitting, training or paid API calls'}, indent=2)+'\n')
