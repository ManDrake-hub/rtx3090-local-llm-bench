import av, sys, numpy as np
from PIL import Image
f, out = sys.argv[1], sys.argv[2]
c = av.open(f)
vs = c.streams.video[0]
info = [f"{vs.codec_context.width}x{vs.codec_context.height} {float(vs.average_rate):.0f}fps {float(c.duration/1e6):.2f}s"]
frames = [fr.to_image() for fr in c.decode(video=0)]
idx = np.linspace(0, len(frames)-1, 6).astype(int)
th = [frames[i].resize((480, int(480*frames[i].height/frames[i].width))) for i in idx]
W, H = th[0].size
sheet = Image.new("RGB", (W*3, H*2))
for k, im in enumerate(th): sheet.paste(im, ((k % 3)*W, (k//3)*H))
sheet.save(out, quality=85)
c.close(); c = av.open(f)
if c.streams.audio:
    a = np.concatenate([fr.to_ndarray().reshape(-1) for fr in c.decode(audio=0)]).astype(np.float32)
    rms = np.sqrt(np.mean(a**2)); info.append(f"audio rms={20*np.log10(rms+1e-9):.1f}dBFS peak={20*np.log10(np.abs(a).max()+1e-9):.1f}dBFS")
else: info.append("no audio")
print(len(frames), "frames", *info)
