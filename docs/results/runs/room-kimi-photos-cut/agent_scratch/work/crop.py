import sys
from PIL import Image
src, out, x, y, w, h, scale = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
im = Image.open(src).crop((x, y, x + w, y + h))
im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
im.save(out)
print(out, im.size)
