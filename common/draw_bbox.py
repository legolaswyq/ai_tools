from PIL import Image, ImageDraw

xmin = int(0.26031557 * 320)
ymin = int(0.33608016 * 320)
xmax = int(0.9872614 * 320)
ymax = int(0.98770833 * 320)



img = "/home/walter/Pictures/123_123_123_123_123_123_0.jpg"
img = Image.open(img)
draw = ImageDraw.Draw(img)
draw.rectangle([xmin, ymin, xmax, ymax], outline='red')
img.show()