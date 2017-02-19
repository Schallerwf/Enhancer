from PIL import Image, ImageFilter

size = 100, 100

# Change to a different path depending on your source img folder
files = glob.glob ("lfw-a/lfw/*/*.jpg")

i = 0

# Saves a bunch of resized images to folder 100x100 in current directory
# Saves a bunch of blurry resized images to folder 100x100BLUR in current directory
for infile in files:
    currfile, ext = os.path.splitext(infile)
    im = Image.open(infile)
    im.thumbnail(size)
    im.save("100x100/" + str(i) + "-100.jpg", "JPEG")
    blurim = im.filter(ImageFilter.GaussianBlur(3))
    blurim.save("100x100BLUR/" + str(i) + "-100.jpg", "JPEG")
    i = i + 1