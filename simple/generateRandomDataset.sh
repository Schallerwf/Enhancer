mkdir dataset

for x in {1..1000}
do
    convert -size 16x16 xc:   +noise Random   "dataset/big-$x.png"
    convert "dataset/big-$x.png" -resize 8X8 "dataset/small-$x.png"
done
