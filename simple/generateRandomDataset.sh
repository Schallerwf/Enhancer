mkdir dataset

for x in {1..10000}
do
    convert -size 4x4 xc:   +noise Random   "dataset/big-$x.png"
    convert "dataset/big-$x.png" -resize 2X2 "dataset/small-$x.png"
done
