#This assumes there is a directory called truthset
for x in {1..1000}
do
    convert -size 16x16 xc:   +noise Random   "truthset/big-$x.png"
    convert "truthset/big-$x.png" -resize 8X8 "truthset/small-$x.png"
done
