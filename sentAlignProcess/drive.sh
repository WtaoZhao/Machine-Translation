#!/bin/bash
cnt=0
nums=1000
echo "Begin $nums-times Champollion!"
while [ $cnt -lt $nums ]
do
    echo "**************************************************"
    echo "processing file $cnt"
    /home/lzh/champollion-1.2/bin/champollion.EC_utf8 train.$cnt.en train.$cnt.zh train.$cnt.align
    let cnt+=1
done
echo "Done!"
