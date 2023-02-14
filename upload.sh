#!/bin/bash
input="$1"
input=${input:="default"}
#echo ${input}
git add *
git commit -m $input
git push origin main
