#!/bin/bash
input=${input:="default"}
git add *
git commit -m input
git push origin main