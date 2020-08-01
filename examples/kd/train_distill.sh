#!/bin/sh
# kd
python  train.py  --distill kd -a 0.9 -b 0.1 -r 0.0
# hint
python  train.py --distill hint -a 0.9 -b 0.1 -r 100
# attention
python  train.py  --distill attention -a 1 -b 0.1 -r 1000  
# sp
python  train.py  --distill sp -a 0.9 -b 0.1 -r 3000  
# cc
python  train.py  --distill cc -a 0.9 -b 0.1 -r 0.02  
# vid
python  train.py  --distill vid  -a 0 -b 1  
# rkd
python  train.py  --distill rkd  -a 0.9 -b 0.1 -r 1.0  
# pkt
python  train.py  --distill pkt  -a 0.9 -b 0.1 -r 30000  
# nst
python  train.py  --distill nst  -a 0.9 -b 0.1 -r 50  
# svd
python  train.py  --distill svd  -a 0.9 -b 0.1 -r 10