# shellcheck disable=SC2102
python Generate_polygon.py \
  --dataset vis \
  --figNum 2 \
  --fill 0 \
  --width 64 \
  --thickness 3 \
  --angNums '3,4,5,6' \
  --floatRates '0.05,0.05' \
  --maskType vertex \
  --maskRate 0.1 \
  --seed 1026

