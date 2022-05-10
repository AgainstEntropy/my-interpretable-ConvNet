# shellcheck disable=SC2102
python Generate_polygon.py \
  --dataset test \
  --figNum 2 \
  --fill 0 \
  --width 64 \
  --thickness 3 \
  --angNums '3,4,5,6' \
  --floatRates '0.05,0.05' \
  --maskType None \
  --maskRate 0.3 \
  --seed 1026

