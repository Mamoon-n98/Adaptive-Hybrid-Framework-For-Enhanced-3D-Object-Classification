#!/bin/bash
# Process starts: Check mode.
if [ "$1" == "train" ]; then
  python training/train.py  # Run train.
elif [ "$1" == "eval" ]; then
  python training/eval.py  # Run eval.
else
  echo "Usage: ./run.sh [train|eval]"  # Usage.
fi
# Process ends: Script done.