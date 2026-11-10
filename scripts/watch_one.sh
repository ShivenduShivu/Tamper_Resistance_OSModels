#!/usr/bin/env bash
# Usage: watch_one.sh <results_dir>. Polls the Kaggle notebook until it ends, then downloads output and prints summary.
R="C:/Users/shive/AppData/Local/Programs/lexsi-hackathon"; K=shivendushivu/lexsi-day1-checks; st=$(date +%s)
sleep 120
for i in $(seq 1 80); do
  s=$(kaggle kernels status $K 2>/dev/null)
  case "$s" in *ERROR*|*COMPLETE*|*CANCEL*)
    echo "$(date +%H:%M) $1 ended after $(( ($(date +%s)-st)/60 )) min: $s"
    mkdir -p "$R/results/$1"; kaggle kernels output $K -p "$R/results/$1" --force >/dev/null 2>&1
    cat "$R/results/$1/summary.jsonl" 2>/dev/null; exit 0;; esac
  [ $i -le 6 ] && sleep 60 || sleep 900
done; echo "TIMEOUT $1"
