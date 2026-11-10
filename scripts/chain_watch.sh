#!/usr/bin/env bash
# Watch Day-2; on success save results and launch Day-3; then watch Day-3. Prints a short report at the end.
R="C:/Users/shive/AppData/Local/Programs/lexsi-hackathon"; K=shivendushivu/lexsi-day1-checks
wait_run() {  # $1 = results dir
  local st=$(date +%s) s i
  for i in $(seq 1 80); do
    s=$(kaggle kernels status $K 2>/dev/null)
    case "$s" in *ERROR*|*COMPLETE*|*CANCEL*)
      echo "$(date +%H:%M) $1 ended after $(( ($(date +%s)-st)/60 )) min: $s"
      mkdir -p "$R/results/$1"; kaggle kernels output $K -p "$R/results/$1" --force >/dev/null 2>&1
      cat "$R/results/$1/summary.jsonl" 2>/dev/null; return;; esac
    [ $i -le 8 ] && sleep 60 || sleep 600
  done; echo "TIMEOUT $1"
}
sleep 120; wait_run day2
if grep -q '"DONE"' "$R/results/day2/summary.jsonl" 2>/dev/null; then
  echo "== launching day3 =="; kaggle kernels push -p "$R/kaggle/day3"; sleep 90; wait_run day3
else echo "== day2 did not finish; day3 NOT launched =="; fi
