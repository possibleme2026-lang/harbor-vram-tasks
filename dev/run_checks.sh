#!/bin/bash
# Run every upstream check script in terminal-bench/scripts/checks against a
# candidate task directory, and print a compact PASS/FAIL summary.
#
# Usage: run_checks.sh <task-dir> [<checks-repo-root>]
#
# The checks take task directories as positional args, so the candidate does
# NOT need to live inside the terminal-bench checkout.

export PATH="/d/Git/usr/bin:/d/Git/cmd:/c/Users/less/.workbuddy/binaries/python/versions/3.13.12:$PATH"

TASK_DIR="${1:?usage: run_checks.sh <task-dir> [checks-repo-root]}"
REPO="${2:-/d/codebase/multimodal/terminal-bench}"
CHECKS="$REPO/scripts/checks"

if [ ! -d "$TASK_DIR" ]; then echo "no such task dir: $TASK_DIR" >&2; exit 2; fi
if [ ! -d "$CHECKS" ]; then echo "no checks dir: $CHECKS" >&2; exit 2; fi

# Resolve to an absolute native path so find/grep inside the checks behave.
TASK_ABS=$(cd "$TASK_DIR" && pwd -W 2>/dev/null || cd "$TASK_DIR" && pwd)

pass=0; fail=0; skip=0
FAILED_NAMES=""

for script in "$CHECKS"/check-*.sh; do
    name=$(basename "$script")
    out=$(cd "$REPO" && bash "$script" "$TASK_ABS" 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
        if echo "$out" | grep -qiE "no (task|file)s? to check|^No "; then
            printf '  SKIP  %s\n' "$name"; skip=$((skip+1))
        else
            printf '  PASS  %s\n' "$name"; pass=$((pass+1))
        fi
    else
        printf '  FAIL  %s\n' "$name"; fail=$((fail+1))
        FAILED_NAMES="$FAILED_NAMES $name"
        echo "$out" | sed 's/^/          | /'
    fi
done

echo ""
echo "===== $pass passed, $fail failed, $skip skipped ====="
if [ $fail -gt 0 ]; then
    echo "failed:$FAILED_NAMES"
    exit 1
fi
exit 0
