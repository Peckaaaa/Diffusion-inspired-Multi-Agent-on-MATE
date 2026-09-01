#!/usr/bin/env bash
# Verify that DIMA/ and mate/ are the pinned upstream commits plus only the
# documented patches (see research/UPSTREAM_PATCHES.md).
#
#   bash research/scripts/verify_upstream.sh
#
# Exits non-zero if anything other than the accepted files differs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DIMA_URL='https://github.com/breez3young/DIMA.git'
DIMA_COMMIT='3dcacaa80162cf6822bf5972b4e3ad4cb2e6ceb0'
MATE_URL='https://github.com/XuehaiPan/mate.git'
MATE_COMMIT='3e631c0c3b043990fc53ae5fc3a37b0f65f230c5'

# Files allowed to differ, per research/UPSTREAM_PATCHES.md.
ACCEPTED='DIMA/environments.py
DIMA/agent/learners/DreamerLearner.py
DIMA/dataset.py
DIMA/agent/memory/DreamerMemory.py'

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fetch() {
    local url="$1" commit="$2" dest="$3"
    git clone --quiet "$url" "$dest"
    git -C "$dest" checkout --quiet "$commit"
    rm -rf "${dest:?}/.git"
}

echo "Fetching upstream references..."
fetch "$DIMA_URL" "$DIMA_COMMIT" "$WORK/DIMA"
fetch "$MATE_URL" "$MATE_COMMIT" "$WORK/mate"

status=0
for name in DIMA mate; do
    echo
    echo "=== $name ==="
    # Compare file *contents*; ignore mode bits and Python bytecode caches.
    differing="$(
        diff -rq --exclude=__pycache__ --exclude='*.pyc' "$name" "$WORK/$name" 2>&1 \
            | sed -E "s|^Files ($name/[^ ]+) and .*|\1|; s|^Only in ($name)([^:]*): (.*)|\1\2/\3|" \
            | sort || true
    )"
    if [[ -z "$differing" ]]; then
        echo "identical to upstream $([[ $name == DIMA ]] && echo "$DIMA_COMMIT" || echo "$MATE_COMMIT")"
        continue
    fi
    while IFS= read -r file; do
        [[ -z "$file" ]] && continue
        if grep -Fxq "$file" <<<"$ACCEPTED"; then
            echo "  documented patch : $file"
        else
            echo "  UNDOCUMENTED     : $file"
            status=1
        fi
    done <<<"$differing"
done

echo
if [[ $status -eq 0 ]]; then
    echo "OK: only the patches documented in research/UPSTREAM_PATCHES.md are present."
else
    echo "FAIL: undocumented differences from upstream (see above)."
fi
exit "$status"
