#!/usr/bin/env bash
# Pre-push audit: refuse to publish anything that looks like a secret, a private
# path, or a name you did not mean to ship. Run from the repo root.
#
#   bash scripts/audit_release.sh
#   PRIVATE_TERMS='yourname|your-org|yourcluster' bash scripts/audit_release.sh
#
# PRIVATE_TERMS is a regex of things specific to you — usernames, hostnames, mount
# points, wandb entities, institution names. Keep it out of the repo (pass it on
# the command line or from a gitignored file); the point is not to commit the very
# strings you are scanning for.
set -uo pipefail
fail=0
IGNORE='max_tokens|num_tokens|tokenizer|tokenize|n_tokens|token_len|token_id|token_level|_tokens\b|tool-call|tool_call'
BENIGN='0\.0\.0\.0|127\.0\.0\.1|namespace\('

check () {
  local name="$1" pattern="$2"
  local hits
  hits=$(grep -rInE "$pattern" . --exclude-dir=.git --exclude-dir=__pycache__ 2>/dev/null \
         | grep -viE "$IGNORE" | grep -vE "$BENIGN" | grep -v "scripts/audit_release.sh")
  if [ -n "$hits" ]; then
    printf 'FAIL  %s\n' "$name"; printf '%s\n' "$hits" | sed 's/^/        /' | head -12; fail=1
  else
    printf 'ok    %s\n' "$name"
  fi
}

check "credential-shaped literals"  'sk-[A-Za-z0-9]{16,}|ghp_|github_pat_|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN'
check "non-empty keys or passwords" '(API_KEY|api_key|SECRET|secret|TOKEN|password)["'"'"' ]*[:=]["'"'"' ]*[A-Za-z0-9_-]{8,}'
check "routable IP addresses"       '([0-9]{1,3}\.){3}[0-9]{1,3}'
check "absolute home paths"         '/Users/|/home/[a-z]'
check "cluster mounts"              '/nfs_|/ceph|/scratch/|/mnt/[a-z]'
check "kubernetes manifests"        'kubectl|apiVersion:'
check "wandb entity hardcoded"      'WANDB_ENTITY=[^$ ]'
[ -n "${PRIVATE_TERMS:-}" ] && check "your private terms" "$PRIVATE_TERMS"

if [ -f .env ]; then
  echo "FAIL  .env exists — it is gitignored, but confirm it is not staged"
  git ls-files --error-unmatch .env >/dev/null 2>&1 && { echo "        .env IS TRACKED"; fail=1; }
fi

echo
[ "$fail" = 0 ] && echo "AUDIT PASSED" || echo "AUDIT FAILED — fix the above before pushing"
exit $fail
