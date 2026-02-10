#!/usr/bin/env bash
set -euo pipefail

# Force git to use the user's SSH key and avoid /root/.ssh/known_hosts issues
KEY="/data/nas-gpu/wang/xianghu/.ssh/id_rsa"
KNOWN_HOSTS="/data/nas-gpu/wang/xianghu/corteva_colab_project/Hyperspec-Building/Incremental_Clustering/.known_hosts_github"

exec ssh \
  -i "$KEY" \
  -o IdentitiesOnly=yes \
  -o UserKnownHostsFile="$KNOWN_HOSTS" \
  -o StrictHostKeyChecking=accept-new \
  "$@"

