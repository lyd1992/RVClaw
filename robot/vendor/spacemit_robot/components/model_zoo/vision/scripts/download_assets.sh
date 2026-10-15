#!/bin/sh
# 仅下载 examples/、applications/ 配置中引用的图片、视频到 ~/.cache/assets
# 远程：https://archive.spacemit.com/spacemit-ai/model_zoo/assets/
# 在 cv 组件根目录执行: bash scripts/download_assets.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_URL="${VISION_ASSETS_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/assets}"
CACHE="${HOME}/.cache/assets"
mkdir -p "$CACHE"

echo "========== Download referenced assets to $CACHE =========="
echo "Remote: $BASE_URL"
echo "Scan: $ROOT/examples, $ROOT/applications (*.yaml)"

ASSET_PATHS=$(find "$ROOT/examples" "$ROOT/applications" -name '*.yaml' \
  -exec grep -hoE '\.cache/assets/(image|video)/[A-Za-z0-9._-]+' {} + 2>/dev/null | sort -u)

if [ -z "$ASSET_PATHS" ]; then
  echo "Warning: no asset paths found in yaml configs."
  exit 0
fi

download_one() {
  rel="$1"
  dest="$CACHE/$rel"
  mkdir -p "$(dirname "$dest")"
  if [ -f "$dest" ]; then
    echo "  Exists: $dest"
    return 0
  fi
  url="$BASE_URL/$rel"
  echo "  >>> $rel"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$dest" "$url"
  elif command -v curl >/dev/null 2>&1; then
    curl -fsSL -o "$dest" "$url"
  else
    echo "Error: need wget or curl."
    exit 1
  fi
}

for path in $ASSET_PATHS; do
  case "$path" in
    .cache/assets/*) rel="${path#.cache/assets/}" ;;
    *) echo "Skip invalid path: $path"; continue ;;
  esac
  download_one "$rel"
done

echo ""
echo "========== Done =========="
echo "Assets cache: $CACHE"
ls -la "$CACHE/image" "$CACHE/video" 2>/dev/null || true
