#!/usr/bin/env bash
# Set up CHD-Bench on this machine, right after `git clone` (Linux or macOS, no sudo).
#
#   ./setup.sh [--imagechd-dir DIR] [--slicer PATH] [--xvfb PATH] [--lib-dir DIR]... [--shared DIR]
#              [--check-case ID] [--skip-check] [--no-install-uv]
#
# 1. Prerequisites: git, and uv (installed with its official installer if missing; --no-install-uv stops instead).
# 2. slicer-eval-mcp: cloned next to CHD-Bench at the commit pinned in src/chd_client/slicer.py (PINNED_COMMIT). An
#    existing checkout must be at that commit with no uncommitted changes; setup never changes it.
# 3. uv sync --inexact (the locked versions; uv.lock is not changed; installed extras such as transformers stay).
# 4. Slicer 5.10.0: --slicer PATH uses an existing one (Linux: with --xvfb PATH and --lib-dir DIR for a headless
#    install), else the one already in config/runtime.local.json. Else on Linux slicer-eval-mcp's
#    tools/setup_headless.sh installs Xvfb, Slicer (sha512-checked download) and the missing system libraries
#    (--shared DIR: into one group folder, so a team installs once). On macOS: /Applications/Slicer.app.
# 5. ImageCHD: --imagechd-dir DIR, else $CHD_IMAGECHD_DIR, else the config's imagechd_dir, else data/ImageCHD_dataset.
#    It cannot be downloaded automatically (Kaggle login).
# 6. config/runtime.local.json of CHD-Bench and of slicer-eval-mcp (gitignored): only the keys setup owns are set;
#    every other setting already there is kept.
# 7. The check: ./run.sh --imagechd <case> --labels gt --dry-run-scripted. Slicer starts, prepares an ImageCHD case,
#    runs one scripted episode (no model, no cost) and stops (Linux: on a virtual display). --check-case ID picks the
#    case (default: the first one); --skip-check skips it.
#
# Run it again at any time: every step skips what is already there. The summary prints the next commands (./run.sh).
# Exit code: 0 = set up and checked (or --skip-check); 3 = set up, but NOT checked (no ImageCHD data); else failed.
set -euo pipefail

set -euo pipefail

CHD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP="$(dirname "$CHD")/slicer-eval-mcp"
MCP_SSH="git@github.com:Maitri-Labs-Amrita/slicer-eval-mcp.git"
MCP_HTTPS="https://github.com/Maitri-Labs-Amrita/slicer-eval-mcp.git"
UV_INSTALL='curl -LsSf https://astral.sh/uv/install.sh | sh'
SLICER_VERSION="5.10.0"
MAC_SLICER="/Applications/Slicer.app/Contents/MacOS/Slicer"
CHD_RT="$CHD/config/runtime.local.json"
MCP_RT="$MCP/config/runtime.local.json"
CHECK_LOG="$CHD/data/dev/setup_check.log"

SLICER=""; XVFB=""; LIB_DIRS=(); SHARED=""; IMAGECHD_DIR=""; CHECK_CASE=""; SKIP_CHECK=0; INSTALL_UV=1
need() {
  local opt="$1" what="$2"; shift 2
  if [ $# -lt 2 ] || [ -z "$2" ] || [ "${2#--}" != "$2" ]; then
    echo "setup.sh: $opt requires $what (see ./setup.sh --help)" >&2
    exit 2
  fi
}
while [ $# -gt 0 ]; do
  case "$1" in
    --slicer) need "$1" "the path to the Slicer launcher (e.g. .../Slicer-5.10.0-linux-amd64/Slicer)" "$@"; SLICER="$2"; shift 2 ;;
    --xvfb) need "$1" "the path to the Xvfb executable" "$@"; XVFB="$2"; shift 2 ;;
    --lib-dir) need "$1" "a folder of extra system libraries" "$@"; LIB_DIRS+=("$2"); shift 2 ;;
    --shared) need "$1" "the group folder to install Xvfb and Slicer into" "$@"; SHARED="$2"; shift 2 ;;
    --imagechd-dir) need "$1" "the ImageCHD dataset folder" "$@"; IMAGECHD_DIR="$2"; shift 2 ;;
    --check-case) need "$1" "an ImageCHD case id (e.g. 1001)" "$@"; CHECK_CASE="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --no-install-uv) INSTALL_UV=0; shift ;;
    -h|--help) sed -n '2,/^set -euo pipefail$/{/^set -euo pipefail$/!p;}' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see ./setup.sh --help)" >&2; exit 2 ;;
  esac
done

say() { printf '\n== %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
helper() { (cd "$CHD" && uv run --no-sync python -m chd_client.setup_config "$@"); }

case "$(uname -s)" in
  Linux) OS=linux ;;
  Darwin) OS=mac ;;
  *) die "unsupported OS $(uname -s): CHD-Bench runs on Linux or macOS" ;;
esac

say "1/7 prerequisites"
command -v git >/dev/null || die "git not found: install git and run ./setup.sh again"
UV_NOTE="reused"
if ! command -v uv >/dev/null; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    UV_NOTE="reused ~/.local/bin/uv (not on your PATH: add 'export PATH=\$HOME/.local/bin:\$PATH' to your shell profile)"
  elif [ "$INSTALL_UV" = 0 ]; then
    die "uv not found. Install it with:  $UV_INSTALL  (then open a new shell and run ./setup.sh again)"
  else
    command -v curl >/dev/null || die "uv and curl not found: install uv (https://docs.astral.sh/uv/) and run ./setup.sh again"
    echo "  uv not found. Installing it with the official installer: $UV_INSTALL"
    echo "  (it puts uv in ~/.local/bin and adds that folder to PATH in your shell profile; --no-install-uv stops here instead)"
    sh -c "$UV_INSTALL"
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null || die "the uv installer finished, but uv is not in ~/.local/bin: open a new shell and run ./setup.sh again"
    UV_NOTE="installed (open a new shell so that uv is on your PATH)"
  fi
fi
echo "  ok: git, uv $(uv --version | awk '{print $2}') ($(command -v uv))"

say "2/7 slicer-eval-mcp (next to CHD-Bench, at the pinned commit)"
PIN="$(sed -n 's/^PINNED_COMMIT = "\([0-9a-f]\{40\}\)".*/\1/p' "$CHD/src/chd_client/slicer.py")"
[ -n "$PIN" ] || die "cannot read PINNED_COMMIT from $CHD/src/chd_client/slicer.py"
if [ -e "$MCP" ]; then
  [ -d "$MCP/.git" ] || [ -f "$MCP/.git" ] || die "$MCP exists but is not a git checkout: move it away and run ./setup.sh again"
  HEAD="$(git -C "$MCP" rev-parse HEAD 2>/dev/null || true)"
  if [ "$HEAD" != "$PIN" ]; then
    die "$MCP is at ${HEAD:-no commit}, not the pinned $PIN. setup.sh never changes your checkout. Commit or stash your work there, then:
  git -C $MCP fetch origin && git -C $MCP checkout $PIN
and run ./setup.sh again."
  fi
  DIRTY="$(git -C "$MCP" status --porcelain --untracked-files=all)"
  if [ -n "$DIRTY" ]; then
    printf '%s\n' "$DIRTY" | head -10 >&2
    die "$MCP has uncommitted changes (above); the harness refuses them. Commit, stash or remove them (setup.sh never discards them), then run ./setup.sh again."
  fi
  MCP_NOTE="reused $MCP at ${PIN:0:12}"
  echo "  ok: $MCP at ${PIN:0:12}, clean"
else
  echo "  cloning $MCP_SSH -> $MCP"
  if ! GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=20" git clone -q "$MCP_SSH" "$MCP"; then
    echo "  ssh clone failed; trying $MCP_HTTPS"
    git clone -q "$MCP_HTTPS" "$MCP" || die "cannot clone slicer-eval-mcp (ssh and https failed): check your network access to GitHub"
  fi
  git -C "$MCP" checkout -q --detach "$PIN" || die "commit $PIN is not in $MCP_SSH"
  MCP_NOTE="cloned into $MCP at ${PIN:0:12}"
  echo "  ok: cloned, at ${PIN:0:12}"
fi

say "3/7 uv sync"
(cd "$CHD" && uv sync --frozen --inexact) || die "uv sync failed in $CHD (output above)"
echo "  ok: $CHD/.venv"

say "4/7 Slicer $SLICER_VERSION"
[ "$OS" = mac ] && { [ -n "$XVFB" ] || [ -n "${LIB_DIRS[*]:-}" ]; } && echo "  note: --xvfb and --lib-dir are for Linux; ignored on macOS"
if [ -z "$SLICER" ] && [ -f "$CHD_RT" ]; then
  OLD="$(helper get --path "$CHD_RT" --key slicer_executable)"
  if [ -n "$OLD" ] && [ -x "$OLD" ]; then
    SLICER="$OLD"
    echo "  reusing the Slicer in $CHD_RT"
    if [ "$OS" = linux ] && [ -f "$MCP_RT" ]; then
      [ -n "$XVFB" ] || XVFB="$(helper get --path "$MCP_RT" --key xvfb_executable)"
      [ -n "${LIB_DIRS[*]:-}" ] || while IFS= read -r d; do [ -n "$d" ] && LIB_DIRS+=("$d"); done \
        < <(helper get --path "$MCP_RT" --key extra_lib_dirs)
    fi
  fi
fi
if [ -n "$SLICER" ]; then
  [ -f "$SLICER" ] && [ -x "$SLICER" ] || die "--slicer $SLICER: not an executable file (the Slicer launcher, e.g. .../Slicer-5.10.0-linux-amd64/Slicer)"
  SLICER="$(abspath "$SLICER")"
  SLICER_NOTE="reused $SLICER"
elif [ "$OS" = mac ]; then
  [ -x "$MAC_SLICER" ] || die "no Slicer at $MAC_SLICER. Install Slicer $SLICER_VERSION:
  1. Open https://download.slicer.org/ and download Slicer $SLICER_VERSION for macOS, the x86_64 (Intel) build
     (it also runs on Apple silicon, through Rosetta).
  2. Open the .dmg and drag Slicer.app into /Applications.
  3. Run ./setup.sh again (or pass --slicer PATH for a Slicer elsewhere)."
  SLICER="$MAC_SLICER"
  SLICER_NOTE="reused $SLICER"
else
  echo "  no --slicer: running slicer-eval-mcp's tools/setup_headless.sh${SHARED:+ --shared $SHARED}"
  echo "  (Xvfb, Slicer $SLICER_VERSION and the missing system libraries; each part is reused when already there)"
  bash "$MCP/tools/setup_headless.sh" ${SHARED:+--shared "$SHARED"} || die "tools/setup_headless.sh failed (output above)"
  SLICER="$(helper get --path "$MCP_RT" --key slicer_executable)"
  XVFB="$(helper get --path "$MCP_RT" --key xvfb_executable)"
  LIB_DIRS=(); while IFS= read -r d; do [ -n "$d" ] && LIB_DIRS+=("$d"); done < <(helper get --path "$MCP_RT" --key extra_lib_dirs)
  [ -x "$SLICER" ] || die "setup_headless.sh wrote no usable slicer_executable into $MCP_RT"
  SLICER_NOTE="headless install by setup_headless.sh: $SLICER"
fi
if [ "$OS" = linux ]; then
  if [ -z "$XVFB" ]; then
    XVFB="$(command -v Xvfb || true)"
    [ -n "$XVFB" ] || die "no Xvfb (Linux runs Slicer on a virtual display): pass --xvfb PATH, or run ./setup.sh without --slicer so setup_headless.sh installs Xvfb"
  fi
  [ -x "$XVFB" ] || die "--xvfb $XVFB: not an executable file"
  XVFB="$(abspath "$XVFB")"
  for i in ${LIB_DIRS[@]+"${!LIB_DIRS[@]}"}; do
    [ -d "${LIB_DIRS[$i]}" ] || die "--lib-dir ${LIB_DIRS[$i]}: no such folder"
    LIB_DIRS[$i]="$(cd "${LIB_DIRS[$i]}" && pwd)"
  done
  echo "  Xvfb: $XVFB"
  [ -z "${LIB_DIRS[*]:-}" ] || echo "  extra library folders: ${LIB_DIRS[*]}"
fi
echo "  ok: $SLICER"

say "5/7 ImageCHD data"
DATA_GIVEN="${IMAGECHD_DIR:-${CHD_IMAGECHD_DIR:-}}"
if [ -z "$DATA_GIVEN" ] && [ -f "$CHD_RT" ]; then
  DATA_GIVEN="$(helper get --path "$CHD_RT" --key imagechd_dir)"
  [ "$DATA_GIVEN" = "data/ImageCHD_dataset" ] && DATA_GIVEN=""
fi
DATA_DIR="${DATA_GIVEN:-$CHD/data/ImageCHD_dataset}"
DATA_DIR="${DATA_DIR/#\~/$HOME}"
case "$DATA_DIR" in /*) ;; *) DATA_DIR="$CHD/$DATA_DIR" ;; esac
DATA_OK=0; N_CASES=0; IMAGECHD_KEY=""
if [ -n "$DATA_GIVEN" ] || [ -d "$DATA_DIR" ]; then
  if OUT="$(helper check-data --dir "$DATA_DIR" ${CHECK_CASE:+--case "$CHECK_CASE"} 2>&1)"; then
    read -r CASE N_CASES <<<"$OUT"
    DATA_OK=1
    DATA_DIR="$(cd "$DATA_DIR" && pwd)"
    [ "$DATA_DIR" = "$CHD/data/ImageCHD_dataset" ] || IMAGECHD_KEY="$DATA_DIR"
    echo "  ok: $DATA_DIR ($N_CASES cases; check case $CASE)"
  else
    die "ImageCHD data: $OUT"
  fi
else
  cat <<EOF
  NOT FOUND: $DATA_DIR
  The ImageCHD dataset (Xu et al., MICCAI 2020) needs a Kaggle login, so setup cannot download it:
    https://www.kaggle.com/xiaoweixumedicalai/datasets
  Put ct_<id>_image.nii.gz, ct_<id>_label.nii.gz, imagechd_dataset_image_info.xlsx and
  imageCHD_dataset_diagnosis_info(updated20210510).xlsx in $CHD/data/ImageCHD_dataset/,
  or pass --imagechd-dir DIR, then run ./setup.sh again. Without the data the episode check is skipped.
EOF
fi

say "6/7 config files"
LINUX_FLAG=""; [ "$OS" = linux ] && LINUX_FLAG="--linux"
MCP_ARGS=(--slicer "$SLICER" $LINUX_FLAG)
if [ "$OS" = linux ]; then
  MCP_ARGS+=(--xvfb "$XVFB")
  for d in ${LIB_DIRS[@]+"${LIB_DIRS[@]}"}; do MCP_ARGS+=(--lib-dir "$d"); done
fi
MCP_CFG="$(helper mcp --path "$MCP_RT" --base "$MCP/config/runtime.json" "${MCP_ARGS[@]}")"
echo "  $MCP_RT: $MCP_CFG"
CHD_CFG="$(helper chd --path "$CHD_RT" --base "$CHD/config/runtime.json" --slicer "$SLICER" $LINUX_FLAG \
  ${IMAGECHD_KEY:+--imagechd-dir "$IMAGECHD_KEY"})"
echo "  $CHD_RT: $CHD_CFG"

on_display() {
  if [ "$OS" = linux ]; then
    (cd "$CHD" && SLICER_MCP_RUNTIME="$CHD_RT" uv run --no-sync python "$MCP/tools/xvfb_session.py" --runtime "$MCP_RT" -- "$@")
  else
    (cd "$CHD" && SLICER_MCP_RUNTIME="$CHD_RT" "$@")
  fi
}
TO=""; command -v timeout >/dev/null && TO="timeout -k 60"

VERSION="$(on_display ${TO:+$TO 120} "$SLICER" --version 2>/dev/null | sed -n 's/^Slicer \([0-9][0-9.]*\).*/\1/p' | head -1 || true)"
if [ -z "$VERSION" ]; then
  VERSION_NOTE="version not read (Slicer --version gave no version line): not checked"
elif [ "$VERSION" = "$SLICER_VERSION" ]; then
  VERSION_NOTE="version $VERSION"
else
  VERSION_NOTE="version $VERSION: WARNING, CHD-Bench is tested with $SLICER_VERSION"
fi
echo "  Slicer $VERSION_NOTE"

say "7/7 check: one scripted episode (no model, no cost)"
CHECK="" RC=0
if [ "$SKIP_CHECK" = 1 ]; then
  CHECK="SKIPPED (--skip-check): the setup is NOT verified"
  echo "  $CHECK"
elif [ "$DATA_OK" = 0 ]; then
  CHECK="SKIPPED (no ImageCHD data): the setup is NOT fully verified"
  echo "  $CHECK"
  RC=3
else
  mkdir -p "$(dirname "$CHECK_LOG")"
  echo "  ./run.sh --imagechd $CASE --labels gt --dry-run-scripted; output also in $CHECK_LOG (takes a few minutes)"
  set +e
  (cd "$CHD" && SLICER_MCP_RUNTIME="$CHD_RT" ${TO:+${TO/timeout/timeout --foreground} 1800} ./run.sh --imagechd "$CASE" \
    --labels gt --dry-run-scripted) 2>&1 | tee "$CHECK_LOG"
  ERC=${PIPESTATUS[0]}
  set -e
  RUN_DIR="$(grep -o 'reports/runs/[^ ]*' "$CHECK_LOG" | tail -1 || true)"
  if [ "$ERC" = 0 ] && grep -q "COMPLETE.json written" "$CHECK_LOG"; then
    CHECK="PASSED (ImageCHD $CASE${RUN_DIR:+, run folder $RUN_DIR})"
  else
    CHECK="FAILED (exit $ERC; log $CHECK_LOG)"
    RC=1
  fi
fi

cat <<EOF

== Summary
  uv:              $UV_NOTE
  slicer-eval-mcp: $MCP_NOTE
  Slicer:          $SLICER_NOTE ($VERSION_NOTE)
  ImageCHD:        $([ "$DATA_OK" = 1 ] && echo "$DATA_DIR ($N_CASES cases)" || echo "NOT FOUND ($DATA_DIR)")
  config:          $CHD_RT ($CHD_CFG)
                   $MCP_RT ($MCP_CFG)
  check:           $CHECK
EOF
if [ "$RC" = 1 ]; then
  cat <<EOF

The check failed. The last lines of $CHECK_LOG:
EOF
  tail -15 "$CHECK_LOG" | sed 's/^/  | /'
  cat <<EOF
Common causes: a Slicer that crashes on exit after case preparation ("exit abnormally"): try another Slicer $SLICER_VERSION
install (--slicer PATH, or run without --slicer for a fresh one); a fixed bridge_port in $CHD_RT that another Slicer
already uses; missing system libraries (Linux: --lib-dir DIR, or run without --slicer).
EOF
  exit 1
fi
if [ "$DATA_OK" = 1 ]; then
  cat <<EOF

Next, from $CHD (./run.sh starts Slicer, runs one episode and stops Slicer$([ "$OS" = linux ] && echo "; on a virtual display")):
  cd $CHD
  ./run.sh --imagechd $CASE --labels gt --dry-run-scripted
A real model calls a paid API. Set its key first: export it, or put KEY=value lines in $CHD/.env (./run.sh reads it).
  export ANTHROPIC_API_KEY=...
  ./run.sh --imagechd $CASE --labels gt --model claude-opus-5-5
  export OPENROUTER_API_KEY=...
  ./run.sh --imagechd $CASE --labels gt --model 'openrouter:openai/gpt-6-astra'
Other cases: uv run python -m chd_client.task.imagechd list. On a SLURM cluster, run episodes on a compute node
(README.md, "Setup in detail").
EOF
else
  cat <<EOF

Next: put the ImageCHD data in place (see above) and run ./setup.sh again; it then runs the check.
EOF
fi
exit "$RC"
