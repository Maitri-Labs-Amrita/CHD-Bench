#!/usr/bin/env bash
# Set up CHD-Bench on this machine, right after `git clone` (Linux or macOS, no sudo). Needs uv and git on PATH.
#
#   ./setup.sh [--imagechd-dir DIR] [--hvsmr-dir DIR] [--slicer PATH] [--xvfb PATH] [--lib-dir DIR]...
#              [--shared DIR] [--check-case ID] [--skip-check]
#
# 1. Prerequisites: uv (https://docs.astral.sh/uv/) and git (uv fetches slicer-mcp from its git repository).
# 2. uv sync --frozen --inexact: the locked versions, including slicer-mcp at the commit pyproject.toml pins
#    ([tool.uv.sources] rev); uv.lock is not changed; installed extras such as transformers stay.
# 3. Slicer 5.10.0: --slicer PATH uses an existing one (Linux: with --xvfb PATH and --lib-dir DIR for a headless
#    install), else the one already in config/runtime.local.json. Else on Linux `slicer-mcp setup-headless` installs
#    Xvfb, Slicer (sha512-checked download) and the missing system libraries (--shared DIR: into one group folder, so
#    a team installs once). On macOS: /Applications/Slicer.app.
# 4. ImageCHD: --imagechd-dir DIR, else $CHD_IMAGECHD_DIR, else the config's imagechd_dir, else data/ImageCHD_dataset.
#    It cannot be downloaded automatically (Kaggle login).
#    HVSMR-2.0 (optional, MRI): --hvsmr-dir DIR (the folder with orig/ and hvsmr_clinical.csv) is checked and saved
#    in config/paths.local.json. Neither dataset is needed to run your own scans (./run.sh --volume ...).
# 5. config/runtime.local.json (gitignored): only the keys setup owns are set; every other setting there is kept.
# 6. The check: ./run.sh --imagechd <case> --labels gt --dry-run-scripted. Slicer starts, prepares an ImageCHD case,
#    runs one scripted episode (no model, no cost) and stops (Linux: on a virtual display). --check-case ID picks the
#    case (default: the first one); --skip-check skips it.
#
# Run it again at any time: every step skips what is already there. The summary prints the next commands (./run.sh).
# Exit code: 0 = set up and checked (or --skip-check); 3 = set up, but NOT checked (no ImageCHD data); else failed.
set -euo pipefail

CHD="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_INSTALL='curl -LsSf https://astral.sh/uv/install.sh | sh'
SLICER_VERSION="5.10.0"
MAC_SLICER="/Applications/Slicer.app/Contents/MacOS/Slicer"
CHD_RT="$CHD/config/runtime.local.json"
HEADLESS_RT="$CHD/data/dev/headless_runtime.json"
CHECK_LOG="$CHD/data/dev/setup_check.log"

SLICER=""; XVFB=""; LIB_DIRS=(); SHARED=""; IMAGECHD_DIR=""; HVSMR_DIR=""; CHECK_CASE=""; SKIP_CHECK=0
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
    --hvsmr-dir) need "$1" "the HVSMR-2.0 dataset folder (with orig/ and hvsmr_clinical.csv)" "$@"; HVSMR_DIR="$2"; shift 2 ;;
    --check-case) need "$1" "an ImageCHD case id (e.g. 1001)" "$@"; CHECK_CASE="$2"; shift 2 ;;
    --skip-check) SKIP_CHECK=1; shift ;;
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

say "1/6 prerequisites"
command -v uv >/dev/null || die "uv not found. Install it (https://docs.astral.sh/uv/), e.g.:
  $UV_INSTALL
then open a new shell and run ./setup.sh again."
command -v git >/dev/null || die "git not found (uv needs it to fetch slicer-mcp): install git and run ./setup.sh again"
echo "  ok: uv $(uv --version | awk '{print $2}') ($(command -v uv)), git"

say "2/6 uv sync (with slicer-mcp at the pinned commit)"
(cd "$CHD" && uv sync --frozen --inexact) || die "uv sync failed in $CHD (output above)"
MCP_NOTE="$(helper slicer-mcp)"
echo "  ok: $CHD/.venv; slicer-mcp $MCP_NOTE"

say "3/6 Slicer $SLICER_VERSION"
[ "$OS" = mac ] && { [ -n "$XVFB" ] || [ -n "${LIB_DIRS[*]:-}" ]; } && echo "  note: --xvfb and --lib-dir are for Linux; ignored on macOS"
if [ -z "$SLICER" ] && [ -f "$CHD_RT" ]; then
  OLD="$(helper get --path "$CHD_RT" --key slicer_executable)"
  if [ -n "$OLD" ] && [ -x "$OLD" ]; then
    SLICER="$OLD"
    echo "  reusing the Slicer in $CHD_RT"
    if [ "$OS" = linux ]; then
      [ -n "$XVFB" ] || XVFB="$(helper get --path "$CHD_RT" --key xvfb_executable)"
      [ -n "${LIB_DIRS[*]:-}" ] || while IFS= read -r d; do [ -n "$d" ] && LIB_DIRS+=("$d"); done \
        < <(helper get --path "$CHD_RT" --key extra_lib_dirs)
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
  echo "  no --slicer: running slicer-mcp setup-headless${SHARED:+ --shared $SHARED}"
  echo "  (Xvfb, Slicer $SLICER_VERSION and the missing system libraries; each part is reused when already there)"
  (cd "$CHD" && uv run --no-sync slicer-mcp setup-headless --runtime-out "$HEADLESS_RT" ${SHARED:+--shared "$SHARED"}) \
    || die "slicer-mcp setup-headless failed (output above)"
  SLICER="$(helper get --path "$HEADLESS_RT" --key slicer_executable)"
  XVFB="$(helper get --path "$HEADLESS_RT" --key xvfb_executable)"
  LIB_DIRS=(); while IFS= read -r d; do [ -n "$d" ] && LIB_DIRS+=("$d"); done < <(helper get --path "$HEADLESS_RT" --key extra_lib_dirs)
  [ -x "$SLICER" ] || die "slicer-mcp setup-headless wrote no usable slicer_executable into $HEADLESS_RT"
  SLICER_NOTE="headless install by slicer-mcp setup-headless: $SLICER"
fi
if [ "$OS" = linux ]; then
  if [ -z "$XVFB" ]; then
    XVFB="$(command -v Xvfb || true)"
    [ -n "$XVFB" ] || die "no Xvfb (Linux runs Slicer on a virtual display): pass --xvfb PATH, or run ./setup.sh without --slicer so slicer-mcp setup-headless installs Xvfb"
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

say "4/6 data: ImageCHD (CT) and HVSMR-2.0 (MRI)"
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

echo "  HVSMR-2.0 (optional, MRI):"
HV_OK=0; HV_NOTE="not set"
if [ -n "$HVSMR_DIR" ]; then
  if OUT="$(helper hvsmr --dir "$HVSMR_DIR" --path "$CHD/config/paths.local.json" 2>&1)"; then
    IFS=$'\t' read -r HV_ROOT HV_N HV_WRITE <<<"$OUT"
    HV_OK=1; HV_NOTE="$HV_ROOT ($HV_N cases)"
    echo "  ok: $HV_NOTE; saved in config/paths.local.json ($HV_WRITE)"
  else
    die "HVSMR-2.0 data: $OUT"
  fi
elif OUT="$(helper hvsmr-current 2>/dev/null)"; then
  IFS=$'\t' read -r HV_ROOT HV_N <<<"$OUT"
  HV_OK=1; HV_NOTE="$HV_ROOT ($HV_N cases)"
  echo "  ok: $HV_NOTE (from config/paths.local.json)"
else
  echo "  not set (optional; for MRI cases: ./setup.sh --hvsmr-dir DIR)"
fi

say "5/6 config/runtime.local.json"
CFG_ARGS=(--slicer "$SLICER")
if [ "$OS" = linux ]; then
  CFG_ARGS+=(--linux --xvfb "$XVFB")
  for d in ${LIB_DIRS[@]+"${LIB_DIRS[@]}"}; do CFG_ARGS+=(--lib-dir "$d"); done
fi
CHD_CFG="$(helper chd --path "$CHD_RT" --base "$CHD/config/runtime.json" "${CFG_ARGS[@]}" \
  ${IMAGECHD_KEY:+--imagechd-dir "$IMAGECHD_KEY"})"
echo "  $CHD_RT: $CHD_CFG"

on_display() {
  if [ "$OS" = linux ]; then
    (cd "$CHD" && SLICER_MCP_RUNTIME="$CHD_RT" uv run --no-sync slicer-mcp xvfb-session --runtime "$CHD_RT" -- "$@")
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

say "6/6 check: one scripted episode (no model, no cost)"
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

KEYS="$(helper keys 2>/dev/null || true)"
case "$CHECK" in PASSED*) CHK_LINE="PASSED: Slicer starts, a case loads, a run folder is written" ;; *) CHK_LINE="$CHECK" ;; esac
RULE="================================================================"
cat <<EOF

$RULE
                          SETUP RESULT
$RULE
 Python + slicer-mcp   ok       slicer-mcp $MCP_NOTE
 3D Slicer             ok       $SLICER ($VERSION_NOTE)
EOF
if [ "$DATA_OK" = 1 ]; then
  echo " ImageCHD (CT)         ok       $DATA_DIR ($N_CASES cases)"
else
  echo " ImageCHD (CT)         not set  optional: ./setup.sh --imagechd-dir DIR"
fi
if [ "$HV_OK" = 1 ]; then
  echo " HVSMR-2.0 (MRI)       ok       $HV_NOTE"
else
  echo " HVSMR-2.0 (MRI)       not set  optional: ./setup.sh --hvsmr-dir DIR"
fi
if [ -n "$KEYS" ]; then
  echo " Model keys            ok       found: $KEYS (values not shown)"
else
  echo " Model keys            none     free test runs only; a paid model needs a key in $CHD/.env"
fi
echo " Test run (free)       $CHK_LINE"
echo "$RULE"

if [ "$RC" = 1 ]; then
  echo
  echo "The test run failed. The last lines of $CHECK_LOG:"
  tail -15 "$CHECK_LOG" | sed 's/^/  | /'
  echo "Common causes: a Slicer that crashes on exit after case preparation (\"exit abnormally\"): try another"
  echo "Slicer $SLICER_VERSION install (--slicer PATH, or run without --slicer for a fresh one); a fixed bridge_port"
  echo "in $CHD_RT that another Slicer already uses; missing system libraries (Linux: --lib-dir DIR, or run"
  echo "without --slicer)."
  exit 1
fi

echo
echo "Next, from $CHD (every option: ./run.sh --help; every situation: README.md):"
if [ "$DATA_OK" = 1 ]; then
  echo "  ./run.sh --list imagechd                                   the cases and their diagnoses"
  echo "  ./run.sh --imagechd $CASE --labels gt --dry-run-scripted        a free test run (no model)"
  echo "  ./run.sh --imagechd $CASE --labels gt --model claude-opus-5-5   a real model (paid; asks before it starts)"
fi
echo "  ./run.sh --volume /path/to/scan_or_dicom_folder --model M  your own scan (no dataset needed)"
echo "  ./run.sh --list classes                                    the class codes for --truth"
if [ -z "$KEYS" ]; then
  echo "  A real model needs a key: put one line in $CHD/.env, for example OPENROUTER_API_KEY=sk-or-..."
fi
if [ "$DATA_OK" != 1 ] && [ "$SKIP_CHECK" != 1 ]; then
  echo "  The free test run needs the ImageCHD data (step 4 above). Without it, setup is installed but NOT verified."
fi
exit "$RC"
