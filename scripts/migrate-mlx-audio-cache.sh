#!/usr/bin/env bash
#
# migrate-mlx-audio-cache.sh — convert flat `mlx-audio/<org>_<repo>/` snapshot
# directories into the canonical HuggingFace blob/snapshot/refs layout under
# `~/.cache/huggingface/hub/models--<org>--<repo>/`.
#
# Why:
#   FFAI's `ModelLocator` (and the broader `huggingface_hub` ecosystem) expects
#   the canonical layout — refs/main pinning a snapshot revision, snapshots/<rev>
#   holding relative symlinks into blobs/<sha256>. The legacy `mlx-audio` sibling
#   cache produced flat snapshot directories that don't satisfy that contract,
#   so the integration tests had to special-case them via
#   `AudioTestHelpers.resolveCheckpoint(mlxAudioSlugs:repoIds:)`. This script
#   rewrites every `mlx-audio/<slug>/` into a proper HF cache entry so the test
#   helpers can drop that branch.
#
# Properties:
#   * **Idempotent** — if the target `models--<org>--<repo>/` already exists with
#     a valid `refs/main`, the slug is skipped.
#   * **Reversible** — source slugs are NOT deleted. To roll back, simply
#     `rm -rf ~/.cache/huggingface/hub/models--<org>--<repo>/` and the original
#     flat snapshot is still in place.
#   * **Slug parsing** — directory names use "first underscore = org/repo split"
#     (e.g. `mlx-community_Kokoro-82M-bf16` -> org `mlx-community`, repo
#     `Kokoro-82M-bf16`). Subsequent underscores/hyphens in the repo half are
#     preserved.
#   * **Subdirectories preserved** — nested files become
#     `snapshots/main/<sub>/<file>` symlinks with the correct relative depth.
#   * **Relative symlinks** — `snapshots/main/<file>` always points at
#     `../../blobs/<sha>` (or `../../../blobs/<sha>` for subdirs), so the cache
#     remains valid if the entire `~/.cache/huggingface/hub/` tree is moved.
#
# This script does NOT touch any other part of the cache. Files outside
# `~/.cache/huggingface/hub/mlx-audio/` are never read or written.

set -euo pipefail

HF_CACHE_ROOT="${HOME}/.cache/huggingface/hub"
MLX_AUDIO_ROOT="${HF_CACHE_ROOT}/mlx-audio"

# Stable marker for refs/main so re-runs can detect a prior migration.
MIGRATION_MARKER="local-migration-$(date +%Y%m%dT%H%M%S)"

if [[ ! -d "${MLX_AUDIO_ROOT}" ]]; then
    echo "No mlx-audio cache found at ${MLX_AUDIO_ROOT} — nothing to migrate."
    exit 0
fi

# Compute the sha256 of a file. macOS ships `shasum -a 256`; Linux ships
# `sha256sum`. Use whichever is on PATH.
sha256_of() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    else
        shasum -a 256 "$f" | awk '{print $1}'
    fi
}

# Number of `../` segments required to climb from snapshots/main/<subpath>/<file>
# back to the snapshots/ sibling blobs/ dir. A file at depth 0 (snapshots/main/x)
# needs ../../blobs; depth 1 (snapshots/main/sub/x) needs ../../../blobs; etc.
relative_blobs_prefix() {
    local depth="$1"
    local prefix=""
    local i
    # +2 for the two fixed levels (main/ + snapshots/).
    for ((i = 0; i < depth + 2; i++)); do
        prefix+="../"
    done
    printf '%sblobs' "$prefix"
}

migrated=0
skipped=0
slugs_processed=0

shopt -s nullglob
for slug_dir in "${MLX_AUDIO_ROOT}"/*/; do
    slug_dir="${slug_dir%/}"
    slug="$(basename "${slug_dir}")"
    slugs_processed=$((slugs_processed + 1))

    # First-underscore split: everything before the first `_` is the org;
    # everything after is the repo (which may itself contain `_` or `-`).
    if [[ "${slug}" != *_* ]]; then
        echo "[skip] ${slug}: no underscore separator; cannot infer org/repo"
        skipped=$((skipped + 1))
        continue
    fi
    org="${slug%%_*}"
    repo="${slug#*_}"

    target_dir="${HF_CACHE_ROOT}/models--${org}--${repo}"
    refs_main="${target_dir}/refs/main"

    # Idempotency: a target with a non-empty refs/main is considered already
    # migrated. We only check for existence + non-empty contents, not the
    # specific marker — a hand-curated cache entry should also count as
    # "already migrated, leave alone".
    if [[ -d "${target_dir}" && -s "${refs_main}" ]]; then
        echo "[skip] ${slug}: target ${target_dir} already exists with refs/main"
        skipped=$((skipped + 1))
        continue
    fi

    echo "[migrate] ${slug} -> models--${org}--${repo}"

    mkdir -p "${target_dir}/blobs"
    mkdir -p "${target_dir}/refs"
    mkdir -p "${target_dir}/snapshots/main"

    # Walk every regular file in the slug source. -print0 / read -d '' is the
    # safe way to handle filenames with spaces or newlines.
    while IFS= read -r -d '' src_file; do
        # Path of `src_file` relative to the slug root (e.g. `config.json` or
        # `mlx-int8/model.safetensors`).
        rel_path="${src_file#${slug_dir}/}"
        rel_dir="$(dirname "${rel_path}")"
        file_name="$(basename "${rel_path}")"

        # Compute the blob digest + place the bytes under blobs/.
        sha="$(sha256_of "${src_file}")"
        blob_path="${target_dir}/blobs/${sha}"
        if [[ ! -f "${blob_path}" ]]; then
            cp "${src_file}" "${blob_path}"
        fi

        # Materialise the snapshot directory (creating any nested subdirs)
        # and replace any pre-existing symlink at the target.
        if [[ "${rel_dir}" == "." ]]; then
            snapshot_dir="${target_dir}/snapshots/main"
            depth=0
        else
            snapshot_dir="${target_dir}/snapshots/main/${rel_dir}"
            mkdir -p "${snapshot_dir}"
            # Depth = number of `/` separators in rel_dir.
            depth="$(awk -F'/' '{print NF}' <<<"${rel_dir}")"
        fi

        blobs_prefix="$(relative_blobs_prefix "${depth}")"
        link_target="${blobs_prefix}/${sha}"
        snapshot_link="${snapshot_dir}/${file_name}"

        # Replace if a link / file already exists — avoids `ln -s` failure.
        rm -f "${snapshot_link}"
        ln -s "${link_target}" "${snapshot_link}"
    done < <(find "${slug_dir}" -type f -print0)

    # Stamp refs/main last so a partial migration leaves the slug looking
    # un-migrated and a re-run will recover it.
    printf '%s' "${MIGRATION_MARKER}" > "${refs_main}"
    migrated=$((migrated + 1))
done
shopt -u nullglob

echo
echo "mlx-audio cache migration complete."
echo "  slugs processed: ${slugs_processed}"
echo "  slugs migrated:  ${migrated}"
echo "  slugs skipped:   ${skipped}"
echo
echo "Source slugs preserved at ~/.cache/huggingface/hub/mlx-audio/;"
echo "remove manually after verifying integration tests pass."
