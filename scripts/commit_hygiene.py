"""Commit-message / PR-text hygiene check.

What this guards: AI **attribution pollution** — `Co-Authored-By:` and
other credit trailers, `🤖 Generated with <tool>` footers, `--trailer`
co-author lines. Those should never land in git history, whatever tool
produced them (Claude, Cursor, Codex, Copilot, Devin, Antigravity, …).

What this deliberately does NOT flag: bare mentions. Naming a file or
directory (`CLAUDE.md`, `.claude/`, `.cursor/`, `AGENTS.md`), or
disclosing AI assistance in prose — which `CONTRIBUTING.md` explicitly
asks contributors to do — is fine. Only *attribution* is rejected, so
the check no longer fights the disclosure policy or trips on kernels
named after models (`llama`, `mistral`, …).

Detection — a line is an issue if it is:
  (T) part of the trailing git-trailer block (`Word: …` paragraph at
      the end) — the repo bans all trailers, see PR #110;
  (K) an explicit attribution-trailer key anywhere
      (`Co-Authored-By:`, `Signed-off-by:`, `Reviewed-by:`, …);
  (P) an AI-attribution phrase — the 🤖 footer emoji, or an
      attribution verb (`generated`/`co-authored`/`created`/…) joined
      by `by`/`with`/`using` to a known AI tool name on the same line.
"""

import json
import os
import re
import subprocess
import sys

# AI tool / model names — used ONLY to qualify rule (P): an attribution
# verb phrase is an issue when it credits one of these. Mentions on
# their own are not flagged, so this list does not need to be
# exhaustive; (T)/(K)/the 🤖 emoji are tool-agnostic catch-alls.
AI_TERMS = [
    r"claude", r"anthropic", r"\bcodex\b", r"openai", r"chatgpt",
    r"\bgpt[- ]?\d", r"antigravity", r"gemini", r"\bbard\b",
    r"copilot", r"\bcursor\b", r"sourcegraph", r"\bcody\b",
    r"\bdevin\b", r"\baider\b", r"windsurf", r"tabnine",
    r"\bllama\b", r"\bmistral\b", r"\bgrok\b", r"perplexity",
    r"replit", r"ghostwriter", r"\bpieces\b",
]
AI_RE = re.compile("|".join(AI_TERMS), re.IGNORECASE)

# Generic git trailer: `Word: value` (used for the trailing-block scan).
TRAILER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]*:\s")

# (K) Explicit attribution-trailer keys — flagged wherever they appear,
# not just in the trailing block (catches a `--trailer` planted
# mid-message).
ATTRIB_KEY_RE = re.compile(
    r"(?im)^\s*(co-?authored-by|signed-off-by|reviewed-by|tested-by"
    r"|acked-by|assisted-by|generated-by|created-by|authored-by"
    r"|helped-by|suggested-by|reported-by|written-by)\s*:"
)

# (P) Attribution verb joined to `by`/`with`/`using`.
ATTRIB_PHRASE_RE = re.compile(
    r"(?i)\b(co-?authored|generated|created|authored|written|produced"
    r"|powered|assisted|made|built|crafted)\b.{0,30}?\b(by|with|using)\b"
)
ROBOT = "\U0001f916"  # 🤖 — the conventional AI-footer emoji.

MARKER = "<!-- ai-mention-hygiene-check -->"


def _attribution_hit(line):
    """True if `line` is an AI-attribution line (rule K or P)."""
    if ATTRIB_KEY_RE.search(line):
        return True
    if ROBOT in line:
        return True
    if ATTRIB_PHRASE_RE.search(line) and AI_RE.search(line):
        return True
    return False


def find_issues(text):
    issues = []
    lines = text.splitlines()

    # (K) + (P): scan every line for explicit attribution.
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if ATTRIB_KEY_RE.search(ln):
            issues.append(("trailer", s))
        elif ROBOT in ln or (ATTRIB_PHRASE_RE.search(ln) and AI_RE.search(ln)):
            issues.append(("attribution", s))

    # (T): the trailing git-trailer block — a run of `Word: …` lines
    # forming the final paragraph (preceded by a blank line).
    body = list(lines)
    while body and not body[-1].strip():
        body.pop()
    trailers = []
    i = len(body) - 1
    while i >= 0 and body[i].strip() != "" and TRAILER_RE.match(body[i]):
        trailers.append(body[i])
        i -= 1
    if trailers and i >= 0 and body[i].strip() == "":
        for t in reversed(trailers):
            issues.append(("trailer", t.strip()))

    # Dedup (a line can match both K and T), preserving first sighting.
    seen = set()
    uniq = []
    for kind, ln in issues:
        if ln in seen:
            continue
        seen.add(ln)
        uniq.append((kind, ln))
    return uniq


def clean_text(text):
    """Strip attribution lines + the trailing trailer block. Bare
    mentions and prose disclosure are left intact."""
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    # Drop the trailing trailer block.
    trailer_start = len(lines)
    i = len(lines) - 1
    while i >= 0 and lines[i].strip() != "" and TRAILER_RE.match(lines[i]):
        trailer_start = i
        i -= 1
    if trailer_start < len(lines):
        if trailer_start > 0 and lines[trailer_start - 1].strip() == "":
            lines = lines[:trailer_start]
            while lines and not lines[-1].strip():
                lines.pop()
    # Drop attribution lines anywhere (NOT bare mentions).
    lines = [l for l in lines if not _attribution_hit(l)]
    out = []
    prev_blank = True
    for l in lines:
        is_blank = (l.strip() == "")
        if is_blank and prev_blank:
            continue
        out.append(l)
        prev_blank = is_blank
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, check=True, **kw)


pr = os.environ["PR_NUMBER"]
repo = os.environ["REPO"]
is_fork = os.environ.get("PR_HEAD_REPO", "") != repo

commits_json = run(
    ["gh", "pr", "view", pr, "--repo", repo, "--json", "commits"]
).stdout
commits = json.loads(commits_json)["commits"]

findings = []
for c in commits:
    sha = c["oid"][:7]
    headline = c.get("messageHeadline") or ""
    body = c.get("messageBody") or ""
    full = headline + (("\n\n" + body) if body else "")
    issues = find_issues(full)
    if issues:
        findings.append({"sha": sha, "subject": headline, "issues": issues})

pr_title = os.environ.get("PR_TITLE", "") or ""
pr_body = os.environ.get("PR_BODY", "") or ""
pr_text = pr_title + (("\n\n" + pr_body) if pr_body else "")
pr_issues = find_issues(pr_text)

sanitized_pr = False
if pr_issues and not is_fork:
    cleaned_title = clean_text(pr_title)
    new_title = cleaned_title if cleaned_title.strip() else pr_title
    new_body = clean_text(pr_body)
    try:
        run(["gh", "pr", "edit", pr, "--repo", repo,
             "--title", new_title, "--body", new_body])
        sanitized_pr = True
        pr_issues = []
        print("::notice::Sanitized PR title and/or body")
    except subprocess.CalledProcessError as e:
        print(f"::warning::Could not auto-sanitize PR title/body: {e.stderr}")

out_lines = [MARKER, "", "## Commit message hygiene check", ""]
clean = not findings and not pr_issues
if clean:
    if sanitized_pr:
        out_lines.append("Auto-cleaned the PR title/body. All commits look fine. :white_check_mark:")
    else:
        out_lines.append("All commit messages and PR text are clean. :white_check_mark:")
else:
    out_lines.append(
        "This PR has commit messages or PR text with AI-attribution"
        " pollution: git trailers (Co-Authored-By, Signed-off-by, any"
        " `--trailer …`) or `🤖 Generated with <tool>`-style footers."
        " Naming files like `CLAUDE.md` / `.cursor/`, or disclosing AI"
        " assistance in prose, is fine — only attribution is rejected."
    )
    out_lines.append("")
    if findings:
        out_lines.append("### Commits with issues")
        out_lines.append("")
        for f in findings:
            out_lines.append(f"- `{f['sha']}` {f['subject']}")
            for kind, ln in f["issues"]:
                tag = "Attribution" if kind == "attribution" else "Trailer"
                out_lines.append(f"  - **{tag}:** `{ln}`")
        out_lines.append("")
    if pr_issues:
        out_lines.append("### PR title / body")
        out_lines.append("")
        for kind, ln in pr_issues:
            tag = "Attribution" if kind == "attribution" else "Trailer"
            out_lines.append(f"- **{tag}:** `{ln}`")
        out_lines.append("")
    out_lines.extend([
        "### How to fix",
        "",
        "- For commits: rewrite the branch (e.g. `git rebase -i <base>`),"
        " drop the offending lines from each commit message, then force-push.",
        "- For the PR title/body: just edit the PR description.",
    ])

comment_body = "\n".join(out_lines)

sticky_ids = []
jq_filter = f'.[] | select(.body | contains("{MARKER}")) | .id'
try:
    ids_out = run(["gh", "api", "--paginate",
                   f"repos/{repo}/issues/{pr}/comments",
                   "-q", jq_filter]).stdout
    sticky_ids = [int(x) for x in ids_out.split() if x.strip()]
except subprocess.CalledProcessError as e:
    print(f"::warning::Could not list existing comments: {e.stderr}")

try:
    if sticky_ids:
        for cid in sticky_ids:
            run(["gh", "api", "--method", "PATCH",
                 f"repos/{repo}/issues/comments/{cid}",
                 "-f", f"body={comment_body}"])
    elif not clean:
        run(["gh", "pr", "comment", pr, "--repo", repo,
             "--body", comment_body])
except subprocess.CalledProcessError as e:
    print(f"::warning::Could not post/update comment: {e.stderr}")

if not clean:
    n_commits = len(findings)
    n_pr = len(pr_issues)
    print(f"::error::Found issues in {n_commits} commit(s) and {n_pr} PR text field(s)")
    sys.exit(1)
