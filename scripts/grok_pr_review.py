"""Post or update a sticky Grok review comment on a GitHub pull request.

Used by ``.github/workflows/grok-review.yml``. Stdlib only so CI needs no extra
pip packages. Set ``XAI_API_KEY`` and ``GITHUB_TOKEN`` in the environment.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

MARKER = "<!-- grok-bot-review -->"
XAI_URL = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4.6"
MAX_DIFF_CHARS = 80_000
EXCLUDE_PATTERNS = (
    "*.joblib",
    "*.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "*.min.js",
    "*.png",
    "*.jpg",
    "*.wasm",
)

SYSTEM_PROMPT = """You are Grok bot, an expert code reviewer for a Python MCP server
(context-eng) that returns query-matched, token-budgeted context packs.

Review the pull request diff. Focus on:
- Correctness bugs and regressions
- Security (path jail, secret leakage, unsafe subprocess, token/secret handling)
- Reliability of the MCP stdio server (uncaught errors, unbounded caches)
- Tests that are missing for the changed behavior

Do not nitpick style. Do not praise filler. If the diff is clean, say so in one
short paragraph.

Format:
## Grok bot review
**Verdict:** Approve | Comment | Request changes
Then a bullet list of findings (file path + why). If none, one sentence.
"""


def path_is_excluded(rel_path: str, patterns: tuple[str, ...] = EXCLUDE_PATTERNS) -> bool:
    name = Path(rel_path.replace("\\", "/")).name
    posix = rel_path.replace("\\", "/")
    for pat in patterns:
        if fnmatch.fnmatch(posix, pat) or fnmatch.fnmatch(name, pat):
            return True
        if not pat.startswith("**/") and fnmatch.fnmatch(posix, f"**/{pat}"):
            return True
    return False


def truncate_diff(diff: str, max_chars: int = MAX_DIFF_CHARS) -> str:
    if len(diff) <= max_chars:
        return diff
    return diff[:max_chars] + f"\n\n[diff truncated to {max_chars} characters]\n"


def _diff_git_path(header: str) -> str:
    """Return the ``b/`` path from a ``diff --git a/... b/...`` header line."""
    parts = header.split()
    if len(parts) >= 4 and parts[-1].startswith("b/"):
        return parts[-1][2:].strip()
    return ""


def filter_unified_diff(diff: str, patterns: tuple[str, ...] = EXCLUDE_PATTERNS) -> str:
    """Drop file hunks whose path matches ``EXCLUDE_PATTERNS``."""
    kept: list[str] = []
    current: list[str] = []
    current_path = ""
    for line in diff.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current and current_path and not path_is_excluded(current_path, patterns):
                kept.extend(current)
            elif current and not current_path:
                kept.extend(current)
            current = [line]
            current_path = _diff_git_path(line)
        else:
            current.append(line)
    if current and (not current_path or not path_is_excluded(current_path, patterns)):
        kept.extend(current)
    return "".join(kept)


def _request(
    url: str,
    *,
    method: str = "GET",
    token: str | None = None,
    payload: dict | None = None,
    accept: str = "application/json",
) -> tuple[int, str]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Accept": accept,
        "User-Agent": "context-eng-grok-bot",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return exc.code, body


def github_api(path: str, token: str, *, method: str = "GET", payload: dict | None = None) -> dict | list:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    status, body = _request(
        f"{api}{path}",
        method=method,
        token=token,
        payload=payload,
        accept="application/vnd.github+json",
    )
    if status >= 400:
        raise RuntimeError(f"GitHub API {method} {path} failed ({status}): {body[:500]}")
    return json.loads(body) if body else {}


def fetch_pr_diff(repo: str, number: int, token: str) -> str:
    api = os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    status, body = _request(
        f"{api}/repos/{repo}/pulls/{number}",
        token=token,
        accept="application/vnd.github.diff",
    )
    if status >= 400:
        raise RuntimeError(f"Failed to fetch PR diff ({status}): {body[:500]}")
    return body


def review_with_grok(diff: str, *, title: str, model: str, api_key: str) -> str:
    status, body = _request(
        XAI_URL,
        method="POST",
        token=api_key,
        payload={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Pull request title: {title}\n\n"
                        f"Unified diff:\n```diff\n{truncate_diff(diff)}\n```"
                    ),
                },
            ],
        },
    )
    if status >= 400:
        raise RuntimeError(f"xAI API failed ({status}): {body[:800]}")
    data = json.loads(body)
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected xAI response: {body[:800]}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("xAI returned an empty review")
    return content.strip()


def upsert_sticky_comment(repo: str, number: int, token: str, markdown: str) -> None:
    body = f"{MARKER}\n{markdown}\n"
    comments = github_api(f"/repos/{repo}/issues/{number}/comments?per_page=100", token)
    existing_id = None
    if isinstance(comments, list):
        for item in comments:
            if isinstance(item, dict) and MARKER in str(item.get("body") or ""):
                existing_id = item.get("id")
                break
    if existing_id is not None:
        github_api(
            f"/repos/{repo}/issues/comments/{existing_id}",
            token,
            method="PATCH",
            payload={"body": body},
        )
        return
    github_api(
        f"/repos/{repo}/issues/{number}/comments",
        token,
        method="POST",
        payload={"body": body},
    )


def load_event_pr(event_path: str) -> tuple[str, int]:
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    pr = event.get("pull_request") or {}
    title = str(pr.get("title") or event.get("action") or "pull request")
    number = int(pr["number"])
    return title, number


def main() -> int:
    api_key = os.environ.get("XAI_API_KEY", "").strip()
    gh_token = os.environ.get("GITHUB_TOKEN", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    event_path = os.environ.get("GITHUB_EVENT_PATH", "").strip()
    model = os.environ.get("XAI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if not api_key:
        print("XAI_API_KEY is not set; skipping Grok review.", file=sys.stderr)
        return 0
    if not gh_token or not repo or not event_path:
        print("GITHUB_TOKEN, GITHUB_REPOSITORY, and GITHUB_EVENT_PATH are required", file=sys.stderr)
        return 1

    title, number = load_event_pr(event_path)
    diff = fetch_pr_diff(repo, number, gh_token)
    diff = filter_unified_diff(diff)
    if not diff.strip():
        markdown = "## Grok bot review\n\n**Verdict:** Approve\n\nNo diff to review."
    else:
        markdown = review_with_grok(diff, title=title, model=model, api_key=api_key)
        if MARKER in markdown:
            markdown = markdown.replace(MARKER, "")
    upsert_sticky_comment(repo, number, gh_token, markdown)
    print(f"Posted Grok review on {repo}#{number} using {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
