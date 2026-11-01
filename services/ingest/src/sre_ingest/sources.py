"""Pin release branches (`pin`) and check them out sparsely (`fetch`)."""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from typing import Any

import httpx
import yaml

from sre_shared.config import CorpusConfig, CorpusSource
from sre_shared.settings import get_settings

SPARSE_DIRS = ["content/en/docs", "content/en/examples", "content/en/includes", "i18n/en"]
CORPUS_HEADER = """\
# Corpus sources. `sre-ingest pin` fills `versions` with release branches and commit SHAs,
# so the lab and CI rebuild exactly the same corpus. Owned by the ingest service.
"""


def _git(args: list[str], cwd: Path | None = None) -> str:
    res = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()[:300]}")
    return res.stdout.strip()


def ls_remote_sha(repo: str, branch: str) -> str:
    out = _git(["ls-remote", repo, f"refs/heads/{branch}"])
    if not out:
        raise RuntimeError(f"branch {branch} not found in {repo}")
    return out.split()[0]


def parse_version_table(hugo_toml: str, count: int) -> list[dict[str, str]]:
    """First `count` entries of `[[params.versions]]` (newest first)."""
    data: dict[str, Any] = tomllib.loads(hugo_toml)
    versions = data.get("params", {}).get("versions", [])
    if len(versions) < count:
        raise RuntimeError(f"hugo.toml lists {len(versions)} versions, wanted {count}")
    return [dict(v) for v in versions[:count]]


def raw_hugo_toml(repo: str) -> str:
    owner_repo = repo.removeprefix("https://github.com/").removesuffix(".git").strip("/")
    url = f"https://raw.githubusercontent.com/{owner_repo}/main/hugo.toml"
    resp = httpx.get(url, timeout=30.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def resolve_sources(cfg: CorpusConfig, hugo_toml: str, count: int) -> list[CorpusSource]:
    out: list[CorpusSource] = []
    for v in parse_version_table(hugo_toml, count):
        branch = v["docsbranch"]
        out.append(
            CorpusSource(
                k8s_version=v["version"].removeprefix("v"),
                branch=branch,
                commit_sha=ls_remote_sha(cfg.repo, branch),
                url_base=v["url"].rstrip("/"),
            )
        )
    return out


def write_corpus_config(cfg: CorpusConfig, path: Path) -> None:
    body = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    path.write_text(CORPUS_HEADER + body, encoding="utf-8", newline="\n")


def pin(count: int, config_dir: Path | None = None) -> CorpusConfig:
    from sre_shared.config import load_corpus_config

    cdir = config_dir or get_settings().config_dir
    cfg = load_corpus_config(cdir)
    cfg.versions = resolve_sources(cfg, raw_hugo_toml(cfg.repo), count)
    write_corpus_config(cfg, cdir / "corpus.yaml")
    return cfg


def source_dir(source: CorpusSource, data_dir: Path | None = None) -> Path:
    return (data_dir or get_settings().data_dir) / "sources" / source.k8s_version


def fetch(cfg: CorpusConfig, source: CorpusSource, data_dir: Path | None = None) -> Path:
    """Sparse, shallow checkout of the pinned commit; a no-op when already at that commit."""
    dest = source_dir(source, data_dir)
    sha = source.commit_sha or ls_remote_sha(cfg.repo, source.branch)
    dest.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        _git(["init", "-q"], dest)
        _git(["remote", "add", "origin", cfg.repo], dest)
    try:
        if _git(["rev-parse", "HEAD"], dest) == sha:
            return dest
    except RuntimeError:
        pass
    _git(["sparse-checkout", "set", "--cone", *SPARSE_DIRS], dest)
    _git(["fetch", "-q", "--depth", "1", "origin", sha], dest)
    _git(["checkout", "-q", "--force", "FETCH_HEAD"], dest)
    return dest
