from __future__ import annotations

from pathlib import Path

import pytest

from sre_ingest.parser import ParseContext, parse_args, parse_document
from sre_ingest.testsite import FILES, make_site
from sre_shared.models import DocType

DOCS = "content/en/docs"


@pytest.fixture
def ctx(tmp_path: Path) -> ParseContext:
    make_site(tmp_path)
    return ParseContext.load(tmp_path, "1.36", "https://v1-36.docs.kubernetes.io")


def parse(ctx: ParseContext, rel: str) -> str:
    doc = parse_document(rel, FILES[rel], ctx)
    assert doc is not None
    return doc.parsed_text


PODS = f"{DOCS}/concepts/workloads/pods/_index.md"
CONFIGURE = f"{DOCS}/tasks/configure-pod/index.md"


def test_front_matter_type_and_url(ctx: ParseContext) -> None:
    doc = parse_document(PODS, FILES[PODS], ctx)
    assert doc is not None
    assert doc.title == "Pods"
    assert doc.doc_type is DocType.CONCEPT
    assert doc.url == "https://v1-36.docs.kubernetes.io/docs/concepts/workloads/pods/"
    assert doc.k8s_version == "1.36"
    task = parse_document(CONFIGURE, FILES[CONFIGURE], ctx)
    assert task is not None
    assert task.doc_type is DocType.TASK
    assert task.url.endswith("/docs/tasks/configure-pod/")


def test_doc_type_falls_back_to_path_and_title_to_filename(ctx: ParseContext) -> None:
    rel = f"{DOCS}/reference/glossary-index.md"
    doc = parse_document(rel, FILES[rel], ctx)
    assert doc is not None
    assert doc.doc_type is DocType.REFERENCE
    assert doc.url.endswith("/docs/reference/glossary-index/")
    bare = parse_document(f"{DOCS}/tutorials/stateless/page.md", "Just text.", ctx)
    assert bare is not None
    assert (bare.title, bare.doc_type) == ("page", DocType.TUTORIAL)


def test_drafts_are_skipped(ctx: ParseContext) -> None:
    rel = f"{DOCS}/concepts/draft.md"
    assert parse_document(rel, FILES[rel], ctx) is None


def test_glossary_tooltip_uses_text(ctx: ParseContext) -> None:
    assert "runs containers." in parse(ctx, PODS)


def test_callouts_get_prefixes(ctx: ParseContext) -> None:
    text = parse(ctx, PODS)
    assert "Note: You need to install a [container runtime]" in text
    assert "Caution: Do not delete Pods by hand." in text


def test_feature_state_and_glossary_definition(ctx: ParseContext) -> None:
    text = parse(ctx, PODS)
    assert "FEATURE STATE: Kubernetes v1.24 [stable]" in text
    assert "A Pod represents a set of running containers in your cluster." in text


def test_code_sample_inlined_as_fence(ctx: ParseContext) -> None:
    text = parse(ctx, PODS)
    assert "```yaml\napiVersion: v1\nkind: Pod" in text
    assert "containerPort: 80\n```" in text


def test_tabs_become_headings_one_level_deeper(ctx: ParseContext) -> None:
    text = parse(ctx, PODS)
    assert "\n### Linux\n\nUse the Linux node image." in text
    assert "\n### Windows\n" in text


def test_mermaid_kept_whole(ctx: ParseContext) -> None:
    assert "```mermaid\ngraph TD\n  A[Pending] --> B[Running]\n```" in parse(ctx, PODS)


def test_version_shortcodes(ctx: ParseContext) -> None:
    assert "This version is v1.36; skew policy allows 1.36." in parse(ctx, PODS)


def test_heading_resolved_from_i18n(ctx: ParseContext) -> None:
    assert "\n## What's next\n" in parse(ctx, PODS)
    assert "\n## Before you begin\n" in parse(ctx, CONFIGURE)


def test_include_comment_and_escape(ctx: ParseContext) -> None:
    text = parse(ctx, CONFIGURE)
    assert "You need to have a Kubernetes cluster" in text
    assert "hidden reviewer note" not in text
    assert "<!--" not in text
    assert '{{< code_sample file="x.yaml" >}}' in text


def test_unknown_shortcodes_stripped_counted_inner_kept(ctx: ParseContext) -> None:
    text = parse(ctx, PODS)
    assert "widget inner text" in text
    assert "gizmo" not in text
    assert dict(ctx.stats.unknown) == {"widget": 1, "gizmo": 1}
    assert ctx.stats.handled["note"] == 1
    assert not ctx.stats.problems


def test_missing_lookups_are_recorded(ctx: ParseContext) -> None:
    raw = '{{% code_sample file="nope.yaml" %}} {{< include "nope.md" >}}'
    parse_document(f"{DOCS}/tasks/x.md", raw + "\ntext", ctx)
    assert ctx.stats.problems["missing_example:nope.yaml"] == 1
    assert ctx.stats.problems["missing_include:nope.md"] == 1


def test_parse_args() -> None:
    pos, named = parse_args(' "a b" file="x/y.yaml" language=\'yaml\' bare /')
    assert pos == ["a b", "bare"]
    assert named == {"file": "x/y.yaml", "language": "yaml"}
