"""A miniature kubernetes/website checkout for tests.

Snippets are adapted from https://github.com/kubernetes/website (content/en), (c) The
Kubernetes Authors, CC BY 4.0, shortened; shortcode syntax is as used upstream.
"""

from __future__ import annotations

from pathlib import Path

PODS = """\
---
title: Pods
content_type: concept
weight: 10
---

<!-- overview -->

_Pods_ are the smallest deployable units of computing that you can create and manage in Kubernetes.
A Pod runs {{< glossary_tooltip text="containers" term_id="container" >}}.

<!-- body -->

## What is a Pod?

{{< note >}}
You need to install a [container runtime](/docs/setup/production-environment/container-runtimes)
into each node.
{{< /note >}}

{{< feature-state for_k8s_version="v1.24" state="stable" >}}

{{< glossary_definition term_id="pod" length="short" >}}

Here is an example Pod:

{{% code_sample file="pods/simple-pod.yaml" %}}

{{< caution >}}
Do not delete Pods by hand.
{{< /caution >}}

## Operating systems

{{< tabs name="os_tabs" >}}
{{% tab name="Linux" %}}
Use the Linux node image.
{{% /tab %}}
{{% tab name="Windows" %}}
Use the Windows node image.
{{% /tab %}}
{{< /tabs >}}

## Lifecycle

{{< mermaid >}}
graph TD
  A[Pending] --> B[Running]
{{</ mermaid >}}

This version is {{< param "version" >}}; skew policy allows {{< skew currentVersion >}}.

{{< widget color="blue" >}}widget inner text{{< /widget >}}
{{< gizmo >}}

{{% heading "whatsnext" %}}

* Read about [Pod lifecycle](/docs/concepts/workloads/pods/pod-lifecycle/).
"""

CONFIGURE = """\
---
title: Configure a Pod
content_type: task
---

{{< include "task-tutorial-prereqs.md" >}}

{{< comment >}}hidden reviewer note{{< /comment >}}

{{% heading "prerequisites" %}}

Escaped shortcode: {{</* code_sample file="x.yaml" */>}}

## Create the Pod

```shell
kubectl apply -f https://k8s.io/examples/pods/simple-pod.yaml
```
"""

DRAFT = """\
---
title: Draft page
draft: true
---
Never ingested.
"""

REFERENCE = """\
---
title: Glossary index
---
Plain page without content_type.
"""

FILES = {
    "content/en/docs/concepts/workloads/pods/_index.md": PODS,
    "content/en/docs/tasks/configure-pod/index.md": CONFIGURE,
    "content/en/docs/reference/glossary-index.md": REFERENCE,
    "content/en/docs/concepts/draft.md": DRAFT,
    "content/en/docs/reference/glossary/pod.md": (
        "---\ntitle: Pod\nid: pod\nshort_description: >\n  A Pod represents a set of running "
        "containers in your cluster.\n---\n The smallest and simplest object.\n"
    ),
    "content/en/docs/reference/glossary/container.md": (
        "---\ntitle: Container\nid: container\nshort_description: >\n  A lightweight image.\n---\n"
    ),
    "content/en/examples/pods/simple-pod.yaml": (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: nginx\nspec:\n  containers:\n"
        "  - name: nginx\n    image: nginx:1.14.2\n    ports:\n    - containerPort: 80\n"
    ),
    "content/en/includes/task-tutorial-prereqs.md": (
        "You need to have a Kubernetes cluster, and the kubectl command-line tool must be "
        "configured to communicate with your cluster.\n"
    ),
    "i18n/en/en.toml": (
        '[note]\nother = "Note:"\n[caution]\nother = "Caution:"\n[warning]\nother = "Warning:"\n'
        '[feature_state]\nother = "FEATURE STATE:"\n'
        '[whatsnext_heading]\nother = "What\'s next"\n'
        '[prerequisites_heading]\nother = "Before you begin"\n'
    ),
}


def make_site(root: Path) -> Path:
    for rel, text in FILES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root
