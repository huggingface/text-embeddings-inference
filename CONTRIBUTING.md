<!---
Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Contribute to Text Embeddings Inference (TEI)

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping
others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

However you choose to contribute, please be mindful and respect our
[code of conduct](https://github.com/huggingface/text-embeddings-inference/blob/main/CODE_OF_CONDUCT.md).

**This guide was heavily inspired by the awesome [scikit-learn guide to contributing](https://github.com/scikit-learn/scikit-learn/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to Text Embeddings Inference (TEI).

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) and open
a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The Text Embeddings Inference (TEI) solution is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the
library itself, and not your code.

Once you've confirmed the bug hasn't already been reported, please include the following information in your issue so
we can quickly resolve it:

* Your **OS type and version**, as well as your environment versions (versions of rust, python, and dependencies).
* A short, self-contained, code snippet that allows us to reproduce the bug.
* The *full* traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

### Do you want a new feature?

If there is a new feature you'd like to see in Text Embeddings Inference (TEI), please open an issue and describe:

1. What is the *motivation* behind this feature? Is it related to a problem or frustration with the library? Is it
   a feature related to something you need for a project? Is it something you worked on and think it could benefit
   the community?

   Whatever it is, we'd love to hear about it!

2. Describe your requested feature in as much detail as possible. The more you can tell us about it, the better
   we'll be able to help you.
3. Provide a *code snippet* that demonstrates the feature's usage.
4. If the feature is related to a paper, please include a link.

If your issue is well written we're already 80% of the way there by the time you create it.

We have added [templates](https://github.com/huggingface/text-embeddings-inference/tree/main/.github/ISSUE_TEMPLATE)
to help you get started with your issue.

## Do you want to implement a new model?

New models are constantly released and if you want to implement a new model, please provide the following information:

* A short description of the model and a link to the paper.
* Link to the implementation if it is open-sourced.
* Link to the model weights if they are available.

If you are willing to contribute the model yourself, let us know so we can help you add it to Text Embeddings Inference (TEI)!

## Do you want to add documentation?

We're always looking for improvements to the documentation that make it more clear and accurate. Please let us know
how the documentation can be improved such as typos and any content that is missing, unclear or inaccurate. We'll be
happy to make the changes or help you make a contribution if you're interested!

## I want to become a maintainer of the project. How do I get there?

Text Embeddings Inference (TEI) is a project led and managed by Hugging Face as it powers our internal services. However, we are happy to have
motivated individuals from other organizations join us as maintainers with the goal of making TEI the best inference
service for embedding models on production on production.

If you are such an individual (or organization), please reach out to us and let's collaborate.

## CI checks and maintainer-requested builds

Pull requests run formatting/lint checks and focused CPU library tests. Documentation-only PRs skip Rust compilation. The image matrix and tests that download models are skipped on ordinary PR runs; GitHub's own fork-workflow approval setting still applies before any workflow can start.

A maintainer with repository write access can request the expensive checks from the Actions tab or CLI after reviewing the exact PR head:

```bash
# Replace both values with the PR and the full reviewed head SHA.
gh workflow run test.yaml --ref main -f pr_number=123 -f head_sha=FULL_HEAD_SHA
gh workflow run build.yaml --ref main -f pr_number=123 -f head_sha=FULL_HEAD_SHA -f variants=cpu
```

For images, `variants` accepts comma-separated names from `.github/workflows/matrix.json`, or `all`. It defaults to `cpu`. Each new PR revision needs a new request; a stale SHA is rejected. PR validation never publishes images or receives registry/AWS/model-read secrets. Public-model tests use hosted runners, and manually selected PR code cannot export shared compiler or image caches.

Relevant main pushes run the trusted model suite and publish images; documentation-only pushes do not rebuild images. Isolated Dockerfile changes select their matching variants. Release tags build the full image matrix and are not superseded by later runs. A non-PR manual image build is validation-only unless `publish=true` is explicitly selected on main or a release tag.

The existing pinned sccache uses cache API v2 in Docker builds and GitHub runtime cache settings on runners, with incremental compilation disabled for cacheable Rust builds and native-CPU feature isolation for host compiler caches. Image layers keep per-variant HTTP/gRPC scopes. Trusted publishing uses the configured S3 cache when available; otherwise it uses GitHub's cache. Missing AWS configuration is skipped rather than failing a validation build. Cache statistics remain available even when compilation fails.

Documentation changes use the pinned doc-builder workflow. Its existing bot publishes the preview URL. A separate trusted `workflow_run` job reports build/publication problems in a sticky comment and clears that failure status after a successful publication. It ignores closed PRs, ambiguous fork branches, and stale heads, and never checks out or executes the PR's code with documentation-publishing credentials. `HF_DOC_BUILD_PUSH` enables preview uploads; `COMMENT_BOT_TOKEN` is optional because the reporter can use `GITHUB_TOKEN`.

These maintainer-dispatch and reporting changes become available on the default branch after this CI PR is merged. Required-check rules should use the lightweight checks; optional model/image jobs are intentionally skipped until requested.
