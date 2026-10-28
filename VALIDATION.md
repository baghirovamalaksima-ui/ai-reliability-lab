# Validation

Validated locally on 2026-10-28. These are smoke and regression test results,
not a production retrieval quality benchmark.

| Check | Result |
| --- | --- |
| Windows Python 3.12, all tests with `RUN_EMBEDDING_TESTS=1` | 35 passed, 0 skipped |
| Dependency consistency (`pip check`) | No broken requirements |
| Docker build, Python 3.12 slim | Passed |
| Default Docker test command | 34 passed, 1 optional embedding test skipped |
| Docker tests with `--network none` | 34 passed, 1 optional embedding test skipped |
| Installed package imported from `/tmp` inside container | Passed |
| Container process user | UID 10001 |

The multilingual integration test used the real
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model, with
Azerbaijani and Turkish queries against English documents. Tested packages:
sentence-transformers 5.7.0, transformers 5.21.0, torch 2.14.1, and pytest 9.2.0.
The model was tested on Windows; the default Docker image intentionally contains
only the offline test dependencies.

The initial model test encountered Windows path length limits in a deeply nested
cache. Repeating with a short `HF_HOME` path passed. Hugging Face emitted two
cache symlink warnings; ordinary file copies were used successfully. No Windows
system settings were changed. See README for the short cache path configuration.

CI configuration is included but has not been executed on GitHub in this session.
