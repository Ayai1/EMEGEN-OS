# Third-party assets, models, and dependencies

This inventory records license declarations from the upstream model, dataset,
project, and package sources linked below (checked 2026-11-05). It covers the
listed direct assets and packages, not every transitive dependency or every
component bundled into third-party binary distributions.

| Asset or package | Upstream source | Declared license |
| --- | --- | --- |
| LongMemEval-S cleaned dataset | [Hugging Face dataset card](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | MIT |
| Muse Glimmer 30B weights | [Unsloth model card](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) | Apache-2.0 |
| BGE-M3 (`BAAI/bge-m3`) | [BAAI model card](https://huggingface.co/BAAI/bge-m3) | MIT |
| llama.cpp b12058 Windows CUDA release | [b12058 release archive](https://github.com/ggml-org/llama.cpp/releases/download/b12058/llama-b12058-bin-win-cuda-13.4-x64.zip); [project license](https://github.com/ggml-org/llama.cpp/blob/master/LICENSE) | MIT for llama.cpp; the local archive and runtime DLLs are not redistributed by this repository |
| aiosqlite | [PyPI project](https://pypi.org/project/aiosqlite/) | MIT |
| numpy | [PyPI project](https://pypi.org/project/numpy/) | BSD-3-Clause, 0BSD, MIT, Zlib, and CC0-1.0 components |
| httpx | [PyPI project](https://pypi.org/project/httpx/) | BSD-3-Clause |
| pytest (development extra) | [PyPI project](https://pypi.org/project/pytest/) | MIT |
| pytest-asyncio (development extra) | [PyPI project](https://pypi.org/project/pytest-asyncio/) | Apache-2.0 |
| hypothesis (development extra) | [PyPI project](https://pypi.org/project/hypothesis/) | MPL-2.0 |
| sentence-transformers (ML extra) | [PyPI project](https://pypi.org/project/sentence-transformers/) | Apache-2.0 |
| torch (ML extra) | [PyPI project](https://pypi.org/project/torch/) | BSD-style; PyPI lists Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSD-3-Clause, BSL-1.0, ISC, MIT, and Zlib components |
| flask (UI extra) | [PyPI project](https://pypi.org/project/Flask/) | BSD-3-Clause |
| setuptools (build dependency) | [PyPI project](https://pypi.org/project/setuptools/) | MIT |
| wheel (build dependency) | [PyPI project](https://pypi.org/project/wheel/) | MIT |

`pyproject.toml` declares the EngramOS project itself as MIT licensed; that
project declaration does not establish the licenses of its dependencies. This
table records the sources' license declarations; it is not a legal opinion or
a complete software bill of materials.
