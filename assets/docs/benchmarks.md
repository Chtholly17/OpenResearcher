# Evaluation Benchmarks

This document provides an overview of the benchmarks used for evaluation in this project.

## Available Benchmarks

| Benchmark | Dataset Key | Size | Language | Search Backend | Description |
|-----------|-------------|------|----------|----------------|-------------|
| **BrowseComp-Plus** | `browsecomp_plus` | 830 | EN | local | Deep-research benchmark isolating retriever and LLM agent effects |
| **HLE** | `hle` | 2,158 | EN | serper | Multiple choice questions from Humanity's Last Exam |
| **GAIA-text** | `gaia` | 103 | EN | serper | Text-only subset of GAIA benchmark (dev split) |
| **WebWalkerQA** | `webwalkerqa` | 680 | EN | serper | Web navigation and interaction questions (test split) |
| **WebWalkerQA-ref** | `webwalkerqa_ref` | 680 | EN | serper | With reference URLs in question |
| **XBench** | `xbench` | 100 | ZH | serper | DeepSearch benchmark with encrypted test cases |
| **SealQA** | `seal` | 111 | EN | serper | Hardest subset of SealQA questions |
| **SealQA-ref** | `seal_ref` | 111 | EN | serper | With reference URLs |

## Usage

To run evaluations on these benchmarks, use the dataset keys listed in the "Dataset Key" column:

```bash
# BrowseComp-Plus (requires local search service)
bash scripts/start_search_service.sh bm25 8000
bash run_agent.sh results/bc 8001 2 browsecomp_plus local <model>

# Other benchmarks (using Serper API)
bash run_agent.sh results/hle 8001 2 hle serper <model>
bash run_agent.sh results/gaia 8001 2 gaia serper <model>
bash run_agent.sh results/xbench 8001 2 xbench serper <model>
```

## Search Backend Notes

- **local**: Required for BrowseComp-Plus. Uses local BM25 or Dense search service.
- **serper**: Uses Serper Google Search API. Requires `SERPER_API_KEY` in `.env`.
