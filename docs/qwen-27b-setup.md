# Qwen 27B Local Setup on Apple Silicon

## Scope

This document records the tested local setup used to run the Ternary-Bonsai 27B 2-bit model on an M4 Mac with 16 GB unified memory.

## Hardware / environment

- Host: Apple M4 Mac, 16 GB unified memory
- Shell: zsh
- Python virtual environment: `~/qwen3-env`
- MLX-LM version verified during testing: `0.31.3`
- MLX-VLM was also installed in the same environment for the later OCR work.

## Activate the environment

```bash
source ~/qwen3-env/bin/activate
```

The shell prompt should show the environment as active.

## Verify MLX-LM

```bash
python -c "import mlx_lm; print(mlx_lm.__version__)"
```

Expected result from the tested environment:

```text
0.31.3
```

To inspect the chat CLI:

```bash
mlx_lm.chat --help
```

## Model used

```text
prism-ml/Ternary-Bonsai-27B-mlx-2bit
```

The tested DSpark generation workflow was CLI-based. A representative command was:

```bash
mlx-dspark generate \
  --model prism-ml/Ternary-Bonsai-27B-mlx-2bit \
  --max-draft auto \
  --prompt "What is RAG and why is it useful?" \
  --max-new-tokens 100
```

For a code-generation test, the model was also prompted to generate a Python CSV analyzer. The generated code was saved as `csv_analyzer.py` and executed locally with:

```bash
python csv_analyzer.py
```

The expected successful result was:

```text
=== Student Analysis Summary ===
Total Students: 5
Average Marks: 87.60
Student with Highest Marks: Diana (95)
```

## CLI behavior observed

The current DSpark CLI used during testing was not an interactive chat UI. Each command starts a fresh generation process, so model initialization/setup contributes to the wall-clock time.

Qwen3-family-style reasoning output may also appear before the final answer. This was observed during the 27B demonstration and was treated as model reasoning output rather than an execution error.

## Performance observations

A representative generation reported approximately:

```text
Generation: 150 tokens
Generation speed: 7.1 tok/s
```

Another code-generation run reported:

```text
1001 tokens · 80.57s · 12.4 tok/s
```

The first number is generation speed; total wall-clock time can be higher because model loading/initialization happens before generation.

## Tamil OCR baseline benchmark

The 27B text model was used as a baseline in the same overall Tamil-page experiment. Ten page images were processed through the tested image/OCR workflow.

Results:

| Metric | Result |
|---|---:|
| Pages | 10 |
| Total | 1,197 sec / 19 min 57 sec |
| Average | 119.7 sec/page (~2 min/page) |
| Fastest | 106 sec (1 min 46 sec) |
| Slowest | 126 sec (2 min 6 sec) |

The OCR output was not error-free. Tamil character/word recognition errors were observed.

## Important reproducibility note

The benchmark is an end-to-end CLI measurement. Because the workflow starts the model command for each page, startup/model initialization is included in the measured wall-clock time. It should not be interpreted as pure token-generation latency.

## Troubleshooting notes

### `pdftoppm: command not found`

The original PDF-to-image conversion attempt used `pdftoppm`, but it was not installed on the test Mac. The workflow was changed to use Python/PyMuPDF for PDF page extraction.

### `fitz` / PyMuPDF missing

The first check produced:

```text
ModuleNotFoundError: No module named 'fitz'
```

PyMuPDF was then installed in the virtual environment before continuing with page extraction.

### `--no-think` / thinking control

The DSpark CLI help used during the test did not expose a `--no-think` option. The demonstration therefore left the model's reasoning output visible.
