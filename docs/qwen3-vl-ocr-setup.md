# Qwen3-VL OCR Setup on Apple Silicon

## Scope

This document records the tested Qwen3-VL vision/OCR workflow used for Tamil book-page digitization on the same M4 Mac with 16 GB unified memory.

## Environment

Activate the existing environment:

```bash
source ~/qwen3-env/bin/activate
```

Verify MLX-VLM:

```bash
python -c "import mlx_vlm; print('mlx-vlm OK')"
```

The tested environment returned:

```text
mlx-vlm OK
```

The MLX-LM version was also verified as `0.31.3`.

Inspect the generator CLI:

```bash
python -m mlx_vlm.generate --help
```

## Model

The tested model was:

```text
mlx-community/Qwen3-VL-8B-Instruct-3bit
```

The first Page 1 run downloaded/reconstructed approximately 4.75 GB of model data. The first run took 12 min 51.39 sec end-to-end, but that time included the model download and therefore was not used as the inference benchmark.

## Input preparation

The Tamil source was a 10-page PDF. The pages were converted into individual images and stored under:

```text
~/Downloads/tamil_ocr_pages/
```

with files such as:

```text
page_01.jpg
page_02.jpg
...
page_10.jpg
```

## OCR prompt

The same OCR instruction was used for the benchmark:

```text
Perform OCR on this Tamil book page. Transcribe ALL visible printed Tamil text as accurately as possible. Output ONLY the Tamil transcription. Do not translate, summarize, explain, or invent text. Preserve paragraph breaks, punctuation, and word boundaries. Pay close attention to Tamil characters, vowel signs, consonant-vowel combinations, and ligatures.
```

## Single-page command

```bash
time python -m mlx_vlm.generate \
  --model mlx-community/Qwen3-VL-8B-Instruct-3bit \
  --image ~/Downloads/tamil_ocr_pages/page_01.jpg \
  --prompt "Perform OCR on this Tamil book page. Transcribe ALL visible printed Tamil text as accurately as possible. Output ONLY the Tamil transcription. Do not translate, summarize, explain, or invent text. Preserve paragraph breaks, punctuation, and word boundaries. Pay close attention to Tamil characters, vowel signs, consonant-vowel combinations, and ligatures." \
  --max-tokens 800 \
  --temperature 0
```

After the model was cached, the repeated Page 1 test completed in:

```text
54.085 seconds
```

This is the useful Page 1 timing; the earlier 12:51 timing included model download.

## 10-page benchmark

The same ten images and OCR prompt were automated and the model output was saved as one text file per page.

Results:

| Metric | Qwen3-VL-8B-Instruct-3bit |
|---|---:|
| Pages | 10 |
| Total | 556 sec / 9 min 16 sec |
| Average | 55.6 sec/page |
| Fastest | 53 sec |
| Slowest | 58 sec |

Compared with the previous 10-page Ternary-Bonsai-27B baseline:

| Metric | Ternary-Bonsai-27B | Qwen3-VL-8B |
|---|---:|---:|
| Total | 19 min 57 sec | **9 min 16 sec** |
| Average/page | ~119.7 sec | **55.6 sec** |
| Fastest | 106 sec | **53 sec** |
| Slowest | 126 sec | **58 sec** |

Qwen3-VL completed the same ten-page workflow about 2.15x faster, or roughly 53% less wall-clock time.

## OCR quality observations

The model successfully produced Tamil text from the page images, but the output was not publication-ready. Observed issues included:

- incorrect Tamil characters
- incorrect word predictions
- missing or altered text
- repeated text in some sections

In particular, some generated sections showed token repetition. Therefore the result should be described as a successful OCR/vision demonstration with significant recognition errors, rather than clean book digitization.

## Saving OCR output

The benchmark stored individual results under:

```text
~/Downloads/qwen3vl_tamil_results/
```

Example:

```text
page_01.txt
page_02.txt
...
page_10.txt
timings.csv
```

A combined Markdown file can be created with:

```bash
cat ~/Downloads/qwen3vl_tamil_results/page_*.txt > ~/Downloads/tamil_qwen3vl_10pages.md
```

## PNG/JPEG + Markdown package

The OCR output can be packaged with the source page images for review:

```bash
mkdir -p ~/Downloads/tamil_qwen3vl_package
cp ~/Downloads/tamil_qwen3vl_10pages.md ~/Downloads/tamil_qwen3vl_package/
cp ~/Downloads/tamil_ocr_pages/page_*.jpg ~/Downloads/tamil_qwen3vl_package/
cd ~/Downloads
zip -r tamil_qwen3vl_ocr_10pages.zip tamil_qwen3vl_package
```

## PNG to Markdown demonstration

A single image can also be passed to Qwen3-VL with a prompt requesting Markdown-only output. The tested workflow created:

```text
~/Downloads/tamil_page.md
```

The Markdown conversion technically succeeded, but the OCR limitations above also apply to this output.

## Notes on timing

The first model invocation includes model download/reconstruction and should not be compared directly with subsequent cached inference runs. The 10-page benchmark was performed after the model was available locally, and its timings represent the same end-to-end per-page CLI workflow used for the comparison.
