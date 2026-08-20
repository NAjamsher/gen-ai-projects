# Qwen OCR Benchmark Summary

## Test environment

Apple M4 Mac with 16 GB unified memory. The same ten Tamil book-page images were used for the OCR comparison.

## Results

| Metric | Ternary-Bonsai-27B | Qwen3-VL-8B-Instruct-3bit |
|---|---:|---:|
| Pages | 10 | 10 |
| Total time | 19 min 57 sec | **9 min 16 sec** |
| Average/page | 119.7 sec | **55.6 sec** |
| Fastest | 106 sec | **53 sec** |
| Slowest | 126 sec | **58 sec** |

Qwen3-VL was approximately 2.15x faster in this tested workflow.

## Quality

Both workflows produced Tamil OCR with recognition errors. Qwen3-VL was significantly faster, but its output still contained incorrect characters/words and repeated text in some sections. Accuracy should therefore be described qualitatively unless a manually verified character/word-level ground truth is created.

## Interpretation

The benchmark demonstrates a substantial speed advantage for Qwen3-VL on this M4 16 GB setup. It does not, by itself, establish that Qwen3-VL has higher OCR accuracy than the 27B baseline.
