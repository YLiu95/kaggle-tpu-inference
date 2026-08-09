# Validation record

## Environment

- Date: 2026-08-09 UTC
- Platform: Kaggle TPU VM
- Detected accelerator: TPU v5e-8 (8 chips, 16 GiB HBM/chip)
- Host: 96 logical CPUs, approximately 377 GiB RAM
- Python: 3.12.13

## Pinned engine dependencies

- `vllm-tpu==0.26.0`
- `tpu-inference==0.26.0`
- `uv==0.12.3`

## Validation stages

- [ ] Lightweight unit tests
- [ ] Clean local `main` pushed and remote SHA confirmed
- [ ] Binary-only vLLM TPU setup
- [ ] Engine import/version validation without model loading
- [ ] All eight TPUs confirmed through runtime telemetry
- [ ] Gemma 4 model loaded with calculated context
- [ ] Streamed reasoning and response
- [ ] CPU affinity/priority and one-sequence limit confirmed
- [ ] CSV and `/kaggle/working` mirror confirmed

## Successful command

Pending end-to-end validation.

