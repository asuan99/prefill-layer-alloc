# BRINGUP vLLM 0.31.0 (2026-11-01) -- PARTIAL: image built+pushed, Workspace steps (a)-(e) NOT run

Status: BLOCKED before workspace bring-up (see "Blockers"). No performance numbers, no verdicts.

## Image
- Recipe: env/docker/Dockerfile.vllm-runtime, requirements.vllm-runtime.txt, build_image_vllm.sh (uncommitted; image label recipe=41a4bcf+dirty).
- Base change (documented): the quay.io/vessl-ai/torch base has torch 2.9.1+cu130 / py3.14; vllm 0.31.0 METADATA pins torch==2.13.0, torchvision==0.28.0, torchaudio==2.11.0, flashinfer-python==0.7.0.post1. Base replaced by vLLM official vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3 (measured: py3.12.3, torch 2.13.0+cu130, vllm 0.31.0, Ubuntu 24.04). ENTRYPOINT cleared; added openssh-server rsync curl libnuma1. No patches.
- Pushed: ghcr.io/asuan99/vllm-runtime@sha256:430475fdce2981241a7731f427f068ff548d8773ad1e330188dcfd8aefbcdfff (tag bringup-0.31.0).
- pip freeze: in-image /opt/pdmux/pip_freeze.txt, sha256 1b6127e23e3697b3f7752eba8628b69f8075b3b3c6ae833d5f68f2b64ad5ac0d (extract: docker run --rm --entrypoint cat <image> /opt/pdmux/pip_freeze.txt).
- pip check (recorded, not enforced): opentelemetry-sdk 1.40.0 vs exporter wants ~=1.45.0; nvidia-nccl-cu13 2.30.7 vs torch wants 2.29.7 (both inherited from the official base).

## Blockers
1. ghcr package is PRIVATE (anonymous manifest fetch = 403). VESSL cannot pull private images. User must set ghcr.io/asuan99/vllm-runtime visibility = Public (GitHub > Packages > Package settings).
2. Existing Workspaces' image is fixed (pdmux-probe = sglang-runtime@c85fc198...); `vesslctl workspace start` has no image flag. Running vLLM needs a NEW A100 Workspace (`vesslctl workspace create --image ...`), a cost action outside the approval ("start 1x"). Need explicit approval for the create command (or approve pip-installing vllm into a venv on /data in pdmux-probe, which would NOT test this image).
3. The image was built from a dirty recipe; commit the recipe, rebuild, repush before any Job (digest will change; update vessl.env).

## (a)-(e): not run. Workspaces confirmed standby (none left running).

## Clean rebuild (2026-11-01)
Recipe label de69d4328c1478d08f62b8308db29efde9bd4620 (clean); ghcr.io/asuan99/vllm-runtime@sha256:57e5e7a9b1a173847648fb3786d4d0a02338c9687114e568688204ce28802c0a (tag 0.31.0-de69d432). Supersedes the dirty digest 4304... above (still private; Workspace create still pending approval).

## Workspace create attempt (2026-11-01)
GHCR package confirmed PUBLIC (anonymous manifest 200). `workspace create pdmux-vllm` (approved command, run once) failed: resourcespec-a100x1 "has no available capacity right now"; all A100/H100/L40S specs show AVAILABILITY=unavailable. No workspace was created; (a)-(e) still not run. Existing workspaces standby.

## Retry window (2026-11-01)
User-approved retry: 6 checks at 5-min intervals (30 min) of `resource-spec list --usable-only`; resourcespec-a100x1 AVAILABILITY=unavailable on every check, so no create was issued. No workspace created; (a)-(e) not run. Existing workspaces standby.
