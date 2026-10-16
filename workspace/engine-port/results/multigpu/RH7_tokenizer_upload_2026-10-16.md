# R-H7 tokenizer check + token-file upload (2026-10-16) -- committed with the rev12 seal (REGISTRATION_rev12_SEAL.txt)

Workspace: pdmux-build (wsp-fbmj36cisl56, CPU), started then paused; final state standby. pdmux-probe untouched (standby).

## 1. Tokenizer check: MATCH
Path: /data/hf/hub/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2-Base/snapshots/dc0661c829b14e5b9246c05cfa89094a0875e052/ (refs/main = same revision = mg_const.MODEL_REVISION)
- tokenizer.json sha256 = 3277c00fe5fb3963b3cb7c07b7f183722d2af4d775a4aea7cfb3684d7cccbc2f == mg_const.X_TOKENIZER_SHA256
- tokenizer_config.json sha256 = f839850bc0a24d71e51b636635e02a5f363824c48a6ff209913a4d090d2add3c (informational)
- special_tokens_map.json sha256 = 6fa06efa2785e450051989a6f8fb4416b10149ded485ddd3f127a40734f5cfd0 (informational)

## 2. Upload
Expected path (m3_job.sh XTOK="$DATA/$(K X_TOKENS_REL)", X_TOKENS_REL = datasets/arxsum_x/arxsum_x_tokens_v1.bin): /data/datasets/arxsum_x/arxsum_x_tokens_v1.bin
Local sha (re-checked) = remote sha (object vol /io/staging/rh7/ and /data copy) = 4c32f1af8eff07d7d0ad39006423163895181381c0d83a73a5f0f5b1ce76888f == mg_const.X_TOKENS_SHA256; 168330617 bytes.
Commands: vesslctl volume upload objvol-9vgmwmerzwd3 <dir> --remote-prefix staging/rh7/; ssh -p 31439 root@betelgeuse.cloud.vessl.ai: sha256sum, mkdir -p /data/datasets/arxsum_x, cp, sync, sha256sum; vesslctl workspace pause wsp-fbmj36cisl56.
Leftover: /io/staging/rh7/arxsum_x_tokens_v1.bin (staging copy, may be deleted).
