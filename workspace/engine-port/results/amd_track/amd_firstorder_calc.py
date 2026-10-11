#!/usr/bin/env python3
"""AMD(Attention–Mamba Disaggregation) 1차 계산 — 바이트·횟수·전송 시간의 **가정 하 가산 비용**만 (2026-10-11, rev2 감사 반영).

무엇을 계산하는가 (전부 **구조 상수 × 대역폭·지연 가정**이다 — 커널 시간은 한 줄도 쓰지 않는다):
  (a) L-B 레이아웃(GPU-A = prefill + decode-attention/KV, GPU-B = decode mamba+MLP/state)에서
      decode step당 GPU 간 crossing 횟수·바이트·가산 지연  (B ∈ {8,16,32,48}, 기판 3종, L0 3종)
  (b) P→D 핸드오프 바이트·시간: KV(attention 층만, L-스케일) vs mamba state(B-스케일, L-불변)
      — L-B는 state만, PD-disagg 대조군은 KV+state (ctx ∈ {1k,4k,8k,16k})
  (c) wall-time 합산 비율 R = (요청당 KV 전송 절감) / (출력 토큰 수 × step당 crossing 가산 비용)
      — ★SLO 지표가 아니다(KV 절감은 TTFT 성분, crossing은 ITL 성분). §3에 (ctx, OUT, 기판, L0) 뒤집힘 격자.
★"상한"이 아니다: `L0 + bytes/BW`는 이상화된 가산 비용이며, A가 prefill 뒤에서 기다리는 시간·sync/launch·
  이동한 attention 계산을 빼면 실제가 더 클 수 있고, overlap이 있으면 더 작을 수 있다(claims-auditor 2026-10-11 (iv)).
입력 출처 라벨: [YAML] workspace/shared/configs/models.yaml (config_status: verified) ·
  [CANON] PROJECT_STATUS.md ~2290 (Nano-9B mamba 136.90 MiB/req = ssm+conv 합, KV cell 16 KiB/token) ·
  [BOOT] 서버 부팅 배너 "Mamba Cache is allocated … conv_state size: X GB, ssm_state size: Y GB" — 단위는 GiB
  (memory_pool.py:74 `GB = 1024**3`), 슬롯 수는 size+1 = **49**(memory_pool.py:255,269) ·
  [ASSUME] 대역폭·지연 가정(아래 ASSUMPTIONS; L0 65/130 µs는 SQD arXiv 2609.13134의 **B200** one-sided put 실측
  64.6–130.7 µs를 A100 가정으로 이식한 값이지 A100 실측이 아니다). 인용 금지 열(r0c knee 절대값 등)은 쓰지 않는다.
사용:  python3 -I amd_firstorder_calc.py            # md를 stdout에
       python3 -I amd_firstorder_calc.py --selftest # 기록값 재현 검사(허용오차 0.5%)
"""
import sys, argparse

BF16 = 2          # bytes/elem, 활성화·KV
FP32 = 4          # bytes/elem, mamba temporal state (sglang HybridLinearKVPool, 배너 역산과 0.01% 내 일치)
MiB = 1024 ** 2
GiB = 1024 ** 3
POOL_SLOTS = 49   # max_mamba_cache_size 48 + 1 (memory_pool.py:255,269)

# ---------------------------------------------------------------- 모델 상수
# n_attn_calls: decode step당 attention 모듈 호출 수 = L-B crossing 쌍 수
# n_kv_layers : KV를 보유하는 attention 호출 수(Zamba2 공유 블록은 block_idx별 RadixAttention·별도 layer_id → 호출마다 KV, zamba2.py:223-235)
# Zamba2 공유 블록 입력은 concat[hidden, original](zamba2.py:341) — A가 embedding을 들고 있으면 crossing 바이트 증가 없음(L0 지배라 무시)
MODELS = {
    "nemotron_h_8b": dict(label="Nemotron-H-8B [YAML]", hidden=4096, n_layers=52,
        n_attn_calls=4, n_kv_layers=4, kv_heads=8, attn_head_dim=128,
        n_mamba=24, ssm_heads=128, ssm_head_dim=64, d_state=128,
        boot_ssm_pool_gib=4.59, boot_conv_pool_gib=0.07, state_canon_mib=None),
    "nemotron_nano_9b": dict(label="Nemotron-Nano-9B-v2 [CANON]", hidden=4480, n_layers=56,
        n_attn_calls=4, n_kv_layers=4, kv_heads=None, attn_head_dim=128,
        n_mamba=27, ssm_heads=None, ssm_head_dim=None, d_state=None,
        boot_ssm_pool_gib=6.46, boot_conv_pool_gib=0.09, state_canon_mib=136.90,   # canon 값은 ssm+conv 합
        kv_cell_bytes=16 * 1024),   # [CANON] cell_size 16 KiB/token (K+V 합)
    "zamba2_2.7b": dict(label="Zamba2-2.7B [YAML]", hidden=2560, n_layers=54,
        n_attn_calls=9, n_kv_layers=9, kv_heads=32, attn_head_dim=160,
        n_mamba=54, ssm_heads=80, ssm_head_dim=64, d_state=64,
        boot_ssm_pool_gib=3.23, boot_conv_pool_gib=0.08, state_canon_mib=None),
    "zamba2_7b": dict(label="Zamba2-7B [YAML]", hidden=3584, n_layers=81,
        n_attn_calls=13, n_kv_layers=13, kv_heads=32, attn_head_dim=224,
        n_mamba=81, ssm_heads=112, ssm_head_dim=64, d_state=64,
        boot_ssm_pool_gib=6.78, boot_conv_pool_gib=0.16, state_canon_mib=None),
    "granite_4_h_micro": dict(label="Granite-4.0-h-micro [BOOT/WORK]", hidden=2048, n_layers=40,
        n_attn_calls=4, n_kv_layers=4, kv_heads=None, attn_head_dim=None,
        n_mamba=36, ssm_heads=None, ssm_head_dim=None, d_state=None,
        boot_ssm_pool_gib=3.45, boot_conv_pool_gib=0.04, state_canon_mib=None,
        kv_cell_bytes=8 * 1024),    # [WORK] K 4 KB/token 역산 → K+V 8 KB (Explore 2026-10-11)
}
EXCLUDED = {"falcon_h1": "매 층 attention∥mamba 병렬(spatial) — 층 단위 분리 자체가 정의되지 않음"}

# ---------------------------------------------------------------- 가정 [ASSUME]
FABRICS = {  # 단방향 실효 대역폭 가정
    "NVLink(A100 SXM, 단방향 300 GB/s 사양)": 300e9,
    "NVLink(실효 보수 200 GB/s)":              200e9,
    "PCIe Gen4 x16(실효 25 GB/s)":             25e9,
}
L0_US = {"10us(커널 p2p 낙관)": 10.0,
         "65us[ASSUME: SQD B200 one-sided put 하한 이식]": 65.0,
         "130us[ASSUME: SQD B200 상한 이식]": 130.0}
BATCHES = [8, 16, 32, 48]
CTXS = [1024, 4096, 8192, 16384]
OUT_TOKENS = 256          # §1–§2 기본 가정(검증 안 된 편의값) — §3 격자가 64/256/1024를 돈다
TPOT_YARDSTICK_MS = 12.0  # [참고] cudagraph ON Zamba2-2.7B rate2 TPOT≈12 ms(results/cudagraph_probe) — **모델 공통 환산 눈금**, 모델별 TPOT 아님

def kv_bytes_per_token(m):
    if m.get("kv_cell_bytes"):
        return m["kv_cell_bytes"]
    return 2 * m["n_kv_layers"] * m["kv_heads"] * m["attn_head_dim"] * BF16

def state_bytes_per_req(m):
    """(ssm, conv) bytes. canon 값은 합이므로 conv=0으로 돌려 이중계상을 막는다."""
    if m.get("state_canon_mib"):
        return m["state_canon_mib"] * MiB, 0.0
    if m.get("ssm_heads"):
        ssm = m["n_mamba"] * m["ssm_heads"] * m["ssm_head_dim"] * m["d_state"] * FP32
    else:
        ssm = m["boot_ssm_pool_gib"] * GiB / POOL_SLOTS
    conv = m["boot_conv_pool_gib"] * GiB / POOL_SLOTS
    return ssm, conv

def banner_state_mib(m):
    return (m["boot_ssm_pool_gib"] + m["boot_conv_pool_gib"]) * GiB / POOL_SLOTS / MiB

def crossing_ms(m, B, bw, l0_us):
    """step당 가산 지연(ms) = 2×attention 호출 × (L0 + B·hidden·bf16 / BW). 경계가 아니라 가정 하 가산 비용."""
    n = 2 * m["n_attn_calls"]
    return n * (l0_us + B * m["hidden"] * BF16 / bw * 1e6) / 1e3

def crossing_table(m):
    rows = []
    n_cross = 2 * m["n_attn_calls"]
    for B in BATCHES:
        bytes_per_cross = B * m["hidden"] * BF16
        for fab, bw in FABRICS.items():
            for l0n, l0 in L0_US.items():
                t_ms = crossing_ms(m, B, bw, l0)
                rows.append((B, fab, l0n, n_cross, bytes_per_cross, t_ms, t_ms / TPOT_YARDSTICK_MS))
    return rows

def handoff_table(m):
    kvt = kv_bytes_per_token(m)
    ssm, conv = state_bytes_per_req(m)
    rows = []
    for L in CTXS:
        kv = kvt * L
        for fab, bw in FABRICS.items():
            rows.append((L, fab, kv, ssm + conv, kv / bw * 1e3, (ssm + conv) / bw * 1e3, (kv + ssm + conv) / bw * 1e3))
    return rows

def R(m, L, B, bw, l0_us, out_tokens):
    kv_ms = kv_bytes_per_token(m) * L / bw * 1e3
    c_ms = crossing_ms(m, B, bw, l0_us)
    return kv_ms, c_ms, kv_ms / (out_tokens * c_ms)

FLIP_GRID = dict(ctx=[4096, 8192, 16384], out=[64, 256, 1024],
                 fab=[("NVLink 200", 200e9), ("PCIe 25", 25e9)], l0=[10.0, 65.0])

def fmt_bytes(b):
    return f"{b/MiB:.1f} MiB" if b >= MiB / 10 else f"{b/1024:.1f} KiB"

def render():
    out = []
    P = out.append
    P("# AMD 1차 계산표 (2026-10-11, rev2) — 바이트·횟수·전송 시간의 가정 하 가산 비용\n")
    P("> 생성: `python3 -I amd_firstorder_calc.py`. **커널 시간 0건 사용.** 모든 지연은 `L0 + bytes/BW`의 **가산 비용(경계 아님)**이며")
    P("> 기판은 미실측(V0 프로브 전) — NVLink/PCIe를 병기한다. L0 65/130 µs는 SQD의 B200 실측을 A100 가정으로 이식한 값[ASSUME].")
    P("> 비율 열의 눈금 TPOT 12 ms는 cudagraph ON Zamba2-2.7B 운영점(results/cudagraph_probe)에서 빌린 **모델 공통 환산 눈금**이지")
    P("> 모델별 TPOT도 예측도 아니다. 부팅 배너 단위는 GiB, 슬롯 수는 49(size+1).\n")
    P("## 0. 입력 상수\n")
    P("| 모델 | hidden | attn 호출/step | mamba 층 | KV (K+V) B/token | state/req (ssm+conv) | 출처 vs 부팅배너 (ssm+conv)/49 |")
    P("|---|---:|---:|---:|---:|---:|---|")
    for k, m in MODELS.items():
        ssm, conv = state_bytes_per_req(m)
        src = "canon" if m.get("state_canon_mib") else ("분석식+배너conv" if m.get("ssm_heads") else "배너")
        P(f"| {m['label']} | {m['hidden']} | {m['n_attn_calls']} | {m['n_mamba']} | {kv_bytes_per_token(m):,} | "
          f"{fmt_bytes(ssm+conv)} | {src} {(ssm+conv)/MiB:.2f} vs {banner_state_mib(m):.2f} MiB |")
    for k, why in EXCLUDED.items():
        P(f"| Falcon-H1 | — | — | — | — | — | **제외**: {why} |")
    P("")
    P("## 1. L-B decode step당 crossing (attention 호출마다 왕복 2회, hidden 텐서 1개 기준; residual까지 보내면 바이트 ×2)\n")
    for k, m in MODELS.items():
        P(f"### {m['label']}\n")
        P("| B | 기판 | L0 | crossing/step | bytes/crossing | 가산 지연/step | ÷TPOT 12 ms 눈금 |")
        P("|---:|---|---|---:|---:|---:|---:|")
        for B, fab, l0n, n, by, t_ms, ratio in crossing_table(m):
            P(f"| {B} | {fab} | {l0n} | {n} | {fmt_bytes(by)} | {t_ms:.3f} ms | {ratio*100:.1f}% |")
        P("")
    P("## 2. P→D 핸드오프: KV(attention 층, L-스케일) vs mamba state(L-불변) — L-B는 state만 건너고 PD-disagg 대조군은 둘 다 건넌다\n")
    for k, m in MODELS.items():
        P(f"### {m['label']}\n")
        P("| ctx L | 기판 | KV/req | state/req | KV 전송 | state 전송 | KV+state(PD-disagg) |")
        P("|---:|---|---:|---:|---:|---:|---:|")
        for L, fab, kv, st, tkv, tst, tboth in handoff_table(m):
            P(f"| {L} | {fab} | {fmt_bytes(kv)} | {fmt_bytes(st)} | {tkv:.3f} ms | {tst:.3f} ms | {tboth:.3f} ms |")
        P("")
    P("## 3. wall-time 합산 비율 R — **SLO 지표 아님**, 뒤집힘 격자\n")
    P("R = (요청당 KV 전송 절감 ms) / (출력 토큰 수 × step당 crossing 가산 ms). KV 절감은 TTFT 쪽 1회, crossing은 ITL 쪽 매 step이라")
    P("같은 ms로 더한 R은 goodput과 무관하다. R은 **ctx가 길고·출력이 짧고·기판이 느리고·L0가 작을수록** 1을 넘는다(아래 격자, B=32).")
    P("어느 축이 SLO 위에서 이기는지는 V4 직접 측정만 답한다.\n")
    P("| 모델 | ctx | 출력 | 기판 | L0 | KV 절감/req | crossing/step | R |")
    P("|---|---:|---:|---|---:|---:|---:|---:|")
    for k, m in MODELS.items():
        for L in FLIP_GRID["ctx"]:
            for o in FLIP_GRID["out"]:
                for fn, bw in FLIP_GRID["fab"]:
                    for l0 in FLIP_GRID["l0"]:
                        kv_ms, c_ms, r = R(m, L, 32, bw, l0, o)
                        mark = "**" if r > 1 else ""
                        P(f"| {m['label']} | {L} | {o} | {fn} | {l0:.0f} | {kv_ms:.3f} ms | {c_ms:.3f} ms | {mark}{r:.3f}{mark} |")
    P("")
    P("굵은 R(>1)은 합산 wall-time에서 L-B가 PD-disagg보다 덜 옮기는 셀이다 — 그래도 ITL 축만 보면 §1의 '÷TPOT' 열이 가산 비용이고,")
    P("TTFT 축만 보면 §2의 전송 열이 가산 비용이다. 둘은 goodput의 서로 다른 술어에 들어간다.")
    return "\n".join(out) + "\n"

def selftest():
    z = MODELS["zamba2_2.7b"]
    assert kv_bytes_per_token(z) == 2 * 9 * 32 * 160 * 2 == 184_320, kv_bytes_per_token(z)      # K 92,160 B/token [WORK 역산]과 일치
    n = MODELS["nemotron_h_8b"]
    assert kv_bytes_per_token(n) == 2 * 4 * 8 * 128 * 2 == 16_384                                 # K 8 KB/token [WORK]
    assert kv_bytes_per_token(MODELS["zamba2_7b"]) == 2 * 13 * 32 * 224 * 2 == 372_736
    # 분석식 ssm vs 배너 ssm/49: 0.5% 이내(허용오차 8%는 GiB/GB·+1 슬롯을 못 가렸다 — 감사 (ii))
    for k in ("zamba2_2.7b", "zamba2_7b", "nemotron_h_8b"):
        m = MODELS[k]; ssm, _ = state_bytes_per_req(m)
        banner_ssm = m["boot_ssm_pool_gib"] * GiB / POOL_SLOTS
        assert abs(ssm / banner_ssm - 1) < 0.005, (k, ssm / banner_ssm)
    nano = MODELS["nemotron_nano_9b"]; s, c = state_bytes_per_req(nano)
    assert c == 0.0 and abs(s / MiB - 136.90) < 1e-6 and abs(banner_state_mib(nano) / 136.90 - 1) < 0.005  # canon = ssm+conv 합
    # 변이 대조: 슬롯 48로 나누면 검사가 실패해야 한다(검사가 분모를 실제로 본다)
    wrong = z["boot_ssm_pool_gib"] * GiB / 48
    assert abs(state_bytes_per_req(z)[0] / wrong - 1) >= 0.005
    # 결정성·구조
    assert render() == render()
    assert crossing_table(z)[0][3] == 18 and crossing_table(n)[0][3] == 8
    assert abs(crossing_ms(n, 32, 200e9, 65.0) - 8 * (65 + 32 * 4096 * 2 / 200e9 * 1e6) / 1e3) < 1e-12
    # 뒤집힘 격자에 R>1 셀이 실제로 존재한다(감사 D1의 사례: Zamba2-2.7B, PCIe, 10 µs, ctx 8192, out 64)
    assert R(z, 8192, 32, 25e9, 10.0, 64)[2] > 1
    print("SELFTEST PASS")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        sys.stdout.write(render())
