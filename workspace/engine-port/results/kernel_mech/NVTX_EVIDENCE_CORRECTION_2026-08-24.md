# NVTX 선행조건 — **근거 정정** (2026-08-24)

작성: 메인 세션. 계기: `audit_stage0pp_2026-08-23/VERDICT.md` 차단 **E9**.
GPU 지출 **0**. **새 성능 판정 0건.** 이 문서는 *측정 결과*가 아니라 *근거의 정정*이다.

---

## 1. 무엇이 잘못 쓰였나

kernel_mech rev5 이후 **6개 문서**가 같은 문장을 상속했다:

> "`pdmux.decode_step` NVTX 방출이 엔진에 없다 — **`grep -rn nvtx src/` = 0건**"

이 grep은 **틀린 트리를 봤다**.

| 트리 | 정체 | `grep -rln nvtx` |
|---|---|---|
| `prefill-layer-alloc/workspace/engine-port/src/` | PD-mux **오버레이**(sync 스크립트가 설치하는 패치 소스) | **0건** ✓ |
| `sglang_engine_dev/python/sglang/srt/` | ★**실제로 도는 엔진**(editable tree) | ★**4건** |

실제 엔진에 있는 것(2026-08-24 실측):

- `server_args.py:616` `enable_layerwise_nvtx_marker: bool = False` · `:5384` `--enable-layerwise-nvtx-marker`
- `utils/nvtx_pytorch_hooks.py` — `PytHooks`
- `model_executor/model_runner.py:1196` — 플래그가 켜지면 `PytHooks().register_hooks(self.model, module_prefix="model")`
- `batch_overlap/operations.py`

⇒ **"엔진에 NVTX가 0건"은 거짓이다.** 저장소에는 NVTX 계측이 **이미 배선돼 있고 CLI 플래그까지 있다**.

## 2. 그런데 결론은 살아남는다 — **다른 이유로**

`grep`이 답하려던 명제는 *"스텝 경계를 주는 `pdmux.decode_step` 마커가 있는가"* 였고, 그 답은
**여전히 아니오**다. 근거는 트리 개수가 아니라 **기전 두 가지**다:

1. ★**모듈 forward hook이다** — `nvtx_pytorch_hooks.py:286-287`은
   `module.register_forward_pre_hook` / `register_forward_hook`을 단다. cudagraph **replay**
   경로에서는 host 쪽 `Module.forward`가 실행되지 않으므로 **스텝마다 발화하지 않는다**.
2. ★**layerwise 마커다** — 이름 그대로 **레이어 단위**이지 **스텝 경계**가 아니다. 이 설계가
   요구한 결정량(`boundary(k)`)은 레이어 경계에서 복원되지 않는다.

⇒ **결론(=쓸 수 있는 NVTX 스텝 경계가 없다)은 유지**. **근거(=`grep` 0건)는 무효, 인용 금지.**

★ 교훈: *결론이 우연히 살아있는 것*과 *근거가 타당한 것*은 다르다. 이 문장은 5개 판본을 가로질러
**상속**됐고 아무도 트리를 확인하지 않았다(감사 E9가 잡음). [[deconfound-measurement-lessons]]
게이트 #31("수입한 보조 수치는 basis 검증하라")의 **grep 층 변종**.

## 3. ★ 더 큰 귀결 — **선행조건 자체가 불필요할 수 있다** (감사 E1-(a))

`model_executor/cuda_graph_runner.py:1155-1161`:

```python
if self.enable_pdmux:
    graph_key = f"{get_current_stream_idx()}_{self.bs}"
else:
    graph_key = self.bs
self.graphs[graph_key].replay()
```

cudagraph-ON decode 스텝은 **정확히 `replay()` 호출 1회**다. 그러면 CUPTI의 graph-launch
row **하나**가 다음 둘을 **동시에** 준다:

- **스텝 경계** — launch의 host timestamp
- **`K_set(k)` 소속** — 그 launch의 `correlationId`(+ `--cuda-graph-trace=node`의 `graphNodeId`)

⇒ **NVTX가 전혀 없어도 `boundary(k)`와 `K_set(k)`를 둘 다 얻을 수 있다.**

★ 이것이 사실이면 rev6·rev7·rev8을 가로질러 이 트랙을 막아 온 **"엔진 핫패스 패치 + manifest +
correctness gate(engine-porter 소관)" 선행조건이 사라진다**. 예산에서 가장 큰 항목이자
**3개 판본이 공유한 유일한 구조적 차단**이었다.

## 4. ★ 쓰면 안 되는 문장

- ✗ *"엔진에 NVTX가 없다"* / *"`grep -rn nvtx src/` = 0건"* — **거짓**. 어느 문서에서도 재인용 금지.
- ✗ *"NVTX 선행조건이 사라졌다"* — §3은 **코드 독해에 근거한 가설**이다. nsys가 이 기판에서
  green-context 스트림의 graph-launch row를 **실제로** 방출하는지는 **미측정**이며, 그것이
  바로 Stage 0‴ **Q2**가 사려는 것이다. (선례: P1 프로브에서 CUPTI×green-context가
  **문서화된 시그니처로 실패**했다 — 도구가 될 것이라 가정하면 안 된다.)
- ✗ *"kernel_mech 트랙이 열렸다"* — rev7·rev8 **`NO-GO` 불변**(B1–B5·B7 / C1–C10 전부 잔존).
  이 정정은 **차단 하나의 근거를 무효화**했을 뿐, 다른 차단을 하나도 닫지 않는다.

## 5. 반영

rev5·rev6·rev7·rev8·B6 사전등록·Stage 0″ 사전등록 **6개 문서**에 정정 배너를 삽입하고 이 문서를
가리킨다. 원 문장은 **삭제하지 않고 취소선 처리**해 이력을 보존한다(정본 규율: 폐기는 격리이지 삭제가 아니다).
