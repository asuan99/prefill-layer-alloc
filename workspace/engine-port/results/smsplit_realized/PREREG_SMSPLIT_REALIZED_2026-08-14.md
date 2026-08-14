# 사전등록 — E-3 realized SM count 판독 (2026-08-14)

**성격**: 서술적 계측 배관 프로브. **성능 판정 0건.** 비용 ≈수 초 GPU(+부팅/임포트).
**대상**: `torch.ops.sgl_kernel.create_greenctx_stream_by_value`가 계산하고
`sgl_kernel.spatial`의 Python 래퍼가 버리는 realized SM 수 2개.

이 문서는 **제출 전에 고정**된다. 하네스는 같은 디렉터리의
`smsplit_realized_probe.py` · `smsplit_realized.sbatch`.

---

## 0. blind 선언

이 결정량은 **이 저장소에서 한 번도 산출된 적이 없다**(래퍼가 `res[0]`/`res[1]`만
읽고 그 뒤를 버리므로 구조적으로 불가능했다). 작성자는 값을 본 적 없다.

작성자가 **실행 전에 아는 것**은 코드 사실뿐이다:
`external/sglang-latest/sgl-kernel/csrc/spatial/greenctx_stream.cu:84-97`이
`resources[0].sm.smCount` / `resources[1].sm.smCount`를 벡터 인덱스 2·3으로
반환한다. **다만 venv의 컴파일된 wheel(`sgl_kernel/spatial_ops.abi3.so`)이 그
소스로 빌드됐는지는 GPU 없이 확정할 수 없었다** — 그래서 프로브는 이를 가정하지
않고 `len(res)`를 기록해 판정에 쓴다(§2 결과 3).

## 1. 이 프로브가 답하는 질문

> 엔진이 `(74, 34)`를 요청했을 때 드라이버는 `(74, 34)`를 만들었는가?

정본의 모든 파티션 문장은 현재 **요청값**이다. Stage 0 D108 사건의 교훈은
"pin은 target이 아니라 realized로 검증하라"인데, SM **개수** 층의 realized는
지금까지 한 번도 읽힌 적이 없다.

## 2. 사전등록 결과 분기 (전부 결과이며, 어느 것도 "실패"가 아니다)

| # | 관측 | 판정 문자열 | 함의 |
|---|---|---|---|
| 1 | 전 대상에서 `realized == requested` | `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION` | 반올림 없음. 정본 caveat의 "green-context 반올림 미확인" 부분이 **드라이버 보고 층에서** 해소 |
| 2 | 어느 대상에서든 `realized != requested` | `DRIVER_ROUNDS_REQUEST` | `g2s_analyze.py:1488`의 `eps = log(mt/mc)/log(108.0/34.0)` 분모가 **틀렸다**. 정정 대상으로 등재(자동 수정 아님 — 별건 사전등록 소관) |
| 3 | `len(res) < 4` | `COUNTS_NOT_EXPOSED_BY_BUILD` | 설치된 wheel이 옛 빌드. **이 경로는 재빌드 없이는 닫히지 않는다**. 계측 실패이지 파티션에 대한 결과가 아니다 |
| 4 | 예외/판독 불능 | `PROBE_FAILED` | 계측 실패. **파티션에 대해 아무것도 주장하지 않는다** |

부수 산출(사전등록): `realized_a + realized_b < requested_a + requested_b`이면
**미할당 SM이 존재**한다. 이는 `%smid` R0 감사가 지목한 다섯 번째 세계
(`disjoint ∧ ∪ ⊊ D`)의 **개수 층 대응물**이며, 그 캠페인 재설계의 입력이 된다.

★ **결과 3·4를 결과 1·2로 접지 않는다.** 측정 실패를 게이트 실패(또는 통과)로
라벨링하는 것은 이 프로젝트의 서명 오류이고 아홉 번 재발했다(방법론 게이트 #21).
프로브는 판정을 exit code로 쓰지 않는다.

## 3. 게이트 #9 (항등식) 점검 — 실행 전 수행

- **대상 파티션을 하드코딩하지 않는다.** `(74, 34)`는 생산자
  `pdmux_context.divide_sm(total, (8,0), 2)`를 직접 호출해 얻는다. 검증하려는
  숫자를 프로브가 스스로 적어 넣으면 항등식에 가깝다(게이트 #9: 대조는 생산자에).
- **드라이버 자기보고로 드라이버 자기보고를 검증하지 않는다.** 이 프로브는
  `get_sm_available()`을 **총 SM 수 맥락 정보로만** 기록하고 판정에 넣지 않는다.
  판정은 `realized`(op 반환) vs `requested`(우리가 넘긴 인자) 대조 하나뿐이며,
  두 값의 출처가 다르다.
- **항등식 아님 확인**: `realized`는 `cuDevSmResourceSplitByCount` **이후**
  값이므로 `requested`와 다를 수 있다 — 실제로 다를 수 있다는 것이 이 프로브의
  존재 이유다. 결과가 사전에 제약돼 있지 않다.

## 4. 해석 제한 (전부 필수)

1. **이것은 드라이버 자기보고다.** "하드웨어가 그 SM들에서 실행했다"가 **아니다**.
   Stage 0 D108 의미의 realized는 id 집합 프로브(`%smid`) 소관이며 별도 캠페인이다.
   ⇒ **"실현 파티션을 측정했다"는 이 결과로 쓸 수 없다.**
2. **split은 2단**(`smA+smB`를 먼저 떼고 그것을 A/B로 분할)이라 반올림 지점이
   둘이다. `delta_a`·`delta_b`만으로 어느 단계의 반올림인지 귀속하지 않는다.
3. **성능 주장 0건.** 이 프로브는 시간을 재지 않는다.
4. **기존 캠페인 결과에 이 수치를 붙이지 않는다** — jobs 873944/874478/875293/
   877974/877756/877757의 판정문에 사후 부착 금지(`PREREG_GATE2S_2026-08-09.md`
   §8-3과 같은 규율). 분모 정정이 필요하면 **별건 사전등록**으로 한다.
5. **`(74,34)` 인용 금지 목록은 이 결과로 해제되지 않는다.** CONSENSUS §1-1의
   Gate 1 블록이 건 금지("실현 파티션을 **측정**했다" / "prefill 74 SM·decode
   34 SM에서 **실행**됐다")는 전부 유지된다.
6. **PD 분리 자체 귀속(Gate 2 본 질문)과 무관.** 한 눈금도 전진시키지 않는다.

## 5. 위생 / 환경

- **`sync_engine_tree.sh`를 돌리지 않는다.** 이 프로브는 엔진 거동을 전혀
  실행하지 않고, 필요한 `pdmux_context.py`는 sync가 설치하지 않는 파일이다
  (게이트 #33). sync는 기본 매니페스트 경로를 다시 쓰는데 E1-b/E1-c의 트립와이어가
  그 기준선에 걸려 있으므로 불필요한 부작용을 만들지 않는다.
- **매니페스트·dev 트리·`src/` 무변경.** 따라서 E1-b/E1-c 및 G1-a와 **제출 순서
  제약이 없다**.
- `$HOME` inode 쿼터(2026-08-14 기준 98,976/100,000)로 홈 밑 캐시 생성이
  `OSError: [Errno 122]`로 죽으므로 `TRITON_CACHE_DIR`·`TORCHINDUCTOR_CACHE_DIR`를
  스크래치로 강제한다.
- 게이트 #26(개정판): 이 프로브는 **신규 코드**이나 GPU 비용이 분 단위이며,
  같은 배치의 E-4·E-1과 **코드를 공유하지 않는다**(별도 디렉터리, 엔진 무변경).
  ⇒ 합산 트리거 대상이 아니다. 다만 그 판단 자체가 이번 개정이 겨냥한 논법
  형태이므로 **감사자 확인 대상으로 표시**한다.

## 6. 후속 (이 프로브가 여는 것 / 열지 않는 것)

- **연다**: `%smid` R0의 재설계 입력(다섯 번째 세계의 개수 층 증거),
  `log(108/34)` 분모 정정의 필요 여부 판정.
- **열지 않는다**: id 집합·disjointness(=`%smid` 고유 질문), 하드웨어 실행 층,
  성능·정책 주장 일체.

---

# Addendum — 2026-08-14 (원문 미수정)

## A1. 환경 사실 정정: 로그인 노드에 A100이 있다

§5와 하네스는 이 프로브가 컴퓨트 노드 할당을 필요로 한다고 전제했다. **틀렸다.**
`glogin01`에 `NVIDIA A100 80GB PCIe`가 있어 프로브가 로그인 노드에서 그대로 돌았고,
`torch.cuda.is_available()` 가드는 발화하지 않았다. ⇒ **이 결과의 SLURM 비용은 0이다.**

(이 사실 자체가 재사용 가치가 있다: green-context 원시함수처럼 **드라이버 층만
건드리는 프로브는 SLURM 없이 검증 가능**하다. 단 아래 A3의 노드 통제 한계가 따른다.)

## A2. 결과 (로그인 노드, `smsplit_realized_glogin01_2026-08-14.json`)

**판정 = `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`** (§2 결과 1).
`n_returned = 4` 전 대상 ⇒ §2 결과 3(`COUNTS_NOT_EXPOSED_BY_BUILD`)은 배제됐다:
설치된 wheel은 realized 수를 노출한다.

| origin | 요청 | realized | Δ | 합 |
|---|---|---|---|---|
| `divide_sm` | (74, 34) | **(74, 34)** | 0, 0 | 108 |
| `divide_sm` | (54, 54) | **(54, 54)** | 0, 0 | 108 |
| `c2_sweep` | (92, 16) | (92, 16) | 0, 0 | 108 |
| `c2_sweep` | (84, 24) | (84, 24) | 0, 0 | 108 |
| `c2_sweep` | (64, 44) | (64, 44) | 0, 0 | 108 |
| `c2_sweep` | (54, 54) | (54, 54) | 0, 0 | 108 |
| `c2_sweep` | (16, 92) | (16, 92) | 0, 0 | 108 |

`n_exact = 7/7`, `any_unassigned = false`(전 대상 `realized_sum = 108`).

**provenance**: §0–§6의 결정 규칙은 이 결과를 보기 **전에** 고정됐고
(이 파일이 프로브 실행보다 먼저 작성됨), addendum은 규칙을 수정하지 않는다.

## A3. 이 결과가 바꾸는 것 / 바꾸지 않는 것

**바꾼다 — `%smid` R0의 payoff가 더 줄었다.** `PREREG_SMID_R0_2026-08-14.md` §0.1의
payoff 항목 1은 `g2s_analyze.py:1488`의 `eps = log(mt/mc)/log(108.0/34.0)` **분모
정정**이었다. 34는 요청값이자 드라이버 보고 realized 값이므로 **정정할 것이 없다**
(드라이버 보고 층에서). 감사(`%smid` F7)는 이미 payoff 3건 중 1건이 자기 §8-3에
막히고 1건은 L1 소관이라 판정했는데, 이 결과는 **막힌 그 항목이 애초에 무의미했음**을
보인다. ⇒ `%smid` 재설계 시 §0.1은 **다시 쓰여야 한다.**

**바꾼다 — 다섯 번째 세계의 개수 층 증거.** `%smid` 감사 F2가 지목한
`disjoint ∧ ∪ ⊊ D`(전달됐으나 108을 덮지 않음)는 **개수 층에서는 관측되지 않는다**
(`realized_sum = 108` 전 대상). id 집합 층에서의 성립 여부는 여전히 미측정이다.

**바꾸지 않는다.** §4의 해석 제한 6개 전부 유효하다. 특히:
- 이것은 **드라이버 자기보고**다. "하드웨어가 그 SM들에서 실행했다"가 아니다.
  CONSENSUS §1-1 Gate 1 블록의 인용 금지("실현 파티션을 **측정**했다" / "prefill
  74 SM·decode 34 SM에서 **실행**됐다")는 **그대로 유지된다.**
- `%smid`의 고유 질문(두 컨텍스트의 SM **id 집합**이 서로소인가)은 한 눈금도
  전진하지 않았다. 드라이버가 74/34를 보고하는 것과 그 74개·34개가 실제로 어떤
  물리 SM이고 서로소인지는 다른 명제다.
- 성능·정책 주장 0건.

## A4. 남은 한계 — 노드·부하 통제

- **`glogin01` 단일 노드, 단일 프로세스, 엔진 미기동, 경합 없음.** 방법론 게이트
  #17("같은 노드로 통제했다는 표를 믿지 말고 원자료를 대조하라")의 정신에 따라
  컴퓨트 노드 확인을 별도로 제출한다(`smsplit_realized.sbatch`, 분 단위).
  두 결과가 다르면 **그 자체가 결과**이며, 같으면 노드 의존성이 없다는 좁은 진술만
  얻는다.
- 엔진이 살아 있는 상태(다른 CUDA 할당·컨텍스트 존재)에서 같은 호출이 같은 파티션을
  내는지는 **미측정**이다. split은 device 층 연산이라 무관할 것으로 기대되나,
  기대는 측정이 아니다 — 이 문장을 근거로 쓰지 않는다.

---

# Addendum 2 — 2026-08-14 (컴퓨트 노드 확인, job 882374, 원문 미수정)

## A5. 결과 (컴퓨트 노드 gpu43, `smsplit_realized_882374.json`)

`smsplit_realized.sbatch` 제출(`--partition=amd_a100nv_8 --gres=gpu:1`, 15분
문턱, `--comment="field=efficientai;appl=pytorch"`). exit 0(`probe_exit=0`).

**판정 = `REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`** — A2(glogin01)와
**레코드 완전 일치**(`n_exact=7/7`, 7지점 전부 `delta_a=delta_b=0`,
`realized_sum=108`, `any_unassigned=false`). ⇒ A4가 예고한 두 결과 중
**"같으면 노드 의존성이 없다는 좁은 진술만 얻는다"** 쪽이 확정됐다.

## A6. ★ 부수 발견 — glogin01과 컴퓨트 노드는 하드웨어 SKU가 다르다

레코드는 같지만 `gpu_name` 필드는 다르다:

| origin | host | `gpu_name` |
|---|---|---|
| A2 (addendum 1) | `glogin01` | `NVIDIA A100 80GB PCIe` |
| A5 (이 addendum) | `gpu43` (job 882374) | `NVIDIA A100-SXM4-80GB` |

`scontrol show node gpu36 gpu40 gpu43`은 셋 다 `AvailableFeatures=
A100-80GB_8,hwperf`·`Gres=gpu:8`로 동일 — `amd_a100nv_8` 파티션의
컴퓨트 노드는 SKU가 동질(SXM4)이며, **로그인 노드만 다른 SKU(PCIe)**다.

이 사실 자체는 이 프로브의 사전등록 질문(§1, 요청 vs 드라이버 보고
SM 개수)과 무관하지만(그 질문은 SKU 무관하게 A2=A5로 이미 닫혔다),
**독립적으로 재사용 가치가 있다** — `TRAFFIC_ROOFLINE_DIAGNOSTIC_
2026-08-11.md`의 `nvidia-smi -q` 인용(§6.2)이 정확히 이 혼동(로그인
노드에서 읽은 하드웨어를 컴퓨트 노드 캠페인에 귀속)을 저질렀음을
드러냈다(같은 문서 §11 addendum, 2026-08-14). 이 addendum이 그
발견의 **원 데이터**다.

## A7. 이 결과가 바꾸는 것 / 바꾸지 않는 것

**바꾼다**: A4의 "노드 의존성" 질문 — glogin01·컴퓨트 노드(SXM4) 양쪽에서
`REQUEST_EQUALS_DRIVER_REPORTED_PARTITION`이 재현됐다(노드·하드웨어 SKU에
무관). 이 결과는 `TRAFFIC_ROOFLINE_DIAGNOSTIC_2026-08-11.md` §11 하드웨어
정정의 근거로 인용 가능(§A6).

**바꾸지 않는다**: §2·§4의 해석 제한 전부 유효. 특히 이것은 여전히
**드라이버 자기보고**이고, `%smid`의 SM id 집합 disjointness·성능·정책
주장은 여전히 미측정/0건이다. §4-1의 인용 금지(Gate 1 블록 등)도 그대로
유지된다.

정본 반영: `reports/CONSENSUS.md` rev26·§1-1(E-3 블록)·§3 항목46(追記),
`PROJECT_STATUS.md` "확정된 결과" 1번(E-3 블록)·"8B decode-SM 민감도 측정
노트"·"방법론 게이트" #32(追記), `workspace/engine-port/RESUME.md`(환경
사실).
