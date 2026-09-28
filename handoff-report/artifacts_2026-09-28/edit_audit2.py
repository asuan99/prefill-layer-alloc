import re, sys
def sub1(s, old, new, label):
    n = s.count(old)
    if n != 1:
        print(f"FAIL {label}: count={n}"); sys.exit(1)
    return s.replace(old, new)

# ---------------- REPORT ----------------
p = 'pdmux_roadmap.html'
s = open(p, encoding='utf-8').read()

# H1 figure: drop numeric ratios (unaudited micro; ratio depends on SM choice)
s = sub1(s, '>attention: 2,940× (초선형)</text>', '>attention: 초선형 (L²에 가까움)</text>', 'H1 attn label')
s = sub1(s, '>Mamba: 27× (선형 이하)</text>', '>Mamba: 선형 이하</text>', 'H1 mamba label')
s = sub1(s, '>KISTI A100 · Zamba2-2.7B · triton · graph OFF 마이크로 · 배치 1</text>',
            '>KISTI A100 · Zamba2-2.7B · triton · graph OFF 마이크로 · 배치 1 · 미감사</text>', 'H1 cond')
s = sub1(s, '>attention=코어만 · Mamba=mixer 전체: 절대값 비교 금지, 형태만</text>',
            '>attention=코어만 · Mamba=mixer 전체 · 배율 수치는 SM 수에 따라 달라 인용 안 함, 형태만</text>', 'H1 note')
s = sub1(s, '실측된 prefill 층당 시간은 입력 256에서 32768 토큰으로 갈 때 attention이 약 2,940배, Mamba가 약 27배 늘어 attention은 초선형, Mamba는 선형 이하다.',
            '미감사 마이크로 벤치마크에서 prefill 층당 시간은 입력 256에서 32768 토큰으로 갈 때 attention이 Mamba보다 자릿수 단위로 더 크게 늘어 attention은 초선형, Mamba는 선형 이하다. 배율 수치는 SM 수에 따라 달라 인용하지 않는다.', 'H1 aria')
s = sub1(s, '그림 H1. 왼쪽(실측): 입력이 128배 길어질 때 attention 층의 prefill 시간은 약 2,940배, Mamba 층은 약 27배 늘었다. attention은 L²에 가깝고 Mamba는 짧은 입력에서 고정 비용이 커 선형 이하다(두 선의 절대값은 계측 버킷이 달라 비교하지 않는다).',
            '그림 H1. 왼쪽(실측, 미감사 마이크로): 입력이 128배 길어질 때 attention 층의 prefill 시간은 Mamba 층보다 자릿수 단위로 더 크게 늘었다. attention은 L²에 가깝고 Mamba는 짧은 입력에서 고정 비용이 커 선형 이하다. 배율의 절대값은 SM 수에 따라 달라지고 두 선의 계측 버킷이 달라 수치로 인용하지 않으며 형태만 본다.', 'H1 caption')

# Section 3 table
s = sub1(s, '(입력 1024 토큰 이상에서 비 ≈1.0)</td><td>측정됨 (H3-a, 마이크로)</td>',
            '(입력 1024 토큰 이상에서 비 ≈1.0, 256·512에서는 1.2–1.34로 작은 차이)</td><td>측정됨 (H3-a, 마이크로)</td>', 'sec3 diffB')
s = sub1(s, '<td>prefill을 조각으로 끊어 decode와 동시에 돌리는 것이 가능</td><td>엔진 코드 확인</td></tr>',
            '<td>prefill을 조각으로 끊어 decode와 동시에 돌리는 것이 가능. 조각 크기는 서버 전역 토큰 예산(65536)으로 정해지고 decode step 시간에는 맞추지 않는다</td><td>엔진 코드 확인</td></tr>\n'
            '<tr><td>선행 연구의 조절 방식</td><td colspan="2">MuxWise: decode 한 iteration을 덮도록 prefill 층 수를 정하고 추정기로 배분 크기를 고름. Bullet: 시간에 따라 prefill SM 수를 바꾸며 층 단위로 실행(SM 겹침·요청 재정렬·decode 중단 포함)</td><td>둘 다 <b>층 단위 prefill 진행량과 SM 배분을 함께</b> 조절(논문 수준). MuxWise 공개 엔진(우리 기판)에는 그 부분이 없고 decode 배치 크기 문턱 표만 있다. 우리는 조각 경계의 SM 재결정과 고정 배분만 시험했고 진행량 제어는 미시험</td><td>선행 연구 감사 (2026-09-22)</td></tr>', 'sec3 span row')

# H3-g chart: residency ranges are from an unregistered script; use registered D44-occupancy values
s = sub1(s, '>동거 3–10%</text>', '>D44 유지 13.64%</text>', 'H3g A')
s = sub1(s, '>동거 43–98%</text>', '>D44 유지 97.63%</text>', 'H3g B')
s = sub1(s, '>처리 한계와 동거 비율 — 두 워크로드</text>', '>처리 한계와 설정 배분의 실제 유지 비율</text>', 'H3g title')
s = sub1(s, '>동거 비율은 가중 규약에 따라 달라짐 · VESSL에서 재측정(3-b)</text>',
            '>유지 비율 = 등록 규약(다음 스냅샷까지·무캡) · 동거 비율은 집계 등록 후 인용</text>', 'H3g note')
s = sub1(s, '처리 한계: 채팅형 shape A 약 3.05 req/s, 문서 요약형 shape B 약 0.70 req/s. 동거 비율은 A에서 3–10%, B에서 43–98%.',
            '처리 한계: 채팅형 shape A 약 3.05 req/s, 문서 요약형 shape B 약 0.70 req/s. decode 44 SM으로 설정한 배분이 실제로 유지된 시간 비율은 A에서 13.64%, B에서 97.63%(등록 규약).', 'H3g aria')
s = sub1(s, 'H3-g. 같은 모델이라도 워크로드에 따라 처리 한계가 4배 넘게 다르고 동거 비율이 정반대다.',
            'H3-g. 같은 모델이라도 워크로드에 따라 처리 한계가 4배 넘게 다르고, "decode 44 SM"으로 설정한 배분이 실제로 유지된 시간 비율이 정반대다. 채팅형은 대부분의 decode 시간을 prefill 없이 GPU 전체로 돌았다. 이것은 설정 유지 비율이지 동거 비율이 아니다.', 'H3g caption')

# Section 4 workload table + map labels
s = sub1(s, '동거 3–10%, decode 배치 5–18, prefill 조각 1개 → 배분을 바꿀 지점이 도착·종료뿐',
            '설정 배분(D44)의 실제 유지 13.64%(나머지는 prefill 없이 GPU 전체), decode 배치 5–18, prefill 조각 1개(관측 배치 ≤1,024 토큰) → 배분을 바꿀 지점이 도착·종료뿐', 'wl A')
s = sub1(s, '동거 43–98%(가중 방식에 따라), 조각 7–14개', '설정 배분 유지 97.63%, 조각 7–14개', 'wl B')
s = sub1(s, '>동거 3–10% · decode 배치 5–18 · 조각 1개</text>', '>D44 유지 13.64% · decode 배치 5–18 · 조각 1개</text>', 'map A')
s = sub1(s, '>동거 43–98%(가중 규약 의존) · decode 배치 1–2</text>', '>D44 유지 97.63% · decode 배치 1–2</text>', 'map B')

# Section 5 mapping table 5-b
s = sub1(s, '<td>모델 구조 정보로 고른 배분 vs 일반 방법 vs 고정 배분의 goodput</td>',
            '<td>모델 구조 정보로 고른 배분 vs 일반 동적 vs 고정 배분의 goodput (제안: 부하 인덱스 배분표 arm 추가)</td>', 'sec5 5b')

# Section 6 table: replace rows ②③⑤⑦⑧ and closing
old2 = s[s.index('<tr><td><b>② 층 종류별로 GPU를 나누는 방식은 실패한다는 기전</b>'):]
old2 = old2[:old2.index('</tr>')+5]
new2 = ('<tr><td><b>② 층 종류 경계마다 GPU를 다시 나누는 방식은 실패한다는 기전</b></td><td>attention/SSM 층 경계마다 SM을 바꾸면 GPU 비우기(격차의 약 절반)와 겹침 손실·CUDA graph 비양립으로 decode가 2–3배 느려지고, 운영 설정(graph ON)에는 들어갈 수 없다</td>'
        '<td>확정 — 현 구현 범위 한정 (H3-a·H3-c, 양쪽 graph OFF 측정)</td><td>—(옛 환경 결과 사용)</td>'
        '<td>우리 엔진·green context·Zamba2 경로 한정. <b>선행(MuxWise·Bullet)이 논문에서 쓰는 방식 — 층 단위 prefill 진행량과 SM 배분을 함께 조절 — 은 시험하지 않았으므로 이 결과가 그 방식을 반증하지 않는다.</b> 그 방식은 엔진 수정(조각 크기를 decode step 시간에 맞추기)이 필요해 현 계획 밖. libsmctrl 등 다른 분할 기술과 비교 없음</td></tr>')
s = s.replace(old2, new2)
old3 = s[s.index('<tr><td><b>③ 실시간 조절이 고정 배분을 못 이기는 이유</b>'):]
old3 = old3[:old3.index('</tr>')+5]
new3 = ('<tr><td><b>③ 반응형 실시간 조절이 고정 배분을 못 이기는 이유</b></td><td>전환 비용이 아니라 decode 굶김이 대기열 폭발로 이어지는 연쇄 때문. 조절기가 "앉을 자리"(decode에 필요한 최소 SM)를 모른 채 움직인 결과</td>'
        '<td>확정 (H3-d·H3-e, 그림 3; 4회 이상 반복, 느슨·빡빡 SLO 양쪽)</td><td>—(옛 환경 결과 사용). 교정 사다리: 3-a(자리 곡선) → 3-b(한계) → 5-a(넘어야 할 선의 높이) → 구조 분리 검증 → 5-b</td>'
        '<td>시험한 조절기 계열(하나의 루프가 두 단계를 관리·SM 분할·반응형)과 모델 1종에 한정. 도달 가능한 상한이 아니다. 기준점을 알려 준 조절기가 이기는지는 ⑨의 열린 질문. 선행(Bullet)의 반대 주장은 열린 항목</td></tr>')
s = s.replace(old3, new3)
old5 = s[s.index('<tr><td><b>⑤ 동거 시간의 정의된 측정과'):]
old5 = old5[:old5.index('</tr>')+5]
new5 = ('<tr><td><b>⑤ 동거 시간의 정의된 측정과 "SM 수 vs 동거" 분해</b></td><td>나눠 놓은 decode가 느린 이유를 SM 부족과 prefill 동거로 가르고, 동거 비율을 정의·분모·규약을 갖춘 집계값으로 센다</td>'
        '<td>옛 자료 재집계 있음(H3-g는 설정 배분의 실제 유지 비율; 동거 비율 집계 스크립트는 미등록이라 수치 인용 전 등록 필요). 분해 실험은 미실행</td><td>4-a, 4-b</td>'
        '<td>일반 transformer에서는 선행(MuxWise·Bullet)이 동거 간섭을 이미 보고. 조사한 4편 범위에서 정의된 집계값으로 보고한 것은 확인되지 않으나(Bullet 그림 20a 각주 확인 필요) "첫 측정"이라 쓰지 않음. 우리 몫은 hybrid 재확인 + 정의된 집계 + 워크로드별 값</td></tr>')
s = s.replace(old5, new5)
old7 = s[s.index('<tr><td><b>⑦ 방법론</b>'):]
old7 = old7[:old7.index('</tr>')+5]
new7 = ('<tr><td><b>⑦ 방법론</b></td><td>배분은 설정값이 아니라 실측값으로 검증해야 하고(설정 D44의 실제 유지가 채팅형에서 13.64%), 환경을 옮길 때 확인 절차가 필요하며, 부하는 처리 한계의 비율로 정의한다</td>'
        '<td>옛 환경에서 두 번의 결론 번복으로 확인</td><td>2단계, 3-b, 4-b</td><td>—</td></tr>')
s = s.replace(old7, new7)
old8 = s[s.index('<tr><td><b>⑧ (조건부) 모델 구조 정보로 배분을 고르면'):]
old8 = old8[:old8.index('</tr>')+5]
new8 = ('<tr><td><b>⑧ 재배분 경계의 결정 표면 — 어디서 바꿀 수 있는가</b></td><td>층 종류 경계는 불가(②), 조각·step 경계는 가능. 그러나 기본 예산에서 짧은 입력은 prefill이 조각 1개라 prefill 도중의 결정 지점이 없다 — 선행이 전제하는 "여러 층 단위 진행"이 채팅형에서는 서지 않는다</td>'
        '<td>옛 자료 재집계(판정 아님, 스크립트 미등록). 조각 수 규칙은 엔진 코드로 확정</td><td>3-b·4-b의 기록에서 추가 비용 없이 재확인. 예산을 줄여 조각을 늘리는 비용(TTFT 2–4배 선례)은 5-a 전 확인 항목</td>'
        '<td>Nano-9B-v2·예산 65536·입력 256 기준. 예산은 서버 전역이라 요청별로 조각을 맞출 수 없음(엔진 수정 필요)</td></tr>\n'
        '<tr><td><b>⑨ (조건부) 기준점을 바로잡은 배분 선택이 고정 배분을 넘는가</b></td><td>decode 최소 SM 곡선(3-a)과 워크로드별 최적(5-a)을 아는 배분 선택 — 특히 attention/SSM 구성 정보를 쓰는 것 — 이 고정 배분·일반 동적·부하 인덱스 배분표보다 goodput 3% 이상 높은가</td>'
        '<td>미검증. 이기든 지든 결과가 남는다: 이기면 "기준점을 알면 동적이 산다", 지면 ③보다 강한 부정 결과</td><td>5-a → 5-b. 비교군: 고정(B1)·일반 동적(B5)·구조 정보(B6)·[제안] 부하 인덱스 배분표(선행 공개 엔진의 선택기 재현, 별도 사전 등록·감사 필요)</td>'
        '<td>성공해도 "실시간 조절"이 아니라 "워크로드에 맞는 배분을 미리 고르는 능력"으로 서술. 선행(MuxWise)이 워크로드별 표를 이미 씀 → 차별점은 구조 정보와 hybrid. 모델을 건너 일반화하려면 구조가 다른 2–3종으로 맞추고 미사용 모델로 예측(hold-out). 부하 인덱스 표 arm이 져도 "MuxWise가 졌다"고 쓰지 않음(추정기·디스패처 없는 재현)</td></tr>')
s = s.replace(old8, new8)
s = sub1(s, '정리하면, 지금 자산과 2–4단계로 ①②③⑤⑥⑦이 서고 ④가 채워진다. 이는 "특성화 + 안 되는 이유의 규명 + 운영 가이드" 논문이다. ⑧은 5단계가 열릴 때만 더해진다.',
            '정리하면, 지금 자산과 2–4단계로 ①②③⑤⑥⑦⑧이 서고 ④가 채워진다. 이는 "이종 token mixer의 시스템 특성화 + 안 되는 이유의 규명 + 운영 가이드" 논문이다. ②③은 시험한 계열의 실패 기전이지 최종 상한이 아니며, 그 교정 경로가 ⑨다. ⑨는 5단계가 열릴 때만 더해진다.', 'closing')
open(p, 'w', encoding='utf-8').write(s)
print('report ok', len(s))

# ---------------- REVIEW ----------------
p = 'pdmux_review.html'
s = open(p, encoding='utf-8').read()
s = sub1(s, '이 수치는 CUDA graph를 끈 상태에서 잰 것이라 크기가 아니라 방향으로 인용한다.</div>',
            '이 수치는 CUDA graph를 끈 상태에서 잰 것이라 크기가 아니라 방향으로 인용한다. 선행(MuxWise·Bullet)이 논문에서 쓰는 "층 단위 prefill 진행량 + SM 배분 동시 조절"은 시험하지 않았으므로 이 결과가 그 방식을 반증하지 않는다.</div>', 'rv sec1 2')
s = sub1(s, '이 결론은 "하나의 실행 루프가 두 단계를 함께 관리하는 반응형 제어"에 한정된다.</div>',
            '이 결론은 "하나의 실행 루프가 두 단계를 함께 관리하는 반응형 제어"에 한정된다. 기준점(decode 최소 SM·워크로드별 최적)을 모른 채 움직인 조절기의 결과이며, 도달 가능한 상한이 아니다.</div>', 'rv sec1 3')
s = sub1(s, '<b>동거 시간이 실제로 얼마나 되는지</b>를 처음으로 정의된 방식으로 세는 것이다.',
            '<b>동거 시간이 실제로 얼마나 되는지</b>를 정의된 방식으로 세는 것이다(조사한 선행 4편에서는 집계값 보고가 확인되지 않음. "첫 측정"이라 쓰지 않는다).', 'rv 4a')
old5b = s[s.index('<div><h4>5-b. 모델 구조 정보를 쓰는 배분 선택'):]
old5b = old5b[:old5b.index('</ul></div>')+11]
new5b = ('<div><h4>5-b. 기준점을 바로잡은 배분 선택 (10–20 GPU-h, arm 추가 시 증가)</h4><ul>'
         '<li>질문: decode 최소 SM 곡선(3-a)과 워크로드별 최적(5-a)을 아는 배분 선택 — 특히 모델의 attention/SSM 구성 정보를 쓰는 것 — 이 고정 배분이나 일반 동적보다 goodput이 3% 이상 좋은가.</li>'
         '<li>조건: 5-a에서 차이가 확인되고, 별도 실행 구조(prefill·decode 각각 독립 스레드) 검증을 통과한 뒤에만.</li>'
         '<li>비교군: 고정(B1) · 일반 동적(B5) · 구조 정보(B6). <b>[제안] 부하 인덱스 배분표 arm</b> — MuxWise 공개 엔진의 선택기(decode 배치 크기 문턱 → 배분 전환)를 108 SM용 표로 재현. 설정만으로 가능하지만 R2 정책·SLO 스케줄·분할 유지 옵션을 모두 꺼야 해 고정 arm과 노브가 여럿 동시에 움직인다 → 별도 사전 등록 + 규칙층 감사 후에만. 선행의 논문 방식(prefill 층 수를 decode step 시간에 맞춤 + 추정기)은 엔진 수정이라 현 계획 밖.</li>'
         '<li>비용: arm 하나당 반복 4회 × 독립 작업 2회가 늘어난다(10–20 GPU-h 산정에 미포함). 표의 행이 늘면 decode CUDA graph 메모리도 늘어 처리 한계를 arm별로 다시 재야 한다.</li>'
         '<li>솔직한 기대: 선행 연구(MuxWise)가 이미 워크로드별 배분표를 쓰고 있어서, 남는 차별점은 "구조 정보를 쓰느냐"와 hybrid 한정이다. 옛 환경 결과는 실시간 조절이 이길 영역이 좁다고 시사한다. 이기든 지든 결과가 남는다. 부하 인덱스 표 arm이 져도 "MuxWise가 졌다"고 쓰지 않는다(추정기·디스패처 없는 재현).</li></ul></div>')
s = s.replace(old5b, new5b)
s = sub1(s, '한계: 우리 엔진·green context 방식·Zamba2 경로에 한정, 다른 분할 기술(libsmctrl)과 비교 없음.</li>',
            '한계: 우리 엔진·green context 방식·Zamba2 경로에 한정, 다른 분할 기술(libsmctrl)과 비교 없음. 선행(MuxWise·Bullet)의 논문 방식 — 층 단위 prefill 진행량과 SM 배분을 함께 조절 — 은 미시험이므로 반증 대상이 아니다.</li>', 'rv tier1 a')
s = sub1(s, '선행 연구(Bullet)가 반대 주장을 하는 부분은 검토 중.</li>',
            '선행 연구(Bullet)가 반대 주장을 하는 부분은 검토 중. 시험한 조절기 계열의 실패이지 도달 가능한 상한이 아니며, 교정 사다리(3-a → 3-b → 5-a → 구조 분리 → 5-b)가 그 다음 질문이다.</li>', 'rv tier1 b')
s = sub1(s, '(4-a). 우리가 조사한 범위(DuetServe·MuxWise·Nexus·Bullet)에서 정의·분모를 갖춘 집계값으로 이를 보고한 선행은 없었다.',
            '(4-a). 조사한 4편(DuetServe·MuxWise·Nexus·Bullet) 범위에서 정의·분모·규약을 갖춘 집계 추정량으로 이를 보고한 것은 확인되지 않는다(Bullet 그림 20a 각주는 확인 필요).', 'rv tier2 res')
s = sub1(s, '<li><b>운영 가이드</b>: 최고 decode 부하 기준의 고정 배분.</li>',
            '<li><b>재배분 경계의 결정 표면</b>(3-b·4-b 기록에서 추가 비용 없이). 기본 예산에서 짧은 입력은 prefill이 조각 1개라 prefill 도중 결정 지점이 없다. 옛 자료 재집계(판정 아님, 스크립트 미등록)를 VESSL 기록으로 재확인한다.</li>\n'
            '    <li><b>운영 가이드</b>: 최고 decode 부하 기준의 고정 배분.</li>', 'rv tier2 surface')
s = sub1(s, '<li><b>모델 구조 정보를 이용한 배분 선택이 일반 방법보다 낫다</b>(5-b). 성공해도 "실시간 조절"이 아니라 "워크로드에 맞는 고정 배분을 미리 잘 고르는 능력"으로 서술한다.</li>',
            '<li><b>기준점을 바로잡은 배분 선택이 고정 배분을 넘는가</b>(5-b) — 구조 정보(B6) vs 일반 동적(B5) vs 고정(B1) vs [제안] 부하 인덱스 표. 성공해도 "실시간 조절"이 아니라 "워크로드에 맞는 고정 배분을 미리 잘 고르는 능력"으로 서술한다. 모델을 건너 일반화하려면 hold-out(구조가 다른 2–3종으로 맞추고 미사용 모델 예측)이 필요하다.</li>', 'rv tier3')
s = sub1(s, '<tr><td>하드웨어 수준 SM 추적</td>',
            '<tr><td>선행 논문 방식의 재현 (MuxWise 층 수 정합 · Bullet SM 겹침/재정렬/decode 중단)</td><td>선행이 논문에서 쓰는 "층 단위 진행량 + SM 배분 동시 조절"이 우리 기판에서도 이기는가</td><td>조각 크기를 decode step 시간에 맞추려면 엔진 수정(요청별 조각 수 + 추정기)이 필요하고, Bullet 방식은 다른 분할 기술(libsmctrl·MPS)이라 별도 빌드. 정확성 게이트부터 다시 시작해야 해 이번 계획 밖. 부하 인덱스 표(공개 엔진 선택기)만 5-b의 제안 arm으로 남긴다.</td><td class="num">미산정</td></tr>\n'
            '<tr><td>하드웨어 수준 SM 추적</td>', 'rv sec4 row')
s = sub1(s, '<td>5-a 결과를 본 뒤 결정</td>', '<td>5-a 결과를 본 뒤 결정. 5-b에 arm을 더하면 그만큼 증가(미산정)</td>', 'rv budget')
s = sub1(s, '<li><input type="checkbox" id="d7">',
            '<li><input type="checkbox" id="d8"><label for="d8">5-b에 부하 인덱스 배분표 arm(선행 공개 엔진 선택기 재현)을 넣을지 — 별도 사전 등록·감사 전제, 반복 비용 증가</label><span class="who">사용자 + 감사</span></li>\n'
            '  <li><input type="checkbox" id="d7">', 'rv decision')
open(p, 'w', encoding='utf-8').write(s)
print('review ok', len(s))
