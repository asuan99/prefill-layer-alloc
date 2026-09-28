import sys
def rep(s, old, new, label):
    n = s.count(old)
    if n != 1: print(f"FAIL {label}: count={n}"); sys.exit(1)
    return s.replace(old, new)
p='pdmux_review.html'; s=open(p,encoding='utf-8').read()
s = rep(s, '검증된 모델 프로파일이 없으면 균등 배분.</p>', '검증된 모델 프로파일이 없으면 엔진 기본 배분 선택기(decode 배치 크기 문턱 표, agnostic)를 쓴다.</p>', 'l107')
# mapping sentence
s = rep(s, '보고용 페이지의 기여 1은 아래 ①④⑧, 기여 2는 ②③⑤, 기여 3은 ⑥⑦⑨에 대응한다.',
           '보고용 페이지의 기여 1은 아래 ①④와 3-b(처리 한계, 사실 기록), 기여 2는 ②③⑤⑧, 기여 3은 ⑥과 5-a 여유·⑨(조건부)에 대응한다. ⑦은 방법론으로 따로 둔다.', 'mapping')
# ⑤ row experiment column
s = rep(s, '분해 실험은 미실행</td><td>4-a, 4-b</td>', '분해 실험은 미실행</td><td>4-a (4-b는 계측 타당성 확인용, 기여 아님)</td>', 'row5')
# ⑦ row 13.64% convention
s = rep(s, '(설정 D44의 실제 유지가 채팅형에서 13.64%)', '(설정 D44의 실제 유지가 채팅형에서 등록 규약 a_r4 기준 13.64%)', 'row7')
# ⑨ row criterion
s = rep(s, '이 고정 배분·일반 동적·부하 인덱스 배분표보다 goodput 3% 이상 높은가</td>',
           '이 고정 배분(B1)·일반 동적(B5)보다 goodput이 짝지은 신뢰구간 기준 3% 이상 높고, 다른 영역에서 3% 넘게 나빠지지 않는가(부하 인덱스 arm은 등록되면 비교군으로 추가)</td>', 'row9')
# proposed arm naming (table ⑨ + 5-b block + decision)
s = rep(s, '[제안] 부하 인덱스 배분표(선행 공개 엔진의 선택기 재현, 별도 사전 등록·감사 필요)',
           '[제안] agnostic 기준선(MuxWise 저자들이 SGLang에 올린 기본 선택기, 저자 주석상 임시 데모)을 108 SM 표로 재현한 arm(별도 사전 등록·감사 필요)', 'row9 arm')
s = rep(s, '<b>[제안] 부하 인덱스 배분표 arm</b> — MuxWise 공개 엔진의 선택기(decode 배치 크기 문턱 → 배분 전환)를 108 SM용 표로 재현.',
           '<b>[제안] agnostic 기준선 arm</b> — MuxWise 저자들이 SGLang에 올린 기본 선택기(decode 배치 크기 문턱 → 배분 전환, 저자 주석상 임시 데모)를 108 SM용 표로 재현. 이미 잰 agnostic 기준선과 같은 기제이므로 새 방식이 아니다.', '5b arm')
s = rep(s, '5-b에 부하 인덱스 배분표 arm(선행 공개 엔진 선택기 재현)을 넣을지', '5-b에 agnostic 기준선 arm(MuxWise 저자 upstream 기본 선택기 재현)을 넣을지', 'decision')
# 5-b question: add second condition
s = rep(s, '이 고정 배분이나 일반 동적보다 goodput이 3% 이상 좋은가.</li>',
           '이 고정 배분(B1)이나 일반 동적(B5)보다 goodput이 짝지은 신뢰구간 기준 3% 이상 좋고, 다른 영역에서 3% 넘게 나빠지지 않는가.</li>', '5b q')
# Bullet 20a wording (two places)
s = s.replace('(Bullet 그림 20a 각주는 확인 필요)', '(Bullet 그림 20a는 구성별 지속시간 타임라인으로 보여 주지만 집계값은 아니다)')
s = s.replace('(Bullet 그림 20a 각주 확인 필요)', '(Bullet 그림 20a는 구성별 지속시간 타임라인, 집계값 아님)')
assert '20a 각주' not in s
# tier list: ⑥ status -> move 운영 가이드 to 이미 확보 tier
s = rep(s, '    <li><b>운영 가이드</b>: 최고 decode 부하 기준의 고정 배분.</li>\n', '', 'tier2 guide remove')
s = rep(s, '교정 사다리(3-a → 3-b → 5-a → 구조 분리 → 5-b)가 그 다음 질문이다.</li>\n  </ul></div>',
           '교정 사다리(3-a → 3-b → 5-a → 구조 분리 → 5-b)가 그 다음 질문이다.</li>\n    <li><b>운영 가이드</b>: 최고 decode 부하 기준의 고정 배분, 검증된 프로파일이 없으면 엔진 기본 선택기(agnostic). 이 규칙이 남기는 여유의 크기는 5-a가 잰다.</li>\n  </ul></div>', 'tier1 guide add')
# mapping table C1 label
s = rep(s, '<td>C1 (Claim A 계열) / Claim B / C3=HE0</td>', '<td>C1 (관측적 PD-mux&gt;fused) / Claim B / C3=HE0</td>', 'c1 label')
open(p,'w',encoding='utf-8').write(s); print('review ok')

p='pdmux_roadmap.html'; s=open(p,encoding='utf-8').read()
s = s.replace('(Bullet 그림 20a 각주 확인 필요)', '(Bullet 그림 20a는 타임라인, 집계값 아님)')
assert '20a 각주' not in s
open(p,'w',encoding='utf-8').write(s); print('report ok')
