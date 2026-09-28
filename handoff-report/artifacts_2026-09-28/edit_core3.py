import sys
def cut(s, start, end_marker):
    i = s.index(start); j = s.index(end_marker, i) + len(end_marker); return s[i:j]

# ---- REPORT: replace section 6 body with three core contributions ----
p='pdmux_roadmap.html'; s=open(p,encoding='utf-8').read()
sec6_start = s.index('<h2>6. 논문으로 정리될 때의 기여</h2>')
sec6_end = s.index('<footer>')
old_sec6 = s[sec6_start:sec6_end]
# keep the detailed table to move into review page
detail_table = cut(old_sec6, '<div class="tbl"><table>', '</table></div>')

new_sec6 = '''<h2>6. 논문으로 정리될 때의 기여</h2>
<p class="prose" style="margin-top:10px">논문은 아래 세 가지를 핵심 기여로 세운다. 각각 "무엇을 보이는가"와 "그것이 성립할 때 무엇이 달라지는가"를 적었다. 아직 측정이 없는 부분은 이번 실험이 채우는 몫이며, 그 세부 항목·근거 상태·한계는 짝 페이지 <a href="https://claude.ai/artifact/NhCAwkHtZb5LeYJqQ4e8fH">PD-mux 실험 검토</a> 3절에 있다.</p>
<div class="tiers">
  <div class="tier"><h3><span class="chip c-req">기여 1</span>이종 token mixer의 서빙 비용 구조를 시스템 관점에서 규명</h3>
    <p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> attention 층과 SSM 층이 한 모델 안에 섞여 있을 때, prefill과 decode 각각의 비용이 입력 길이·문맥 길이·SM 수·부하에 따라 어떻게 다르게 움직이는지를 실제 서빙 엔진에서 잰다. prefill에서는 두 층의 SM 민감도가 같아 층 종류로 SM을 나눌 이유가 없다는 것(측정됨), decode에서는 소수의 attention 층이 문맥과 함께 비용을 지배해 "decode가 필요로 하는 최소 SM"이 문맥·부하의 함수로 움직인다는 것(이번 실험 3-a), 그리고 KV가 작아 배치 여유가 커서 처리 한계와 배치 크기가 일반 transformer와 다르다는 것(3-b)을 한 틀로 묶는다.</p>
    <p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> hybrid 모델을 위한 PD-mux 배분이 "어느 축에 민감하고 어느 축에는 둔감한가"를 처음으로 하나의 비용 지도로 제공한다. 이후의 배분 설계(기여 3)와 다른 hybrid 모델·다른 GPU로의 이전이 이 지도 위에서 이루어진다.</p></div>
  <div class="tier"><h3><span class="chip c-done">기여 2</span>재배분이 작동하는 경계와 작동하지 않는 기전의 규명</h3>
    <p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> GPU 배분을 "어디서" 바꿀 수 있는지를 실측으로 가른다. 층 종류 경계마다 바꾸는 방식은 GPU 비우기와 CUDA graph 비양립 때문에 운영 설정에 들어갈 수 없다(확정). 부하를 보며 반응적으로 바꾸는 방식은 전환 비용이 아니라 decode 굶김이 대기열 폭발로 이어지는 연쇄 때문에 고정 배분을 넘지 못한다(확정, 그림 3). 그리고 나눠 놓은 decode가 느려지는 이유를 "SM 부족"과 "prefill과의 동거"로 분해해 동거 시간을 정의된 측정량으로 세운다(4-a·4-b).</p>
    <p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> "세밀하게, 자주 바꿀수록 좋다"는 직관이 hybrid PD-mux에서 어디까지 참인지의 경계를 기전 수준으로 제공한다. 이 경계가 곧 배분 설계의 제약 조건이 되어, 다음 설계가 같은 실패를 반복하지 않게 한다.</p></div>
  <div class="tier"><h3><span class="chip c-cond">기여 3</span>모델 구조를 아는 배분 선택과 운영 가이드</h3>
    <p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> 기여 1의 비용 지도와 기여 2의 제약 위에서, 모델의 attention/SSM 구성과 워크로드로부터 "decode가 앉을 자리"를 미리 예측해 배분을 고르는 방법을 세우고, 고정 배분·구조 정보 없는 일반 동적 배분과 goodput으로 비교한다(5-a → 5-b). 이와 함께 지금 바로 쓸 수 있는 운영 규칙 — 최고 decode 부하 기준으로 decode 쪽에 넉넉한 고정 배분, 검증된 프로파일이 없으면 균등 — 을 4모델에서 확인한다(4-c).</p>
    <p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> 새 hybrid 모델이 나왔을 때 서빙 실험을 다시 돌리지 않고 구성 정보만으로 초기 배분을 고를 수 있게 된다. 비교에서 이기면 "기준점을 알면 배분 선택이 산다"가, 지면 "기준점을 알아도 고정 배분이 낫다"가 결과로 남아 어느 쪽이든 운영 가이드의 근거가 된다.</p></div>
</div>
<p class="prose" style="margin-top:12px">한 줄로 요약하면, <b>hybrid LLM의 PD-mux 배분을 "왜 그렇게 나눠야 하는가"부터 "어디서 바꿀 수 있는가", "무엇을 보고 고를 것인가"까지 실측으로 잇는 논문</b>이다. 기여 1·2는 지금 자산과 2–4단계로 완성되고, 기여 3의 비교 부분은 5단계가 열릴 때 더해진다.</p>

'''
s = s[:sec6_start] + new_sec6 + s[sec6_end:]
open(p,'w',encoding='utf-8').write(s); print('report ok')

# ---- REVIEW: move detailed table under section 3 ----
p='pdmux_review.html'; s=open(p,encoding='utf-8').read()
anchor = '<p class="prose" style="margin-top:12px">투고 방향:'
assert s.count(anchor)==1
detail_table = detail_table.replace('짝 페이지', '보고용 페이지')
block = ('<h3 style="margin-top:22px">세부 항목과 근거 상태 (보고용 6절의 세 핵심 기여를 이루는 항목)</h3>\n'
         '<p class="prose" style="margin-top:8px">보고용 페이지의 기여 1은 아래 ①④⑧, 기여 2는 ②③⑤, 기여 3은 ⑥⑦⑨에 대응한다. "어느 실험이 만드는가"는 이 페이지 2절의 단계 번호다.</p>\n'
         + detail_table + '\n')
s = s.replace(anchor, block + anchor)
open(p,'w',encoding='utf-8').write(s); print('review ok')
