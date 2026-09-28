import sys
def rep(s, old, new, label):
    n = s.count(old)
    if n != 1: print(f"FAIL {label}: count={n}"); sys.exit(1)
    return s.replace(old, new)
def rep_between(s, start, end, new, label):
    i = s.index(start); j = s.index(end, i)
    return s[:i] + new + s[j:]

# ================= REPORT =================
p='pdmux_roadmap.html'; s=open(p,encoding='utf-8').read()
# 기여 1
s = rep_between(s, '<p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> attention 층과 SSM 층이 한 모델 안에',
    '</p>',
    '<p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> attention 층과 SSM 층이 한 모델 안에 섞여 있을 때, prefill과 decode의 비용이 입력 길이·문맥 길이·SM 수·부하에 따라 어떻게 움직이는지를 실제 서빙 엔진에서 잰다. prefill 쪽은 이미 답이 있다. 층 종류별로 SM을 나눠도 모델 전체로 보면 얻을 것이 거의 없고, 실제 서빙에서 층별로 나눈 방식은 오히려 느려졌다(측정됨). decode 쪽은 이번 실험이 채운다. decode에 필요한 최소 SM이 문맥 길이와 부하에 따라 움직이는지, 움직인다면 얼마나 움직이는지를 운영 설정에서 처음부터 다시 잰다(3-a). 이 설정이 받아낼 수 있는 처리 한계도 함께 잰다(3-b). 그 움직임이 attention/SSM 구성 때문이라고 말하려면 구성이 다른 모델을 더 재야 하며, 이것이 다음 확장이다.', 'c1 what')
s = rep_between(s, '<p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> hybrid 모델을 위한 PD-mux 배분이',
    '</p>',
    '<p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> hybrid 모델의 PD-mux 배분이 어느 축에 민감하고 어느 축에 둔감한지를 운영 설정의 실측으로 정리한다. hybrid 모델의 PD-mux 문헌에서 비어 있는 부분이다. 이후의 배분 설계(기여 3)가 이 측정을 기준점으로 삼는다.', 'c1 effect')
# 기여 2
s = rep(s, '재배분이 작동하는 경계와 작동하지 않는 기전의 규명</h3>', '재배분이 가능한 경계와, 이득이 나지 않는 기전의 규명</h3>', 'c2 title')
s = rep_between(s, '<p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> GPU 배분을 "어디서" 바꿀 수 있는지를 실측으로 가른다.',
    '</p>',
    '<p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> GPU 배분을 "어디서" 바꿀 수 있는지를 실측으로 가른다. 우리 엔진에서 층 종류 경계마다 바꾸는 방식은 경계마다 GPU를 비워야 해 decode가 2–3배 느려지고, CUDA graph와 함께 쓸 수 없어 운영 설정에 들어가지 못한다(확정). 부하를 보며 배분을 바꾸는 반응형 조절기는, 우리가 시험한 구조에서 고정 배분을 넘지 못했다(확정, 그림 3). 원인은 전환 비용이 아니었다. decode가 굶으면 대기열이 막히는 연쇄가 원인이었다. 이번 실험에서는 나눠 놓은 decode가 느려지는 이유를 "SM 부족"과 "prefill과의 동거"로 나누고, hybrid 모델에서 동거 시간이 실제로 얼마나 되는지를 정의된 방식으로 센다(4-a).', 'c2 what')
# 기여 3
s = rep(s, '모델 구조를 아는 배분 선택과 운영 가이드</h3>', '운영 가이드와 그 여유의 크기</h3>', 'c3 title')
s = rep_between(s, '<p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> 기여 1의 비용 지도와 기여 2의 제약 위에서',
    '</p>',
    '<p style="font-size:13.5px;color:var(--ink-2)"><b>무엇을 보이는가.</b> 지금 바로 쓸 수 있는 운영 규칙이 있다. 최고 decode 부하를 기준으로 decode 쪽에 넉넉한 고정 배분을 쓰고, 검증된 모델 프로파일이 없으면 엔진의 기본 배분 선택기를 쓴다(확정). 이어서 워크로드마다 가장 좋은 고정 배분이 이 규칙보다 얼마나 나은지를 잰다(5-a). 이 차이는 어떤 똑똑한 배분 선택 방법을 만들더라도 얻을 수 있는 이득의 상한이 된다. 그 차이가 충분히 크면, 모델의 attention/SSM 구성과 decode 기준점을 아는 배분 선택이 고정 배분과 일반 동적 배분을 넘는지를 비교한다(5-b).', 'c3 what')
s = rep_between(s, '<p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> 새 hybrid 모델이 나왔을 때',
    '</p>',
    '<p style="font-size:13.5px;color:var(--ink-2);margin-top:8px"><b>기대 효과.</b> 운영자는 "고정 배분을 쓰면 무엇을 얼마나 포기하는가"를 수치로 갖게 된다. 비교까지 가서 이기면 "기준점을 알면 배분 선택이 산다"가 결과가 되고, 지면 "기준점을 알아도 고정 배분이 낫다"가 결과가 된다. 어느 쪽이든 운영 가이드의 근거로 남는다. 새 모델에 구성 정보만으로 배분을 고르는 것은 구성이 다른 여러 모델로 검증한 뒤의 확장이다.', 'c3 effect')
# closing
s = rep(s, '기여 1·2는 지금 자산과 2–4단계로 완성되고, 기여 3의 비교 부분은 5단계가 열릴 때 더해진다.',
           '기여 1·2는 지금 자산과 2–4단계 측정으로 완성될 예정이다(2단계에서 분할이 새 환경에서도 그대로 잡힌다는 것이 확인되는 경우). 기여 3의 여유 측정과 비교는 5단계가 열릴 때 더해진다.', 'closing')
open(p,'w',encoding='utf-8').write(s); print('report ok')

# ================= REVIEW =================
p='pdmux_review.html'; s=open(p,encoding='utf-8').read()
s = s.replace('검증된 프로파일이 없으면 균등', '검증된 프로파일이 없으면 엔진 기본 배분 선택기(decode 배치 크기 문턱 표, agnostic)')
i = s.index('검증된 모델 프로파일이 없으면', s.index('실무 권고(확정)'))
seg = s[i:i+80]; print('line107 seg:', seg)
open(p,'w',encoding='utf-8').write(s)
