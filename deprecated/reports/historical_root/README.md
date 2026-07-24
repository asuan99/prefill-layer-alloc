# Root reports archive

이 디렉터리는 주로 characterization, simulator, 초기 vLLM validation과 철회된
layer-aware 논문 트랙의 historical artifact다. 현재 프로젝트/논문 정본은
[`../PROJECT_STATUS.md`](../../../PROJECT_STATUS.md)와
[`../workspace/engine-port/reports/paper/`](../../../reports/paper)다.

특히 `PAPER_RESULTS.md`, `FINAL_SUMMARY.md` 및 layer-aware positive reports의
goodput 배수는 현재 real-engine claim으로 사용하지 않는다. 각 문서의 커널
측정이나 simulator 결과를 인용할 때는 해당 fidelity와 CUDA Graph/engine 조건을
명시한다.
