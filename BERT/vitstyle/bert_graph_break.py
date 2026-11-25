#!/usr/bin/env python3
"""torch._dynamo.explain으로 측정 경로의 graph break 확인 (CPU 전용, NPU 안 씀).

측정과 같은 설정: BF16 모델, create_bidirectional_mask passthrough 패치,
CPU 4D additive BF16 마스크, 길이 128, int32 input_ids/token_type_ids,
int32 position_ids 명시 전달(--posid-int32 경로), inference_mode.
backend 없이 dynamo 캡처만 하므로 NPU가 필요 없다.

batch마다 torch._dynamo.reset() 후 explain을 돌려 그래프 개수, graph break
개수와 위치·이유, 그래프별 op 수를 출력한다.

사용법: HF_HUB_OFFLINE=1 python bert_graph_break.py [--batches 1,16]
"""
import argparse

import torch

from bert_model import load_model, encode, SENTENCES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batches', type=lambda s: [int(x) for x in s.split(',')], default=[1, 16])
    a = ap.parse_args()
    print('torch', torch.__version__)
    tok, model = load_model()
    for B in a.batches:
        inp = encode(tok, [SENTENCES[i % len(SENTENCES)] for i in range(B)], posid_int32=True)
        kwargs = {k: inp[k] for k in ['input_ids', 'token_type_ids', 'attention_mask', 'position_ids']}
        torch._dynamo.reset()   # batch 간 캐시/카운터 공유 방지
        with torch.inference_mode():
            ex = torch._dynamo.explain(model)(**kwargs)
        print(f'\n=== batch {B} ===')
        print('graph_count:', ex.graph_count)
        print('graph_break_count:', ex.graph_break_count)
        print('ops_per_graph:', [len(ops) for ops in ex.ops_per_graph])
        for i, br in enumerate(ex.break_reasons, 1):
            print(f'[break {i}] {br.reason}')
            for fr in br.user_stack:
                print(f'    at {fr}')
        if not ex.break_reasons:
            print('(graph break 없음)')


if __name__ == '__main__':
    main()
