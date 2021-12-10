# -*- coding: utf-8 -*
import torch
from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration

hash_funcs={torch.nn.parameter.Parameter: lambda _: None}
def load_model():
    model = BartForConditionalGeneration.from_pretrained('./kobart_summary')
    # tokenizer = get_kobart_tokenizer()
    return model

model = load_model()
tokenizer = get_kobart_tokenizer()

text = "김민재 외(2021)는 중학생의 창의적 문제해결력과 컴퓨팅 사고력에 데이터 리터러시를 활용한 소프트웨어 교육이 미치는 영향을 분석하기 위해 ‘미세먼지 예측 프로그램’이라는 주제로 CT-CPS 모형 기반 수업을 설계하여 적용하였다. 그 결과로 중학생의 창의적 문제해결력과 컴퓨팅 사고력 향상에 데이터 리터러시를 활용한 소프트웨어 교육이 유의미한 효과가 있음을 검증하였다[7]. 해당 연구는 간단한 데이터 예측 프로그램만 적용했다는 한계가 있다. 그래서, 본 연구에서는 CT-CPS 모형이 아닌 문제중심학습 모형을 적용하여 데이터 분석, 통계 및 시각화를 통해서 공공데이터를 해석하고 학사운영을 조정하는 안을 만드는 등 실질적인 데이터 리터러시 향상을 위한 수업을 개발하였다. 송유경(2021)은 기존의 데이터 과학 교육과 토론 교육을 융합하여 학습자가 직접 분석한 데이터를 기반으로 토론 활동에 참여하는 데이터 기반 토론 수업을 시도하고, 실제적으로 적용하는데에 도움을 주는 수업 모형과 교수 전략을 개발하였다. 그 결과로 개발한 수업모형과 교수 전략이 데이터 리터러시 향상에 도움을 주었다는 것을 증명하였다[8]."

input_ids = tokenizer.encode(text)
input_ids = torch.tensor(input_ids)
input_ids = input_ids.unsqueeze(0)

output = model.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
output = tokenizer.decode(output[0], skip_special_tokens=True)

print(output)
