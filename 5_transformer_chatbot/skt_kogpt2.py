from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

def load_model_and_tokenizer():
    TOKENIZER_NAME = "skt/kogpt2-base-v2"
    MODEL_NAME = "skt/kogpt2-base-v2"

    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        TOKENIZER_NAME,
        bos_token='<s>',
        eos_token='</s>',
        unk_token='<unk>',
        pad_token='<pad>',
        mask_token='<mask>'
    )
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=1024):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # 입력 길이
    input_length = len(input_ids[0])
    
    output = model.generate(
        input_ids,
        max_length=max_length,
        max_new_tokens=50,  # 새로 생성할 최대 토큰 수
        num_beams=3,
        temperature=0.8,
        top_k=40,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True
    )
    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True)

def chat():
    print("챗봇과 대화를 시작합니다. (종료하려면 'quit' 또는 '종료'를 입력하세요)")
    print("=" * 50)
    
    model, tokenizer = load_model_and_tokenizer()
    system_prompt = """이것은 사용자와 AI 챗봇의 대화입니다.
챗봇은 다음과 같은 규칙을 따릅니다:
- 항상 명확하고 자연스러운 한국어로 대답합니다
- 이모티콘이나 특수문자를 과도하게 사용하지 않습니다
- 한 번에 한 문장으로 간단히 대답합니다
- 친근하고 공손한 어투를 사용합니다"""

    conversation_history = []
    MAX_CONVERSATIONS = 50  # 최대 대화 턴 수
    
    while True:
        if len(conversation_history) >= MAX_CONVERSATIONS:
            print("\n대화 한도에 도달했습니다.")
            print("더 나은 대화를 위해 새로운 대화를 시작해주세요.")
            print("대화를 종료합니다.")
            break
            
        user_input = input("\n사용자: ")
        
        if user_input.lower() in ['quit', '종료']:
            print("\n대화를 종료합니다.")
            break
        
        # 대화 기록에 사용자 입력 추가
        conversation_history.append(f"사용자: {user_input}")
        
        # 프롬프트 생성 (시스템 프롬프트 + 최근 10개의 대화만 사용)
        recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        prompt = system_prompt + "\n\n" + "\n".join(recent_history) + "\n챗봇:"
        
        try:
            # 입력 길이 체크
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            if len(input_ids[0]) >= 900:  # max_length(1024)에 여유를 둠
                print("\n죄송합니다. 대화가 너무 길어져서 더 이상 진행하기 어렵습니다.")
                print("더 나은 대화를 위해 새로운 대화를 시작해주세요.")
                print("대화를 종료합니다.")
                break
                
            response = generate_text(model, tokenizer, prompt)
            
            # 응답 정제
            if "챗봇:" in response:
                response = response.split("챗봇:")[-1].strip()
            
            # 첫 번째 줄만 사용하고 불필요한 텍스트 제거
            response = response.split("\n")[0]
            response = response.replace("AI:", "").replace("AI :", "").strip()
            
            print(f"챗봇: {response}")
            
            # 대화 기록에 챗봇 응답 추가
            conversation_history.append(f"챗봇: {response}")
                
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")
            continue

if __name__ == "__main__":
    chat() 