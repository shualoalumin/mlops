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

def generate_text(model, tokenizer, prompt, max_length=50):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=max_length,
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
    return tokenizer.decode(output[0], skip_special_tokens=True)

def chat():
    print("챗봇과 대화를 시작합니다. (종료하려면 'quit' 또는 '종료'를 입력하세요)")
    print("=" * 50)
    
    model, tokenizer = load_model_and_tokenizer()
    system_prompt = """이것은 사용자와 AI 챗봇의 대화입니다.
챗봇은 다음과 같은 규칙을 따릅니다:
- 항상 명확하고 자연스러운 한국어로 대답합니다
- 이모티콘이나 특수문자를 과도하게 사용하지 않습니다
- 한 번에 한 문장으로 간단히 대답합니다
- 친근하고 공손한 어투를 사용합니다

대화 예시:
사용자: 안녕하세요
챗봇: 안녕하세요! 만나서 반갑습니다.

사용자: 이름이 뭐예요?
챗봇: 저는 AI 어시스턴트입니다. 편하게 대화하실 수 있어요.

"""
    context = system_prompt
    
    while True:
        user_input = input("\n사용자: ")
        
        if user_input.lower() in ['quit', '종료']:
            print("\n대화를 종료합니다.")
            break
            
        prompt = f"{context}사용자: {user_input}\n챗봇:"
        
        try:
            response = generate_text(model, tokenizer, prompt, max_length=100)
            
            if "챗봇:" in response:
                response = response.split("챗봇:")[-1].strip()
            
            print(f"챗봇: {response}")
            
            context = system_prompt + prompt.replace(system_prompt, "") + " " + response + "\n"
            context_lines = context.split('\n')
            if len(context_lines) > 10:
                context = system_prompt + "\n".join(context_lines[-8:]) + "\n"
                
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")
            continue

if __name__ == "__main__":
    chat() 