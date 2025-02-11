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
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def chat():
    print("챗봇과 대화를 시작합니다. (종료하려면 'quit' 또는 '종료'를 입력하세요)")
    print("=" * 50)
    
    model, tokenizer = load_model_and_tokenizer()
    context = ""
    
    while True:
        user_input = input("\n사용자: ")
        
        if user_input.lower() in ['quit', '종료']:
            print("\n대화를 종료합니다.")
            break
            
        if context:
            prompt = f"{context}\n사용자: {user_input}\n챗봇:"
        else:
            prompt = f"사용자: {user_input}\n챗봇:"
            
        try:
            response = generate_text(model, tokenizer, prompt, max_length=100)
            
            if "챗봇:" in response:
                response = response.split("챗봇:")[-1].strip()
            
            print(f"챗봇: {response}")
            
            context = f"{prompt} {response}"
            if len(context.split('\n')) > 4:
                context = '\n'.join(context.split('\n')[-4:])
                
        except Exception as e:
            print(f"오류가 발생했습니다: {str(e)}")
            continue

if __name__ == "__main__":
    chat() 