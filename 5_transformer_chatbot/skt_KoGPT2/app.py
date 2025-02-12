from flask import Flask, render_template, request, jsonify
from skt_kogpt2 import load_model_and_tokenizer, generate_text

app = Flask(__name__)

# 모델과 토크나이저 로드
model, tokenizer = load_model_and_tokenizer()

# 시스템 프롬프트
SYSTEM_PROMPT = """이것은 사용자와 AI 챗봇의 대화입니다.
챗봇은 다음과 같은 규칙을 따릅니다:
- 항상 명확하고 자연스러운 한국어로 대답합니다
- 이모티콘이나 특수문자를 과도하게 사용하지 않습니다
- 한 번에 한 문장으로 간단히 대답합니다
- 친근하고 공손한 어투를 사용합니다"""

# 대화 기록 저장을 위한 딕셔너리 (세션별로 관리)
conversation_histories = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    # 세션별 대화 기록 가져오기 또는 새로 생성
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    conversation_history = conversation_histories[session_id]
    
    # 대화 기록이 너무 길어지면 초기화
    if len(conversation_history) >= 50:
        return jsonify({
            'response': '대화 한도에 도달했습니다. 새로고침하여 새로운 대화를 시작해주세요.',
            'reset': True
        })
    
    # 대화 기록에 사용자 입력 추가
    conversation_history.append(f"사용자: {user_message}")
    
    # 최근 10개의 대화만 사용
    recent_history = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
    prompt = SYSTEM_PROMPT + "\n\n" + "\n".join(recent_history) + "\n챗봇:"
    
    try:
        # 입력 길이 체크
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        if len(input_ids[0]) >= 900:
            return jsonify({
                'response': '대화가 너무 길어져서 더 이상 진행하기 어렵습니다. 새로고침하여 새로운 대화를 시작해주세요.',
                'reset': True
            })
        
        # 응답 생성
        response = generate_text(model, tokenizer, prompt)
        
        # 응답 정제
        if "챗봇:" in response:
            response = response.split("챗봇:")[-1].strip()
        response = response.split("\n")[0]
        response = response.replace("AI:", "").replace("AI :", "").strip()
        
        # 대화 기록에 챗봇 응답 추가
        conversation_history.append(f"챗봇: {response}")
        
        return jsonify({
            'response': response,
            'reset': False
        })
        
    except Exception as e:
        return jsonify({
            'response': f'오류가 발생했습니다: {str(e)}',
            'reset': False
        })

if __name__ == '__main__':
    app.run(debug=True) 