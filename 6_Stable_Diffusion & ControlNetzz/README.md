![](controlNet/rawfile.png) ![](controlNet/image.png) 
![](controlNet/ladyfashion.png) ![](controlNet/person_mt.png)
<img src="controlNet/mangolf.png" width="324" height="512"> <img src="controlNet/mangolf_openpose.png" width="324" height="512">


## @controlNet_colab.ipynb 프로젝트 소개

이 프로젝트에서는 Stable Diffusion과 ControlNet을 결합하여 더 정교하고 세밀한 이미지 생성을 시도합니다. Colab 환경에서 빠르게 실행할 수 있도록 @controlNet_colab.ipynb 노트북을 준비했으며, 아래 과정을 거쳐 이미지 생성과 제어 과정을 체험하실 수 있습니다.

### 주요 단계
1. **환경 설정**: 노트북 초기 설정 후, 필요한 라이브러리를 설치하고 Stable Diffusion 모델과 ControlNet 모델을 로드합니다.  
2. **데이터 준비**: 원하는 이미지나 텍스트 프롬프트를 구성하여 ControlNet에 전달할 입력 이미지를 준비합니다.  
3. **Inference 실행**: 노트북에서 제공되는 코드 셀을 실행하여 Prompt, Guidance Scale, Control 필터 등을 설정한 뒤 이미지를 생성합니다.  
4. **결과 확인**: 생성된 이미지가 출력되며, ControlNet을 적용했을 때와 적용하지 않았을 때 어떤 차이가 있는지 비교할 수 있습니다.

### ControlNet 컨트롤러 종류
본 프로젝트에서는 주로 두 가지 ControlNet 컨트롤러를 사용합니다:

1. **Canny Edge Detection**
   - 입력 이미지의 윤곽선을 감지하여 이미지 생성의 기준으로 활용
   - 이미지의 주요 형태와 구조를 유지하면서 새로운 스타일 적용 가능
   - threshold1, threshold2 파라미터로 edge 감지 강도 조절 가능

2. **OpenPose**
   - 인물의 자세와 포즈를 감지하여 이미지 생성 가이드로 활용
   - 사람의 관절 위치와 자세를 인식하여 유사한 포즈의 이미지 생성
   - 얼굴, 손, 발 등의 키포인트도 함께 감지 가능


실행 과정이나 생성 과정에서 궁금한 점이 있다면, @controlNet_colab.ipynb 노트북 내부의 주석 및 추가 설명을 참고해 주세요.  
원활한 실행을 위해 Colab 환경 사용 시 런타임 유형을 GPU로 설정하는 것을 권장합니다.
