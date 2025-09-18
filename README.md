# <i> 'Aariz: </i> 자동 두부 계측 랜드마크 감지 및 CVM 단계 분류를 위한 벤치마크 데이터셋
<p align="justify">
두부 계측 랜드마크의 정확한 식별 및 정밀한 위치 파악은 해부학적 이상을 분류하고 정량화하는 데 필수적입니다. 측면 두부 방사선 사진에 두부 계측 랜드마크를 수동으로 표시하는 전통적인 방식은 단조롭고 시간이 많이 소요되는 작업입니다. 자동 랜드마크 감지 시스템을 개발하려는 노력은 꾸준히 이루어졌지만, 신뢰할 수 있는 데이터셋의 부족으로 인해 교정 치료 적용에는 부적합했습니다. 우리는 정량적 형태 분석을 위한 견고한 AI 솔루션 개발을 촉진하기 위해 새로운 최첨단 데이터셋을 제안했습니다. 이 데이터셋은 7가지 다른 방사선 영상 장치에서 다양한 해상도로 얻은 1000개의 측면 두부 방사선 사진(LCR)을 포함하며, 현재까지 가장 다양하고 포괄적인 두부 계측 데이터셋입니다. 우리 팀의 임상 전문가들은 각 방사선 사진에 29개의 두부 계측 랜드마크를 세심하게 주석 처리했으며, 여기에는 공개적으로 사용 가능한 어떤 데이터셋에서도 표시된 적 없는 가장 중요한 연조직 랜드마크가 포함됩니다. 또한, 우리 전문가들은 방사선 사진에서 환자의 경추골 성숙(CVM) 단계를 라벨링하여, 이 데이터셋을 CVM 분류를 위한 최초의 표준 리소스로 만들었습니다. 각 영상 장치에서 얻은 방사선 사진은 훈련, 검증 및 테스트 세트로 균등하게 분배되었습니다. 우리는 이 데이터셋이 교정 치료 및 그 이상의 분야에서 신뢰할 수 있는 자동 랜드마크 감지 프레임워크 개발에 중요한 역할을 할 것이라고 믿습니다.

우리 데이터셋의 주요 특징은 다음과 같습니다:
  <ul>
    <li> 우리 데이터셋은 다양한 해상도를 가진 <strong> 7가지 다른 X선 영상 장치 </strong>에서 획득한 <strong> 1000개의 두부 방사선 사진 </strong>으로 구성된 다양하고 광범위한 컬렉션을 자랑하며, 현재까지 가장 포괄적인 두부 계측 데이터셋입니다. </li>
    <li> 이 데이터셋은 <strong> 29개의 가장 일반적으로 사용되는 해부학적 랜드마크 </strong>를 특징으로 하며, 15개의 골격, 8개의 치아, 6개의 연조직 랜드마크가 포함되어 있습니다. 이들은 광범위한 라벨링 및 검토 프로토콜에 따라 6명의 숙련된 교정의 팀에 의해 두 단계에 걸쳐 주석 처리되었습니다. </li>
    <li> 우리 데이터셋의 각 두부 방사선 사진에 대한 <strong> CVM 단계 </strong>를 주석 처리함으로써, 우리는 자동 CVM 분류를 위한 최초의 표준 리소스를 만들었습니다. </li>
  </ul>
</p>

<div align="center">
  <img src="docs/dataset-example-images.svg">
</div>
<div align="center"> 다양한 영상 장치에서 얻은 샘플 이미지와 해당 두부 계측 랜드마크 및 CVM 단계의 다양한 컬렉션 </div>

# 사용 가이드
데이터셋 클래스의 객체를 생성하고 이를 사용하여 이미지와 해당 주석을 읽으려면, 먼저 <code>AarizDataset</code>을 임포트한 다음, 이미지와 주석이 포함된 <code>dataset_folder_path</code>, 파일을 읽을 <code>mode</code>(예: <code>TRAIN</code>, <code>VALID</code>, <code>TEST</code>)와 같은 적절한 매개변수로 클래스를 인스턴스화할 수 있습니다.
```
from dataset import AarizDataset
dataset = AarizDataset(dataset_folder_path="folder/to/dataset", mode="TRAIN")
images, landmarks, cvm_stages = dataset[0]
```

# CEPHA29 챌린지 2023
형태 분석을 위한 견고한 AI 솔루션 개발을 촉진하기 위해, 우리는 <a href="https://2023.biomedicalimaging.org/en/CHALLENGES.html">IEEE 국제 생체 의료 영상 심포지엄</a> (ISBI 2023)과 함께 <a href="http://vision.seecs.edu.pk/CEPHA29/">CEPHA29 자동 두부 계측 랜드마크 감지 챌린지</a>를 개최했습니다. 이는 해당 분야의 연구원과 실무자들이 표준화된 플랫폼에서 자신들의 알고리즘을 테스트할 수 있는 좋은 기회입니다. 우리는 데이터셋을 챌린지로 제공함으로써, 해당 분야의 문제에 대한 새롭고 혁신적인 솔루션 개발을 촉진할 수 있다고 믿습니다.

# 감사
이 데이터셋의 주석 작업에 대한 끊임없는 노력과 기여에 대해 <a href="https://www.riphah.edu.pk/dental-sciences/">Islamic International Dental College</a>의 모든 임상의들에게 진심으로 감사드립니다. 이 연구는 파키스탄 이슬라마바드의 <a href="https://www.riphah.edu.pk">Riphah International University</a>의 지원 없이는 불가능했을 것입니다. 또한, 두부 계측 이미지 사용에 동의해 주신 환자분들께도 진심으로 감사드립니다. 마지막으로, 이 데이터셋의 품질 향상에 도움을 주신 익명의 검토자들의 귀중한 피드백과 제안에 감사드립니다.

# 인용
본 연구에 도움이 되었다면, 저희 <a href="https://arxiv.org/pdf/2302.07797.pdf">데이터셋</a>을 연구에 인용해 주시면 감사하겠습니다.
```
@article{
  title={'Aariz: A Benchmark Dataset for Automatic Cephalometric Landmark Detection and CVM Stage Classification},
  author={Khalid, Muhammad Anwaar and Zulfiqar, Kanwal and Bashir, Ulfat and Shaheen, Areeba and Iqbal, Rida and Rizwan, Zarnab and Rizwan, Ghina and Fraz, Muhammad Moazam}
  journal={arXiv preprint arXiv:2302.07797},
  year={2023}
}
```