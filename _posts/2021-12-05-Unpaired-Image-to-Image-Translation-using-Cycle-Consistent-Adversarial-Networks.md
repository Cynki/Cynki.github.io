---
title: "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks"
excerpt: "이미지-이미지 변환은 결합된 이미지 쌍으로 구성된 훈련 세트를 사용하여 입력과 출력 이미지 간의 매핑을 하는 것이 목표인 비전 및 그래픽 문제의 분야이다. 하지만 많은 작업들에서 쌍을 이루는 훈련 데이터(paired training data)를 준비하는 것이 가능하지 않다. 우리는 이러한 훈련 데이터 없이 이미지의 원래의 영역 $X$에서 목표 영역인 $Y$로의 변환을 학습하는 것에 대한 접근법을 소개한다. 우리의 목표는 adversarial loss를 사용하여 $G(X)$와 분포 $Y$를 구별할 수 없도록 매핑 $G:X\rightarrow Y$를 학습하는 것이다. 이 매핑에는 제약이 상당히 적으므로 우리는 이에 대한 역 매핑 $F:Y\rightarrow X$를 연결지어 생각하며 이에 따라 $F(G(X)) \approx X$를 강제시키기 위해 (반대도 마찬가지) cycle consistency를 도입한다. Collection style transfer, object transfiguration, season transfer, photo enhancement 등을 포함한, paired training data가 없는 몇 가지 작업에 대한 정성적 결과가 소개되어 있다. 이전의 몇 가지 방법에 대한 정량적인 비교는 우리의 접근법의 우월성을 입증한다."
categories:
  - GAN Study
tags:
  - PseudoLab
  - GAN
  - CycleGAN
  - Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
# table of contents
toc: true # 오른쪽 부분에 목차를 자동 생성해준다.
toc_label: "table of content" # toc 이름 설정
toc_icon: "bars" # 아이콘 설정
toc_sticky: true # 마우스 스크롤과 함께 내려갈 것인지 설정
---

[Paper](https://junyanz.github.io/CycleGAN/)

![Figure 1: 주어진 순서 없는 이미지 집합 $X$와 $Y$에 대해, 우리의 알고리즘은 이미지를 한 편에서 반대편으로 (양쪽에서) 자동으로 "변환하는" 방법을 학습한다. (왼쪽) Monet의 그림과 Flickr에서 수집한 풍경 사진; (가운데) ImageNet의 얼룩말과 말; (오른쪽) Flickr에서 수집한 Yosemite 국립공원의 여름과 겨울 사진. (아래) 응용  예제에서 우리의 방법은 유명한 예술가들의 그림 집합을 이용하여 자연 사진을 그에 관련한 스타일로 렌더링하는 것을 배운다.](/assets/images/2021-12-05/Untitled.png)

Figure 1: 주어진 순서 없는 이미지 집합 $X$와 $Y$에 대해, 우리의 알고리즘은 이미지를 한 편에서 반대편으로 (양쪽에서) 자동으로 "변환하는" 방법을 학습한다. (왼쪽) Monet의 그림과 Flickr에서 수집한 풍경 사진; (가운데) ImageNet의 얼룩말과 말; (오른쪽) Flickr에서 수집한 Yosemite 국립공원의 여름과 겨울 사진. (아래) 응용  예제에서 우리의 방법은 유명한 예술가들의 그림 집합을 이용하여 자연 사진을 그에 관련한 스타일로 렌더링하는 것을 배운다.

# Abstract

*이미지-이미지 변환은 결합된 이미지 쌍으로 구성된 훈련 세트를 사용하여 입력과 출력 이미지 간의 매핑을 하는 것이 목표인 비전 및 그래픽 문제의 분야이다. 하지만 많은 작업들에서 쌍을 이루는 훈련 데이터(paired training data)를 준비하는 것이 가능하지 않다. 우리는 이러한 훈련 데이터 없이 이미지의 원래의 영역 $X$에서 목표 영역인 $Y$로의 변환을 학습하는 것에 대한 접근법을 소개한다. 우리의 목표는 adversarial loss를 사용하여 $G(X)$와 분포 $Y$를 구별할 수 없도록 매핑 $G:X\rightarrow Y$를 학습하는 것이다. 이 매핑에는 제약이 상당히 적으므로 우리는 이에 대한 역 매핑 $F:Y\rightarrow X$를 연결지어 생각하며 이에 따라 $F(G(X)) \approx X$를 강제시키기 위해 (반대도 마찬가지) cycle consistency를 도입한다. Collection style transfer, object transfiguration, season transfer, photo enhancement 등을 포함한, paired training data가 없는 몇 가지 작업에 대한 정성적 결과가 소개되어 있다. 이전의 몇 가지 방법에 대한 정량적인 비교는 우리의 접근법의 우월성을 입증한다.*

# 1. Introduction

Claude Monet이 1873년의 멋진 봄날에 Argenteuil 주변의 Seine 강변에 이젤을 놓았을 때 그는 어떠한 장면을 보고 있었을까(Figure 1, 좌상면)? 그때 컬러 사진이 발명되었다면 화창한 푸른 하늘과 이를 비추는 유리같은 강을 담아냈을 것이다. Monet은 이 장면에 대한 자신의 *인상*을 섬세한 붓놀림과 밝은 팔레트의 색채로 전달해냈다.

만약 Monet이 시원한 여름 저녁에 Cassis의 작은 항구에서 이젤을 놓았다면 어땠을까(Figure 1, 좌하면)? Monet의 그림들을 잠시 둘러보면 그가 어떻게 그 장면을 표현했을지 상상할 수 있다. 아마도 파스텔 색조로, 가벼운 붓칠과 다소 좁은 범위의 색을 사용했을 것이다.

우리는 이러한 과정을 사진과 그 장면에 대한 그의 그림을 나란히 볼 필요 없이 상상할 수 있다. 그러는 대신, 우리는 일련의 Monet 그림과 일련의 풍경 사진을 기억하고 있다. 우리는 이 두 집합의 스타일적인 차이를 추론해낼 수 있으며, 이에 따라 한 집합에서 다른 집합으로 "변환"을 할 때 가능한 장면을 상상할 수 있다.

이 논문에서 우리는 위와 똑같이 해내도록 학습하는 방법을 제시한다. 한 이미지 집합의 특정한 특성을 포착해 이 특성이 다른 이미지 집합에 어떠한 방식으로 변환되어야 하는지를 알아낸다. 이러한 작업은 어떠한 paired training example 없이 이루어진다.

![Figure 2: 왼쪽의 Paired training data는 훈련 예제 $$\{x_i, y_i\}_{i=1}^N$$이며, 여기서 $$x_i$$와 $$y_i$$ 사이의 관련성이 존재한다. 우리는 그 대신 오른쪽의 unpaired training data를 다루며, 이 데이터는 source 집합 $$\{x_i\}_{i=1}^N(x_i \in X)$$과 target 집합 $$\{y_i\}_{j=1}^N(y_j \in Y)$$를 포합하고 어떤 $$x_i$$가 어떤 $$y_j$$와 일치하는 지에 대한 정보는 주어지지 않는다.](/assets/images/2021-12-05/Untitled%201.png)

Figure 2: 왼쪽의 Paired training data는 훈련 예제 $$\{x_i, y_i\}_{i=1}^N$$이며, 여기서 $$x_i$$와 $$y_i$$ 사이의 관련성이 존재한다. 우리는 그 대신 오른쪽의 unpaired training data를 다루며, 이 데이터는 source 집합 $$\{x_i\}_{i=1}^N(x_i \in X)$$과 target 집합 $$\{y_i\}_{j=1}^N(y_j \in Y)$$를 포합하고 어떤 $$x_i$$가 어떤 $$y_j$$와 일치하는 지에 대한 정보는 주어지지 않는다.

이 문제는 주어진 장면에 대한 한 쪽의 이미지 $x$에 대해 다른 표현인 $y$로 전환시킨다는 점에서 이미지-이미지 변환으로서 더 넓게 설명될 수 있다. 예를 들면 흑백에서 컬러로, 이미지에서 시멘틱 레이블로, 엣지 맵에서 사진으로 변환하는 작업이 해당된다. 컴퓨터 비전, 이미지 처리, computational photography, 그리고 그래픽스에서의 수년간의 연구는 이미지 쌍 $\{x_i, y_i\}_{i=1}^N$이 존재하는 지도 학습 설정 아래에서 강력한 변환 시스템을 만들어왔다. 그러나 paired training data를 구하는 것이 어렵고 비효율적일 수 있다. 시멘틱 세그멘테이션과 같은 작업에 매우 적은 수의 데이터만이 존재할 수 있다. 예술적 스타일화와 같은 그래픽스 작업에서 입출력 쌍을 얻어내는 일은 의도하는 출력이 매우 복잡하고, 일반적으로 예술적인 창작을 요구하므로 더 어렵다. Object transfiguration (얼룩말 ↔ 말) 과 같은 많은 작업에서는 의도로 하는 출력이 심지어 잘 정의되지도 않는다.

 따라서 우리는 짝지어진 입출력 예제들이 없는 영역간의 변환을 학습할 수 있는 알고리즘을 찾는다. 우리는 영역 사이에 기저하는 관계가 존재한다고 가정하며 ─ 예를 들면 똑같은 장면에 대한 두 가지 다른 렌더링이 있다 ─ 그리고 그 관계를 학습하는 방법을 찾는다. 짝지어진 예제에 대한 supervision은 없지만, 집합의 단계에서는 supervision을 이용할 수 있다 (영역 $X$에 대한 이미지 집합과 영역 $Y$에 대한 다른 이미지 집합이 주어진다). 우리는 출력 $\hat{y}=G(x), x\in X$과 이미지 $y\in Y$를 구별하도록 훈련받는 경쟁자에 대해 매핑 $G:X\rightarrow Y$를 경쟁자가 $\hat{y}$와 $y$를 구별하지 못하도록 훈련시킬 수 있다. 이론적으로, 이러한 목적(함수)는 경험적 분포 $p_{data}(y)$에 일치하는 $\hat{y}$에 대한 출력 분포를 유도해낼 수 있다 (일반적으로 이를 위해 $G$가 확률적이어야 한다). 따라서 최적의 $G$는 영역 $X$를 $Y$에 동일하게 분포된 영역 $\hat{Y}$으로 변환한다. 그러나, 이러한 변환은 각각의 입력 $x$와 출력 $y$가 의미있는 방향으로 짝지어진다고 보장하지 않는다. $\hat{y}$에 대한 똑같은 분포를 유도하는 무한히 많은 매핑 $G$가 존재한다. 게다가 실제로는, 고립되어 있는 적대적 목적 함수를 최적화하는 것이 어렵다는 사실을 알아냈다. 최적화의 표준적인 절차가 자주 잘 알려진 문제인 모드 붕괴(모든 입력 이미지가 똑같은 출력 이미지로 매핑되고 최적화가 진행되지 않는 현상)를 불러일으킨다. 

이러한 문제는 우리의 목적 함수에 더 복잡한 구조를 추가할 것을 요구한다. 따라서, 우리는 변환이 "cycle consistent"해야 한다는 특성을 이용하는데, 이는 예를 들어 우리가 문장을 영어에서 프랑스어로 번역하고 다시 프랑스어에서 영어로 번역할 경우 원래의 문장으로 돌아가야 한다는 것이다. 수학적으로 보았을 때, 변환기 $G:X\rightarrow Y$와 다른 변환기 $F:Y\rightarrow X$가 존재한다면, $G$와 $F$는 반드시 서로의 역변환(inverse)이어야 하며 두 매핑은 일대일 대응이어야 한다. 우리는 두 매핑 $G$와 $F$를 동시에 훈련시키고, $F(G(x)) \approx x$와 $G(F(y)) \approx y$를 만족하도록 하는 *cycle consistency loss*를 추가함으로 이러한 구조적 가정을 적용했다. 이러한 손실을 영역 $X$와 $Y$에 대한 적대적 손실과 합치는 것은 unpaired image-to-image translation에 대한 우리의 전체적인 목적 함수를 만들어낸다.

우리는 우리의 방법을 넓은 범위에 응용했는데, collection style transfer, object transfiguration, season transfer, 그리고 photo enhancement 등이 있다. 우리는 또한 이전의 수작업으로 정의된 style과 content, 또는 공유되는 임베딩 함수 등에 의존하는 접근법들과 비교하고 우리의 방법이 이러한 기준을 능가하는 것을 보여준다.

# 2. Related work

## Generative Adversarial Networks (GANs)

GAN은 이미지 생성, 편집, 그리고 representation learning에서 인상적인 결과를 보여주었다. 최근의 방법은 이같은 아이디어를 조건부 이미지 생성 어플리케이션에 적용했는데 예를 들면 text2image, image inpainting, 그리고 future prediction과 영상 및 3D 데이터와 같은 다른 영역이 있다. GAN의 성공의 핵심은 이론적으로 생성된 이미지가 실제  사진과 구별할 수 없도록 만드는 *adversarial loss*이다. 이 손실은 특히 이미지 생성 작업에 강한데, 이는 이 손실이 정확히 많은 컴퓨터 그래픽스 작업이 최적화하려는 것이기 때문이다. 우리는 이 adversarial loss를 채택하여 변환된 이미지가 타깃 영역의 이미지들과 구별할 수 없도록 매핑을 학습시킨다.

## Image-to-Image Translation

이미지-이미지 변환의 아이디어는 최소한 Hertzmann et al.의 Image Analogies까지 거슬러 올라가며, 여기서는 단일 입출력 훈련 이미지 쌍을 비파라미터적 텍스쳐 모델을도입했다. 더 최근의 접근은 CNN을 사용한 파라미터적 변환 함수를 학습하는데 입출력 예제의 *데이터 세트*를 사용한다. 우리의 접근법은 conditional GAN을 사용하여 입력과 출력 사이의 매핑을 학습하는 Isola et al.의 "pix2pix" 프레임워크를 기반으로 한다. 비슷한 아이디어가 다양한 작업에 적용되었는데 예를 들면 스케치에서 사진 생성 또는 특성으로부터 시멘틱 레이아웃 생성 등이 있다. 하지만 위의 이전 작업들과는 다르게, 우리는 paired training example 없이 매핑을 학습한다.

## Unpaired Image-to-Image Translation

몇 가지 다른 방법들도 두 영역 $X$와 $Y$를 관련짓기 위해 unpaired setting을 시도한다. Rosales et al.은 베이지안 프레임워크를 제안하는데, 이 프레임워크는 원 이미지에서 계산된 patch-based Markov random field에 기반하는 prior와 다중 스타일 이미지로부터 얻은 가능도 항을 포함한다. 더 최근에는 CoGAN과 cross-modal scene network는 가중치 공유 전략을 사용하여 영역 간의 공통된 representation을 학습한다. 우리의 방법과 동시에, Liu et al.은 변분 오토인코더와 GAN의 조합으로 위의 프레임워크를 홧장한다. 동시에 진행된 또다른 방향은 입력과 출력이 다른 "style"을 가지고 있더라도 특정한 "content" 특성을 공유하도록 한다. 이러한 방법들은 또한 적대 신경망을 사용하는데, class label space, image pixel space, 그리고 image feature space와 같이 미리 정의된 metric space에서 출력이 입력과 가까워지도록 추가적인 항들을 같이 사용한다.

위의 접근들과는 다르게, 우리의 공식은 어떠한 작업에 특정하고 미리 정의되는 입출력 사이의 유사도 함수에 의존하지 않으며, 입력과 출력이 같은 저차원 임베딩 공간에 있어야 한다는 가정도 하지 않는다. 이는 우리의 방법이 많은 비전과 그래픽 작업에 대한 범용의 해결 방안이 되도록 만든다. 우리는  Section 5.1에서 몇 가지 이전, 그리고 동시에 진행된 접근법들과 비교했다.

## Cycle Consistency

Transitivity를 구조화된 데이터를 일반화시키는 방법으로 사용하는 아이디어는 그 역사가 길다. Visual tracking에서, 간단한 forward-backward 일관성을 강제하는 것은 수십년간 표준 트릭이 되었다. 언어 영역에서, "back translation and reconciliation"을 통해 번역을 확인하고 향상시키는 것은 기계뿐만이 아니라 인간 번역자도 사용하는 기술이다. 더 최근에는 고차원 cycle consistency가 structure from motion, 3D shape matching, co-segmentation, dense semantic alignment, 그리고 depth estimation에 사용되었다. 이 중에서 Zhou et al.과 Godard et al.이 우리의 작업과 가장 비슷하며, 그 이유는 그들이 *cycle consistency loss*를 transitivity를 CNN 지도 학습에 사용하는 방법으로서 사용하는 데에 있다. 이 논문에서 우리는 비슷한 손실을 도입하여 $G$와 $F$가 서로 일관적이도록 만든다. 우리의 논문과 동시에, 이러한 같은 과정에서, Yi et al.은 기계 번환의 dual learning에서 영감을 받아 독립적으로 unpaired image-to-image transtlation을 위한 비슥한 목적 함수를 사용한다.

## Neural Style Transfer

Neural Style Transfer는 미리 학습된 심층 특성의 Gram matrix 통계량을 일치시켜 한 이미지의 content와 다른 이미지(일반적으로 그림)의 style을 합쳐 새로운 이미지를 합성하는, 일종의 이미지-이미지 변환을 수행하는 또다른 방법이다. 한편 우리의 첫 번쨰 목적은 고수준의 외관 구조 사이의 관련성을 포착하여 특정한 두 이미지가 아닌, 두 이미지 집합 사이의 매핑을 학습시키는 것이다. 따라서, 우리의 방법은 painting→photo, object transfiguration 등의 단일 샘플 변환 방법이 좋은 성능을 내지 못하는 작업에 적용될 수 있다. 우리는 Section 5.2에서 이 두 방법을 비교할 것이다.

# 3. Formulation

![Figure 3: (a) 우리의 모델은 두 매핑 함수 $G:X \rightarrow Y$와 $F:Y \rightarrow X$, 그리고 관련한 적대적 판별자 $D_Y$와 $D_X$를 포함한다. $D_Y$는 $G$가 $X$를 영역 $Y$로부터 구분할 수 없는 출력을 만들도록 하며, $D_X$와 $F$의 경우에도 마찬가지로 그렇다. 나아가 이 매핑을 일반화하기 위해 우리는 두 가지의 *cycle consistency loss*를 도입하며 이 손실은 한 영역에서 다른 영역으로 변환하고 다시 돌아올 때 시작한 곳에 있어야 한다는 직관을 잡아낸다. (b) forward cycle-consistency loss: $x \rightarrow G(x) \rightarrow F(G(x)) \approx x$ (c) backward cycle-consistency loss: $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$](/assets/images/2021-12-05/Untitled%202.png)

Figure 3: (a) 우리의 모델은 두 매핑 함수 $G:X \rightarrow Y$와 $F:Y \rightarrow X$, 그리고 관련한 적대적 판별자 $D_Y$와 $D_X$를 포함한다. $D_Y$는 $G$가 $X$를 영역 $Y$로부터 구분할 수 없는 출력을 만들도록 하며, $D_X$와 $F$의 경우에도 마찬가지로 그렇다. 나아가 이 매핑을 일반화하기 위해 우리는 두 가지의 *cycle consistency loss*를 도입하며 이 손실은 한 영역에서 다른 영역으로 변환하고 다시 돌아올 때 시작한 곳에 있어야 한다는 직관을 잡아낸다. (b) forward cycle-consistency loss: $x \rightarrow G(x) \rightarrow F(G(x)) \approx x$ (c) backward cycle-consistency loss: $y \rightarrow F(y) \rightarrow G(F(y)) \approx y$

우리의 목표는 주어진 훈련 샘플들 $$\{x_i\}_{i=1}^N, x_i\in X$$와 $$\{y_j\}_{j=1}^M, y_j\in Y$$(편의성을 위해 간혹 아래첨자 $i$와 $j$를 생략한다)를 통해 두 영역 $X$와 $Y$ 사이의 매핑 함수를 학습하는 것이다. 데이터 분포를 $$x \sim p_\text{data}(x)$$와 $$y \sim p_\text{data}(y)$$로 정의한다. Figure 3 (a)에서 보이는 대로, 우리의 모델은 두 매핑 $G:X\rightarrow Y$와 $F:Y\rightarrow X$를 포함한다. 거기에 더해서, 두 적대적 판별자 $D_X$와 $D_Y$를 도입하며 이때 $D_X$는 이미지들 $\{x\}$와 변환된 이미지 $\{F(y)\}$를 구별하는 것이 목적이며, $D_Y$도 같은 방식으로 $\{y\}$와 $\{G(x)\}$를 판별하는 것이 목적이다. 우리의 목적 함수는 두 가지의 항을 포함한다. 하나는 *adversarial loss*로 생성된 이미지의 분포를 타깃 영역의 데이터 분포에 일치시키며, 다른 하나는 *cycle consistency loss*로 학습된 매핑 $G$와 $F$가 서로 모순되지 않도록 한다.

## 3.1 Adversarial Loss

우리는 adversarial loss를 양쪽의 매핑 함수에 적용한다. 매핑 함수 $G:X \rightarrow Y$와 그 판별자 $D_Y$에 대한 목적 함수는 다음과 같다.

$$\begin{align}
\mathcal{L}_\text{GAN}(G,D_Y,X,Y)&=\mathbb{E}_{y\sim p_\text{data}(y)}[\log D_Y(y)] \\&+
\mathbb{E}_{x\sim p_\text{data}(x)}[\log 1-D_Y(G(x))].
\end{align}$$

이 때 $G$는 영역 $Y$의 이미지와 비슷해보이는 이미지 $G(x)$를 생성하려 하며, 반면 $D_Y$는 변환된 샘플 $G(x)$와 실제 샘플 $y$를 구별하려고 한다. $G$는 이 목적 함수를 최소화하려고 하며 경쟁자 $D$는 이를 최대화하려고 한다. 다시 말해, $$\min_G\max_{D_Y}\mathcal{L}_\text{GAN}(G,D_X,X,Y)$$이다. 우리는 매핑 함수 $F:Y\rightarrow X$와 그 판별자 $D_X$에도 또한 비슷한 adversarial loss를 도입한다. 즉 $$\min_F\max_{D_X}\mathcal{L}_\text{GAN}(F,D_X,Y,X)$$이다.

## 3.2 Cycle Consistency  Loss

경쟁적 훈련은 이론적으로 매핑 $G$와 $F$가 각각 타깃 영역 $Y$와 $X$에 대하여 동일한 분포를 이루는 출력을 만들도록 학습한다 (엄격하게 말해서, 이를 위해 $G$와 $F$는 확률 함수여야 한다). 하지만, 용량이 충분히 클 경우 신경망은 똑같은 입력 이미지 집합에서 타깃 영역의 임의의 무작위적인 이미지 순열로 매핑할 수 있으며 이 때 학습된 매핑은 출력 분포를 타깃 분포와 일치하도록 유도할 수 있다. 따라서 adversarial loss 단독으로는 학습된 함수가 각각의 입력 $x_i$를 의도된 출력 $y_i$으로 매핑시킬 것임을 보증할 수 없다. 가능한 매핑 함수의 공간을 더욱 축소시키기 위해서, 우리는 학습된 매핑 함수들이 cycle-consistent해야 한다고 주장한다. Figure 3 (b)에서 보여지듯이, 영역 $X$의 각 이미지 $x$에 대해 이미지 변환 cycle은 $x$를 원래 이미지로 되돌릴 수 있어야 한다. 즉 $x\rightarrow G(x)\rightarrow F(G(x))\approx x$을 만족해야 한다. 이를 *forward cycle consistency*로 부르겠다. 비슷하게 Figure 3 (c)에서 볼 수 있듯이, 영역 $Y$의 각 이미지 $y$에 대해, $G$와 $F$는 또한 *backward cycle consistency*, 즉 $y\rightarrow F(y)\rightarrow G(F(y))\approx y$를 만족해야 한다. 우리는 이 행위에 *cycle consistency loss*를 사용하여 인센티브를 부여한다:

$$\begin{align}
\mathcal{L}_\text{cyc}(G,F) &= \mathbb{E}_{x\sim p_\text{data}(x)}[||F(G(x))-x||_1] \\
&+ \mathbb{E}_{y\sim p_\text{data}(y)}[||G(F(y))-y||_1]
\end{align}$$

예비 실험에서 우리는 또한 L1 노름을 $F(G(x))$와 $x$ 사이, 그리고 $G(F(y))$와 $y$ 사이의 adversarial loss로 대체해보았지만, 향상된 성능을 확인하지 못했다.

Cycle consistency loss를 통해 유도된 결과는 Figure 4에서 확인할 수 있다. 재구성된 이미지 $F(G(x))$는 입력 이미지 $x$에 밀접하게 일치하게 된다.

![Figure 4: 다양한 실험에서의 입력 이미지 $x$와 출력 이미지 $G(x)$, 그리고 재구성된 이미지 $F(G(x))$. 위에서부터 photo↔Cezanne, horses↔zebras, winter→summer Yosemite, aerial photos↔Google maps.](/assets/images/2021-12-05/Untitled%203.png)

Figure 4: 다양한 실험에서의 입력 이미지 $x$와 출력 이미지 $G(x)$, 그리고 재구성된 이미지 $F(G(x))$. 위에서부터 photo↔Cezanne, horses↔zebras, winter→summer Yosemite, aerial photos↔Google maps.

## 3.3 Full Objective

우리의 전체 목적 함수는 다음과 같으며, $\lambda$는 두 목적 함수의 상대적인 중요성을 조절한다:

$$\begin{align}
\mathcal{L}(G,F,D_X,D_Y)
&= \mathcal{L}_\text{GAN}(G,D_Y,X,Y) \\
&+ \mathcal{L}_\text{GAN}(F,D_X,Y,X) \\
&+ \lambda\mathcal{L}_\text{cyc}(G,F).
\end{align}$$

우리의 목표는 다음을 해결하는 것이다:

$$\begin{equation}
G^*,F^*=\arg\min_{G,F}\max_{D_X,D_Y}\mathcal{L}(G,F,D_X,D_Y).
\end{equation}$$

우리의 모델은 두 개의 "오토인코더"를 훈련시키는 것으로 볼 수 있다. 하나의 오토인코더 $F\circ G:X\rightarrow X$와 다른 오토인코더 $G\circ F:Y\rightarrow Y$를 공동으로 학습한다. 하지만 이러한 오토인코더들은 각각 특별한 내부 구조를 가지는데 이미지를 다른 영역으로 변환하는 중간 표현을 통해 이미지를 그 자체로 매핑하는 방식이다. 이러한 설정은 adversarial loss를 사용하여 임의의 타깃 분포와 일치하도록 오토인코더의 병목 레이어를 훈련시키는, 이른바 "적대적 오토인코더"의 특별한 경우로 볼 수 있다. 우리의 경우에, $X\rightarrow X$ 오토인코더의 타깃 분포는 영역 $Y$의 분포이다.

Section 5.1.4에서 우리는 전체 목적 함수에서 adversarial loss $$\mathcal{L}_\text{GAN}$$과 cycle consistency loss $$\mathcal{L}_\text{cyc}$$ 항을 제거해보며 우리의 방법을 비교한다. 그리고 경험적으로 두 목적 함수가 고품질의 졀과에 도달하기 위해 결정적인 역할을 수행함을 보여준다. 우리는 또한 단방향의 cycle loss으로 우리의 방법을 평가하고 단방향 cycle만으로는 이러한 제약이 적은 문제에 대한 훈련을 일반화하기에 충분하지 않다는 것을 보여준다.

# 4. Implementation

## Network Architecture

우리는 neural style transfer와 superresolution에서 뛰어난 결과를 보여준 Johnson et al.의 논문을 우리의 GAN에 적용했다. 이 신경망은 3개의 convolution, 몇 개의 residual block, 2개의 stride $1/2$인 fractionally-strided convolution, 그리고 특성을 RGB로 매핑하는 하나의 convolution을 포함한다. 우리는 128 × 128 이미지에대해 6개 블록을 사용하고 256 × 256 이상의 해상도 훈련 이미지에 대해서는 9개 블록을 사용한다. Johnson et al.과 비슷하게, 우리는 instance normalization을 사용한다. 판별자 신경망에는  70 × 70의 겹쳐 지나가는 이미지 패치의 참 거짓 분류에 집중하는 70 × 70 PatchGAN을 사용한다. 이러한 패치 수준의 판별자 구조는 전체 이미지를 다루는 판별자에 비해 더 적은 수의 파라미터를 포함하며 임의의 크기의 이미지에 대해서도 fully convolution을 수행하여 작업할 수 있다.

## Training details

우리는 모델 훈련 절차를 안정화시키기 위해 최근 연구에서 얻은 두 가지 기술을 적용한다. 첫째로, $$\mathcal{L}_\text{GAN}$$ (Equation 1)에 대해, 음의 로그 가능도 목적 함수를 최소제곱 손실로 대체한다. 이 손실은 훈련 동안에 더 안정적이며 더 좋은 품질의 결과를 만들어낸다. 특히, GAN 손실 $$\mathcal{L}_\text{GAN}(G,D,X,Y)$$에서, 우리는 $G$가 $$\mathbb{E}_{x\sim p_\text{data}(x)}[(D(G(x))-1)^2]$$을 최소화하고 $D$가 $$\mathbb{E}_{y\sim p_\text{data}(y)}[(D(y)-1)^2]+\mathbb{E}_{x\sim p_\text{data}(x)}[D(G(x))^2]$$을 최소화하도록 훈련시킨다.

둘째로, 모델의 진동을 줄이기 위해, 우리는 Shrivastava et al.의 전략을 따르고 최신의 생성자가 생성한 것이 아닌 이전의 이미지들의 히스토리를 사용하여 판별자를  업데이트한다. 이미지 버퍼가 이전에 생성된 50개의 이미지들을 저장하도록 한다.

모든 실험에서, 우리는 Equation 3의 $\lambda$를 10으로 설정했다. 우리는 배치 크기가 1인 Adam 최적화기를 사용한다. 모든 신경망은 처음부터 훈련되었으며 학습률 0.0002로 훈련되었다. 학습률은 첫 100 에포크동안 유지되며 다음 100 에포크동안 선형적으로 0으로 감소한다. 부록 (Section 7)에 데이터 세트와 아키텍쳐, 그리고 훈련 절차에 대한 세부 사항이 나와 있다.

# 5. Results

우리는 먼저 평가에 입출력 쌍을 사용할 수 있는 짝지어진 데이터세트를 사용하여 최근의 unpaired image-to-image translation에 대한 방법들과 우리의 접근법을 비교한다. 그 이후에는 adversarial loss와 cycle consistency loss의 중요성을 연구하고 전체적인 방법과 일부 손실을 제거한 몇 가지 변종을 비교한다. 마지막으로, 짝지어진 데이터가 없는 넓은 범위의 응용에서 우리의 알고리즘의 일반성을 입증한다. 간결하게, 우리의 방법을 `CycleGAN`이라고 부른다. [PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)와 [Torch](https://github.com/junyanz/CycleGAN) 코드, 모델, 그리고 전체 결과는 우리의 [웹사이트](https://junyanz.github.io/CycleGAN/)에서 확인할 수 있다.

## 5.1 Evaluation

"pix2pix"와 같은 평가 데이터 세트와 기준을 사용하여 질적으로 그리고 양적으로 몇 가지 베이스라인에 우리의 방법을 비교한다. 비교 작업은 Cityscapes 데이터 세트의 semantic labels↔photo, 그리고 Google Maps에서 스크래핑한 데이터의 map↔aerial photo를 포함한다. 또한 전체 손실 함수에 대해 ablation study를 수행한다.

![Figure 5: Cityspaces 이미지들에서 훈련된 labels↔photos 매핑에 대한 다른 방법들.](/assets/images/2021-12-05/Untitled%204.png)

Figure 5: Cityspaces 이미지들에서 훈련된 labels↔photos 매핑에 대한 다른 방법들.

![Figure 6: Google Maps photo↔maps 매핑에 대한 다른 방법들.](/assets/images/2021-12-05/Untitled%205.png)

Figure 6: Google Maps photo↔maps 매핑에 대한 다른 방법들.

![Table 1: 256 × 256 해상도의 maps↔aerial photos AMT "실제 대 거짓" 테스트.](/assets/images/2021-12-05/Untitled%206.png)

Table 1: 256 × 256 해상도의 maps↔aerial photos AMT "실제 대 거짓" 테스트.

![Table 2: Cityscapes labels→photo에서 측정된 서로 다른 방법들에 대한 FCN 점수.](/assets/images/2021-12-05/Untitled%207.png)

Table 2: Cityscapes labels→photo에서 측정된 서로 다른 방법들에 대한 FCN 점수.

![Table 3: Cityscapes photo→labels에서 측정된 서로다른 방법들에 대한 분류 성능.](/assets/images/2021-12-05/Untitled%208.png)

Table 3: Cityscapes photo→labels에서 측정된 서로다른 방법들에 대한 분류 성능.

![Table 4: Ablation study: Cityscapes labels→photo에서 측정된 우리의 방법의 다른 변종들에 대한 FCN 점수.](/assets/images/2021-12-05/Untitled%209.png)

Table 4: Ablation study: Cityscapes labels→photo에서 측정된 우리의 방법의 다른 변종들에 대한 FCN 점수.

![Table 5: Ablation study: Cityscapes photo→labels에서 측정된 서로 다른 손실에 대한 분류 성능.](/assets/images/2021-12-05/Untitled%2010.png)

Table 5: Ablation study: Cityscapes photo→labels에서 측정된 서로 다른 손실에 대한 분류 성능.

### 5.1.1 Evaluation Metrics

- **AMT perceptual studies** Map↔aerial photo 작업에서, 우리의 출력물들의 현실성을 평가하기 위해 Amazon Mechanical Turk (AMT)에서 "실제 대 거짓" 인지 연구를 진행한다. 기본적으로 Isola et al. [22]의 인지 연구 프로토콜을 따르며, 테스트한 알고리즘별로 25명의 참가자의 데이터만 모으는 것이 다르다. 참가자들에게 일련의 이미지 쌍을 보여주었으며, 이 이미지쌍은 하나는 실제 사진이나 지도이고 다른 하나는 (우리의 알고리즘이나 베이스라인이 생성한) 가짜 이미지이다. 그리고 참가자들이 생각하기에 실제일 것같은 이미지를 클릭하도록 했다. 각 세션의 첫 10번의 시도는 연습이고 참가자 반응의 정답 여부를 피드백으로 보여주었다. 남은 40번의 시도가 각 알고리즘이 참가자를 속인 비율을 평가하는데 사용되었다. 각 세션은 오직 하나의 알고리즘을 테스트했으며, 참가자들은 오직 하나의 세션만 참여할 수 있도록 했다. 우리가 이 논문에 작성한 결괏값들은 [22]의 것들과 직접적으로 비교할 수 없으며 이는 실측 이미지들을 조금 다르게 처리했으며 (우리는 256 × 256 이미지로 훈련시키지만 pix2pix는 512 × 512 이미지의 256 × 256 패치로 훈련되었고 테스트에 512 × 512 이미지를 컨볼루션을 적용하여 진행했다. 많은 기준 모델들이 높은 해상도의 이미지로 확대될 수 없으며 CoGAN이 컨볼루션을 적용하여 테스트될 수 없으므로 256 × 256 이미지를 선택한다.) 우리가 테스트한 참가자 집단의 분포가 [22]와 (다른 때에 실험이 진행되었으므로) 다르기 때문이다. 따라서 우리의 값들은 우리의 논문에서 동일한 조건으로 실험한 기준 모델들과 비교하는 데에만 사용되어야 한다.
- **FCN score** 인지 연구들이 그래픽 현실성을 평가하는데 좋은 표준이 될 수 있지만, 우리는 인간 대상의 실험을 필요로 하지 않는 자동적인 양적 기준을 찾는다. 이를 위해 우리는 [22]에서 "FCN 점수"를 채택하며 이를 사용하여 Cityscapes labels→photo 작업을 평가한다. FCN 기준은 생성된 사진이 널리 사용되는 시멘틱 분할 알고리즘인 fully-convolutional network (FCN) 으로 얼마나 해석될 수 있는지를 평가한다. FCN은 생성된 사진에 대해 레이블 맵을 예측한다. 그 후 이 레이블 맵은 아래에 설명된 표준 시멘틱 분할 기준(semantic segmentation metric)을 사용하여 실제 레이블과 비교된다. 이 의도는 우리가 "도로 위의 자동차" 레이블 맵에서 사진을 생성해낸다고 하면 FCN을 생성된 사진에 적용했을 때 성공적으로 "도로 위의 자동차"를 탐지해야 한다는 것이다.
- **Semantic segmentation metrics** Photo→labels 작업의 성능을 평가하기 위해 픽셀별 정확도, 클래스별 정확도, 그리고 평균 class Intersection-Over-Union (Class IOU) 을 포함하는 Cityscapes 벤치마크의 표준적인 측정 기준을 사용한다.

### 5.1.2 Baselines

- **CoGAN** 이 방법은 영역 $X$와 $Y$에 대한 GAN 생성자를 각각 학습하며 공유되는 잠재 표현을 위해 초반 일부 레이어들에 공통된 가중치를 사용한다. $X$에서 $Y$로의 변환은 이미지 $X$를 생성하는 잠재 표현을 찾고 이 표현을 스타일 $Y$로 렌더링함으로 이루어진다.
- **SimGAN** 우리의 방법처럼, Shrivastava et al.은 $X$에서 $Y$로의 변환을 훈련시키기 위해 adversarial loss를 사용한다. 픽셀 수준에서 큰 변화를 불러일으키는 것에 패널티를 주기 위해 규제 항 $\|\|x-G(x)\|\|_1$이 사용되었다.
- **Feature loss + GAN** 우리는 또한 RGB 픽셀값이 아니라, 사전훈련된 신경망(VGG-16 `relu4_2`)을 사용한 deep image features로 L1 손실이 계산되는 SimGAN의 변종을 테스트했다. 이처럼 deep feature space에서 거리가 계산되는 것은 때로 "perceptual loss"를 사용하기도 한다.
- **BiGAN/ALI** Unconditional GAN은 랜덤 노이즈 $z$에서 이미지 $x$로 매핑하는 생성자 $G : Z \rightarrow X$를 학습한다. BiGAN과 ALI는 역 매핑 함수 $F : X \rightarrow Z$도 학습하는 것을 제안한다. 원래는 잠재 벡터 $z$를 이미지 $x$로 매핑하도록 설계되었지만, 우리는 이 목적 함수를 원 이미지 $x$에서 타깃 이미지 $y$로 매핑하도록 구현했다.
- **pix2pix** 우리는 또한 짝지어진 데이터에서 훈련된 pix2pix와도 비교해, 짝지어진 데이터를 사용하지 않고 이 "상한선"에 얼마나 가깝게 도달할 수 있는지를 확인한다.

공정한 비교를 위해, 우리는 CoGAN을 제외한 모든 기준 모델들을 우리의 방법과 같은 아키덱처와 세부사항을 사용해 구현한다. CoGAN은 공유된 잠재 표현에서 이미지를 만들어내는 생성자를 만들어내며, 따라서 우리의 이미지-이미지 신경망과 호환되지 않는다. 대신 공개된 CoGAN [구현](https://github.com/mingyuliutw/CoGAN)을 사용한다.

### 5.1.3 Comparison against baseline

Figure 5와 6에서 볼 수 있듯이, 그동안 우리는 어떠한 베이스라인으로도 설득력있는 결과를 만들어낼 수 없었다. 반면에 우리의 방법은 fully supervised pix2pix와 자주 비슷한 수준의 변환을 만들어낼 수 있다.

Table 1은 AMT 현실성 인지 작업에 따른 성능을 보여준다. 여기서, 256 × 256 해상도의 maps→aerial photos와 aerial photos→maps 모든 방향에서 우리의 방법이 참가자 전체 시도의 1/4를 속일 수 있다. 모든 베이스라인은 참가자들을 거의 속일 수 없었다.

Table 2은 Cityscapes에서 labels→photo 작업에 대한 성능을 평가하고 Table 3은  역 매핑(photos→labels)에 대해 측정한다. 두 경우에 대해 우리의 방법은 마찬가지로 베이스라인을 능가하는 모습을 보여준다.

### 5.1.4 Analysis of the loss function

Table 4와 5에서, 우리는 전체 손실에 대한 ablation들을 비교했다. GAN 손실을 없애는 것은 결과를 상당히 저하시키며 cycle-consistency 손실을 줄이는 것도 마찬가지이다. 따라서 두 항들은 우리의 결과에 상당히 중요하다고 결론을 내린다. 우리는 또한 단방향에서만의 cycle 손실(GAN + forward cycle 손실인 $$\mathbb{E}_{x\sim p_\text{data}(x)}[\|F(G(x))-x\|_1]$$ 또는 GAN + backward cycle 손실인 $$\mathbb{E}_{y\sim p_\text{data}(y)}[\|G(F(y))-y\|_1]$$을 사용하여 우리의 방법에 대해 평가했으며 (Equation 2) 이를 통해 단뱡향만의 손실은 자주 훈련 불안정성과 모드 붕괴를 유발한다는 것을 찾아냈다. Figure 7은 몇 가지 질적인 예시를 보여준다.

![Figure 7: Cityscapes labels↔photos 매핑에 대해 훈련시킨 우리의 방법의 다른 변종들. *Cycle alone* (cycle-consistency loss)만 훈련시킨 결과와 *GAN + backward*만 훈련시킨 결과는 타깃 영역과 비슷한 이미지를 생성하는데 실패한다. *GAN alone*과 *GAN + forward*는 모드 붕괴를 겪으며, 입력 사진과 관련없이 동일한 레이블 맵을 생성한다.](/assets/images/2021-12-05/Untitled%2011.png)

Figure 7: Cityscapes labels↔photos 매핑에 대해 훈련시킨 우리의 방법의 다른 변종들. *Cycle alone* (cycle-consistency loss)만 훈련시킨 결과와 *GAN + backward*만 훈련시킨 결과는 타깃 영역과 비슷한 이미지를 생성하는데 실패한다. *GAN alone*과 *GAN + forward*는 모드 붕괴를 겪으며, 입력 사진과 관련없이 동일한 레이블 맵을 생성한다.

### 5.1.5 Image reconstruction quality

Figure 4에서 재구성된 이미지 $F(G(x))$의 몇 가지 무작위 샘플을 볼 수 있다. 우리는 훈련 및 테스트 시간동안, 재구성된 이미지들이 자주 원래의 이미지 $x$에 가까우며 이는 심지어 map↔aerial photos 매핑과 같이한 쪽 영역이 다른 쪽에 비해 상당히 다양한 정보를 표현하는 경우에도 마찬가지임을 관찰했다.

### 5.1.6 Additional results on paired datasets

![Figure 8: Architectural labels↔photos와 edges↔shoes와 같이 "pix2pix"에서 쓰인 paired datasets에서 훈련된 CycleGAN 예시.](/assets/images/2021-12-05/Untitled%2012.png)

Figure 8: Architectural labels↔photos와 edges↔shoes와 같이 "pix2pix"에서 쓰인 paired datasets에서 훈련된 CycleGAN 예시.

Figure 8은 CMP Facade Database의 architectural labels↔photos와 UT Zappos50K dataset의 edges↔shoes와 같이 "pix2pix"에서 쓰인 paired datasets에 대한 몇 가지 결과의 예시를 보여준다. 우리의 방법은 paired supervision 없이 매핑을 학습함에도 결과물 이미지의 품질은 fully supervised pix2pix의 결과물과 비슷하다.

## 5.2 Applications

우리는 paired training data가 존재하지 않는 몇 가지 응용에 대해 우리의 방법을 입증했다. 데이터 세트에 대한 세부 사항은 appendix (Section 7)을 참고하라. 훈련 데이터에서의 변환이 자주 테스트 데이터에서의 변환보다 더 좋아보였으며, 훈련과 테스트 데이터에서 응용의 전체 결과는 우리의 프로젝트 [웹사이트](https://junyanz.github.io/CycleGAN/)에서 확인할 수 있다.

### Collection style transfer (Figure 10 and Figure 11)

![Figure 10: Style transfer 컬렉션 I: 입력 이미지들을 Monet, Van Gogh, Cezanne, Ukiyo-e의 예술적 스타일로 변환했다.](/assets/images/2021-12-05/Untitled%2013.png)

Figure 10: Style transfer 컬렉션 I: 입력 이미지들을 Monet, Van Gogh, Cezanne, Ukiyo-e의 예술적 스타일로 변환했다.

![Figure 11: Style transfer 컬렉션 II: 입력 이미지들을 Monet, Van Gogh, Cezanne, Ukiyo-e의 예술적 스타일로 변환했다.](/assets/images/2021-12-05/Untitled%2014.png)

Figure 11: Style transfer 컬렉션 II: 입력 이미지들을 Monet, Van Gogh, Cezanne, Ukiyo-e의 예술적 스타일로 변환했다.

우리는 Flickr와 WikiArt에서 다운로드한 도시 경관 사진들에 모델을 훈련시켰다. "Neural style transfer"의 최근 성과, 즉 하나의 선택된 예술 작품의 스타일로 변환하는 것과는 다르게, 우리의 방법은 예술 작품 전체 *모음집*의 스타일을 흉내낸다. 따라서 우리는 예를 들어 단지 Starry Night의 스타일만이 아닌, Van Gogh의 스타일 자체의 이미지를 생성해내도록 학습할 수 있다. 각 예술가/스타일에 대한 데이터 세트의 크기는 Cezanne, Monet, Van Gogh, 그리고 Ukiyo-e에 대해 526, 1073, 400, 563이다.

### Object transfiguration (Figure 13)

![Figure 13: 우리의 방법을 몇 가지 변환 문제에 적용시킨 결과. 이 이미지들은 상대적으로 성공적인 결과로 선택되었다. 맨 위 두 행에서는 ImageNet의 *wild horse* 클래스의 939개 이미지와 *zebra* 클래스의 1177개 이미지로 훈련시킨 말과 얼룩말 사이 객체 변형의 결과를 보여준다. Horse→zebra 데모 [비디오](https://youtu.be/9reHvktowLY)도 있다. 가운데 두 행은 Flickr에서 얻은 Yosemite의 겨울과 여름 사진으로 훈련된 계절 변환의 결과물이다. 마지막 두 행에서는 ImageNet의 996개 *apple* 이미지와 1020개 *navel orange* 이미지들로 모델을 훈련시켰다.](/assets/images/2021-12-05/Untitled%2015.png)

Figure 13: 우리의 방법을 몇 가지 변환 문제에 적용시킨 결과. 이 이미지들은 상대적으로 성공적인 결과로 선택되었다. 맨 위 두 행에서는 ImageNet의 *wild horse* 클래스의 939개 이미지와 *zebra* 클래스의 1177개 이미지로 훈련시킨 말과 얼룩말 사이 객체 변형의 결과를 보여준다. Horse→zebra 데모 [비디오](https://youtu.be/9reHvktowLY)도 있다. 가운데 두 행은 Flickr에서 얻은 Yosemite의 겨울과 여름 사진으로 훈련된 계절 변환의 결과물이다. 마지막 두 행에서는 ImageNet의 996개 *apple* 이미지와 1020개 *navel orange* 이미지들로 모델을 훈련시켰다.

모델은 ImageNet의 하나의 객체 클래스를 다른 클래스로 변환하도록 훈련되었다 (각 클래스는 거의 1000개의 훈련 이미지를 포함한다). Turmukhambetov et al.은 하나의 객체를 같은 범주에 속하는 다른 객체로 변환하는 subspace 모델을 제안하지만, 반면에 우리의 방법은 두 가지 시각적으로 비슷한 범주 사이 객체 변환에 초점을 맞춘다.

### Season transfer (Figure 13)

모델은 Flickr에서 다운로드한 Yosemite의 854개 겨울 사진과 1273개 여름 이미지에 훈련되었다.

### Photo generation from paintings (Figure 12)

![Figure 9: Monet's painting→photos에서 *identity mapping loss*의 효과. Identity mapping loss은 입력 그림의 색을 보존하도록 돕는다.](/assets/images/2021-12-05/Untitled%2016.png)

Figure 9: Monet's painting→photos에서 *identity mapping loss*의 효과. Identity mapping loss은 입력 그림의 색을 보존하도록 돕는다.

![Figure 12: Monet의 그림에서 사진 스타일로의 매핑의 상대적으로 성공적인 결과들.](/assets/images/2021-12-05/Untitled%2017.png)

Figure 12: Monet의 그림에서 사진 스타일로의 매핑의 상대적으로 성공적인 결과들.

Painting→photo에 대해, 우리는 어떠한 추가적인 손실을 도입하는 것이 매핑이 입출력간의 색 구성을 보존하도록 도움을 준다는 것을 알아냈다. 특히, 우리는 Taigman et al.의 기술을 채택하고 타깃 영역의 실제 샘플들이 생성자의 입력으로 주어질 때 생성자가 identity mapping에 근접하도록 규제한다. 즉, $\mathcal{L}_\text{identity}(G,F)=
\mathbb{E}_{y\sim p_\text{data}(y)}[||G(y)-y||_1]+
\mathbb{E}_{x\sim p_\text{data}(x)}[||F(x)-x||_1]$이다.

$\mathcal{L}_\text{identity}$가 없으면 생성자 $G$와 $F$는 그럴 필요가 없음에도 입력 이미지의 색조를 자유롭게 바꾼다. 예를 들어, Monet의 그림들과 Flickr 사진들 사이의 매핑을 학습시킬 때, 생성자는 자주 낮의 그림을 석양에 찍힌 사진으로 매핑하는데 이는 이러한 매핑이 adversarial loss와 cycle consistency loss 아래에서 똑같이 유효하기 때문이다. 이 *identity mapping loss*의 효과는 Figure 9에서 보여주고 있다.

Figure 12에서는 Monet의 그림을 사진으로 변환하는 추가적인 결과를 보여주고 있다. 이 figure와 Figure 9는 *훈련 세트*에 포함된 그림에 대한 변환 결과를 보여준다. 반면에, 이 논문에서 모든 다른 실험들은 오직 테스트 세트에서 측정하며 그 결과를 보여준다. 훈련 세트가 paired data를 포함하지 않기 때문에, 훈련 세트 그림에 대한 그럴듯한 변환을 만드는 것은 중요하지 않은 작업이다. 실제로, Monet이 더 이상 새로운 그림, 보이지 않는 "테스트 세트"로 일반화를 만들어 낼 수 없으므로, 그림은 시급한 문제가 되지 않는다.

### Photo enhancement (Figure 14)

![Figure 14: 사진 향상: 일련의 스마트폰 스냅샷들에서 전문적인 DSLR 사진으로의 매핑. 시스템은 자주 좁은 초점을 생성해낸다. 여기서 보여주는 이미지는 테스트 세트의 가장 성공적인 결과이며 평균적인 성능은 상당히 좋지 못하다.](/assets/images/2021-12-05/Untitled%2018.png)

Figure 14: 사진 향상: 일련의 스마트폰 스냅샷들에서 전문적인 DSLR 사진으로의 매핑. 시스템은 자주 좁은 초점을 생성해낸다. 여기서 보여주는 이미지는 테스트 세트의 가장 성공적인 결과이며 평균적인 성능은 상당히 좋지 못하다.

우리의 방법이 더 얕은 피사계 심도( depth of field, DoF)를 가진 사진을 생성하는데 사용될 수 있다. 우리는 Flickr에서 다운로드한 꽃 사진에 모델을 훈련시켰다. 원 영역은 스마트폰으로 찍어, 작은 조리개로 인한 깊은 DoF를 가지는 꽃사진들을 포함한다. 타깃 영역은 큰 조리개를 가지는 DSLR로 찍힌 사진을 포함한다. 우리의 모델은 더 얕은 피사계 심도를 가진 사진을 스마트폰으로 찍은 사진으로부터 성공적으로 생성해내었다.

### Comparison with Gatys et al. [13]

![Figure 15: 우리의 방법을 이미지 스타일화에서 neural style transfer와 비교했다. ](/assets/images/2021-12-05/Untitled%2019.png)

Figure 15: 우리의 방법을 이미지 스타일화에서 neural style transfer와 비교했다. 

![Figure 16: 우리의 방법을 다양한 응용에서 neural style transfer와 비교했다.](/assets/images/2021-12-05/Untitled%2020.png)

Figure 16: 우리의 방법을 다양한 응용에서 neural style transfer와 비교했다.

Figure 15에서, 우리는 사진 스타일화에 대해 우리의 결과와 neural style transfer [13]를 비교했다. 각 행에서 우리는 먼저 두 대표적인 예술작품을 [13]의 스타일 이미지로 사용했다. 한편 우리의 방법은 전체 *컬렉션*의 스타일에서 사진을 생성할 수 있다. 전체 컬렉션에 대한 neural style transfer와 비교하기 위해서, 우리는 타깃 영역에 대한 평균 Gram 행렬을 계산했고 이 행렬을 Gatys et al. [13]의 "평균 스타일"로 변환하는데 사용했다.

Figure 16은 다른 변환 작업에 대한 비슷한 비교를 보여준다. 우리는 Gatys et al. [13]가 원하는 출력과 밀접하게 일치하는 대상 스타일 이미지들을 찾아야 하지만 자주 사실적인 결과물을 만들어내지 못하는 반면, 우리의 방법은 타깃 영역과 비슷한, 자연스럽게 보이는 결과물들을 성공적으로 만들어냄을 관찰할 수 있었다.

# 6. Limitations and Discussion

![Figure 17: 우리의 방법의 일반적인 실패 케이스들. 좌측: dog→cat 변형에서, CycleGAN은 입력에 대한 작은 변화만을 만들 수 있다. 우측: CycleGAN은 또한 horse→zebra 예시에서 실패하는데, 이는 우리의 모델이 훈련 동안 승마를 하는 이미지를 학습하지 못했기 때문이다.](/assets/images/2021-12-05/Untitled%2021.png)

Figure 17: 우리의 방법의 일반적인 실패 케이스들. 좌측: dog→cat 변형에서, CycleGAN은 입력에 대한 작은 변화만을 만들 수 있다. 우측: CycleGAN은 또한 horse→zebra 예시에서 실패하는데, 이는 우리의 모델이 훈련 동안 승마를 하는 이미지를 학습하지 못했기 때문이다.

우리의 방법이 많은 케이스에 대해 설득력 있는 결과물들을 만들어낼 수 있지만, 항상 긍정적인 것과는 거리가 멀다. Figure 17은 몇 가지 일반적인 실패 케이스들을 보여준다. 색상과 질감에 대한 변환 작업에서는 위의 많은 부분에서 보이듯이 우리의 방법은 종종 성공적인 결과를 만들어낸다. 우리는 또한 기하학적인 변화를 필요로 하는 작업 또한 탐색했지만 거의 성공적이지 못했다. 예를 들어 dog→cat 변형 작업에서, 학습된 변환은 입력에 대한 최소한의 변화로 퇴화된다. 이러한 실패는 아마도 생성자 아키텍처가 외관의 변화에 좋은 성능을 내도록 만들어졌기 때문이다. 더 다양하고 극단적인 변환, 특히 기하학적 변환을 다루는 것은 앞으로의 중요한 연구 문제이다.

몇 가지 실패 케이스들은 훈련 데이터 세트의 분포 특성의 변화에 의해 만들어진다. 예를 들어 우리의 방법은 horse→zebra 예제(Figure 17, 우측)에서 혼동을 보였는데, 이는 우리의 모델이 ImageNet의 *wild horse*와 *zebra* 집합에서 훈련되었는데 이 집합이 말을 탄 사람을 포함하지 않기 때문이다.

우리는 또한 paired training data로 도달할 수 있는 결과와 unpaired method로 도달할 수 있는 결과 사이에 남아 있는 차이를 탐색했다. 몇 가지 케이스에서 이 차이는 줄어들기가 불가능할 정도로 매우 어렵다. 예를 들어 우리의 방법은 때로 photo→labels 작업의 출력에서 나무와 건물의 레이블을 뒤바꾼다. 이러한 모호함을 해결하기 위해서는 weak semantic supervision이 필요할 것이다. Weak supervised data나 semi-supervised data를 통합하는 것은 fully-supervised system의 어노테이션 비용의 매우 일부에 불과함에도 상당히 강력한 변환기를 만들어낼 수 있다.

그럼에도 불구하고, 많은 케이스에서 완전한 unpaired data는 풍부하며 사용되어야 한다. 이 논문은 이러한 "비지도" 설정에서 가능한 것의 경계를 넓혔다.

# 7. Appendix

## 7.1. Training details

우리는 모델을 처음부터 훈련시켰으며, 학습률은 0.0002이다. 실제로, $G$의 비율에 대해 $D$가 학습하는 비율을 늦추기 위해 $D$를 최적화하는 동안 목적 함수를 2로 나누었다. 첫 100 에포크동안 동일한 학습률을 적용했으며, 다음 100 에포크동안 선형적으로 학습률을 0으로 줄였다. 가중치는 가우시안 분포 $\mathcal{N}(0, 0.02)$로 초기화되었다.

### Cityscapes label↔photo

Cityscapes 훈련 세트에서 128 × 128 크기의 이미지 2975개로 훈련시켰다. Cityscapes 검증 세트로 테스트를 진행했다.

### Maps↔aerial photograph

Google Maps에서 스크래핑한 256 × 256 크기의 이미지 1096개로 훈련시켰다. 이미지들은 New York City의 내부와 주변에서 샘플링되었다. 그 후 데이터를 샘플 영역의 중간 위도에 대해서 훈련과 테스트 세트로 나누었다 (테스트 세트에서 훈련 데이터의 픽셀이 나타나지 않도록 버퍼 영역이 추가되었다).

### Architectural facades labels↔photo

CMP Facade Database로 부터 400개의 훈련 이미지가 사용되었다.

### Edges↔shoes

UT Zappos50K 데이터 세트로부터 약 50,000개의 훈련 이미지가 사용되었다. 모델은 5 에포크동안 훈련되었다.

### Horse↔Zebra and Apple↔Orange

ImageNet에서 *wild horse*, *zebra*, *apple*, 그리고 *navel orange*라는 키워드를 검색해 이미지를 다운로드했다. 이미지들은 256 × 256 크기로 스케일링되었다. 각 클래스의 훈련 세트 크기는 939 (horse), 1177 (zebra), 996 (apple), 그리고 1020 (orange)이다.

### Summer↔Winter Yosemite

Flickr API를 이용하여 *yosemite* 태그와 *date-taken* 필드로 이미지를 다운로드했다. 흑백 사진은 사용하지 않았다. 이미지들은 256 × 256 크기로 스케일링되었다. 각 클래스의 훈련 세트 크기는 1273 (summer) 그리고 854 (winter)이다.

### Photo↔Art for style transfer

Wikiart.org에서 예술 작품 이미지들을 다운로드했다. 몇 가지 스케치나 음란한 예술 작품들은 사용하지 않았다. 사진들은 Flickr에서 *landscape*와 *landscapephotography* 태그로 다운로드했다. 흑백 사진은 사용하지 않았다. 이미지들은 256 × 256 크기로 스케일링되었다. 각 클래스의 훈련 세트 크기는 1074 (Monet), 584 (Cezanne), 401 (Van Gogh), 1433 (Ukiyo-e), 그리고 6853 (Photographs)이다. Monet 데이터 세트는 오직 풍경 사진만을 포함하기 위해 불필요한 것을 제거했으며, Van Gogh 데이터 세트에서는 그의 가장 유명한 예술적 스타일을 대표하는 후반부 작품들만을 포함했다.

### Monet's paintings→photos

메모리를 절약하면서 고해상도의 이미지를 만들기 위해, 훈련에서 이미지에 대해 random square crop을 사용했다. 결과를 생성하기 위해, 종횡비를 일정하게 하여 너비를 512 픽셀로 변환한 이미지들을 생성자 네트워크에 입력으로 사용했다. Identiry mapping loss의 가중치는 cycle consistency loss의 가중치 $\lambda$의 절반 $0.5\lambda$로 두었다. $\lambda = 10$으로 설정했다.

### Flower photo enhancement

스마트폰으로 찍은 꽃 사진은 Flickr에서 *Apple iPhone 5*, *5s*, 또는 *6*으로 찍은 사진에 텍스트 *flower*를 검색해 다운로드했다. 얕은 DoF를 가진 DSLR 이미지들 또한 Flickr에서 검색 태그 *flower*, *dof*로 다운로드했다. 이미지들은 너비 360 픽셀로 스케일링되었다. 가중치 $0.5\lambda$의 identity mapping loss가 사용되었다. 스마트폰과 DSLR 데이터 세트의 훈련 세트 크기는 각각 1813과 3326이다. $\lambda = 10$으로 설정했다.

## 7.2. Network architectures

[PyTorch](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)와 [Torch](https://github.com/junyanz/CycleGAN) 구현 모두 준비했다.

### Generator architectures

우리는 Johnson et al. [23]에서 아키텍처를 채택했다. 128 × 128 훈련 이미지에 대해 6개의 residual block을 사용했고, 256 × 256 또는 더 고해상도의 훈련 이미지에 대해 9개의 residual block을 사용했다. 아래에서는 Johnson et al.의 Github [레포지토리](https://github.com/jcjohnson/fast-neural-style)에서 사용되는 명명 규칙을 따른다.

`c7s1-k`는 $k$개 필터와 stride 1의 7 × 7 Convolution-InstanceNorm-ReLU 레이어를 의미한다. `dk`는 $k$개 필터와 stride 2의 3 × 3 Convolution-InstanceNorm-ReLU 레이어를 의미한다. 아티팩트들을 줄이기 위해 Reflection padding이 사용되었다. `Rk`는 두 개의 3 × 3 convolutional 레이어이며 두 레이어의 필터 수는 같다. `uk`는 $k$개 필터와 stride $\dfrac{1}{2}$의 3 × 3 fractional-strided-Convolution-InstanceNorm-ReLU 레이어를 의미한다.

6개의 residual block을 포함한 신경망은 다음과 같다:

`c7s1-64`, `d128`, `d256`, `R256`, `R256`, `R256`, `R256`, `R256`, `R256`, `u128`, `u64`, `c7s1-3`

9개의 residual block을 포함한 신경망은 다음과 같다:

`c7s1-64`, `d128`, `d256`, `R256`, `R256`, `R256`, `R256`, `R256`, `R256`, `R256`, `R256`, `R256`, `u128`, `u64`, `c7s1-3`

### Discriminator architectures

판별자 신경망에는 70 × 70 PatchGAN을 사용했다. `Ck`는 $k$개 필터와 stride 2의 4 × 4 Convolution-InstanceNorm-LeakyReLU 레이어를 의미한다. 마지막 레이어 이후에, 1차원 출력을 만들기 위해 convolution을 적용했다. 처음 `C64` 레이어에는 InstanceNorm을 사용하지 않았다. 경사 0.2의 leaky ReLU를 사용했다. 판별자의 신경망은 다음과 같다: `C64-C128-C256-C512`