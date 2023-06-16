# テキストから画像への拡散モデルへの条件付制御の追加

Lvmin Zhang and Maneesh Agrawala
スタンフォード大学

我々は、プリトレーニングされた大規模拡散モデルを制御し、追加の入力条件に対応するためのニューラルネットワーク構造、**ControlNet** を提示する。ControlNet は、タスクに応じた条件を end-to-end で学習し、学習データセットが小さい場合（5万以下）でも安定して学習が可能である。また、ControlNet の学習は、拡散モデルのファインチューニングと同程度のスピードで行うことができ、モデルの学習は個人の端末で行うことが可能である。また、強力な計算クラスタがあれば、数百万から数十億といった膨大な量にデータにも対応できる。Stable Diffusion のような大規模な拡散モデルを ControlNet で補強することで、エッジマップ、セグメンテーションマップ、キーポイントなどの条件入力が可能になる。これにより、大規模な拡散モデルを制御する手法が充実し、関連して応用がさらに促進されることが期待される。
[https://github.com/lllyasviel/ControlNet]

## 1 はじめに

大きな text-to-image モデルでは、魅力的な画像を生産するために、ユーザーが短く記述的なプロンプトを入力しなければならない場合がある。テキストを入力し、画像を取得した後、私たちは当然、このプロンプトベースの制御は私たちのニーズを満たしているのかと疑問に思うだろう。例えば画像処理では、明確な問題設定を持つ多くの古くからのタスクを考慮すると、これらの大規模なモデルはこれらの特定のタスクの解決のために適用できるのだろうか。さまざまな問題、条件やユーザー操作に対応するためには、どのようなフレームワークを構築すればよいのか。特定のタスクにおいて、大規模モデルは数十億の画像から得られる利点や能力を維持することができるのであろうか。

これらの疑問に答えるため、さまざまな画像処理アプリケーションを調査し、3つの知見を得た。第一に、タスクに特化した領域で利用可能なデータ規模は、一般的な image-text 領域ほど大きくないことがある。特殊な問題（例えば、物体の形状・法線、ポーズ理解など）の最大データセットサイズは10万以下であることが多く、LAION-5B の $\frac{1}{5\times10^4}$ のサイズである。このため、特定の問題に対して大規模なモデルを学習させる場合、過学習を回避し、汎化能力を維持するための頑健なニューラルネットワークの学習方法が必須となる。

第二に、画像処理タスクがデータ駆動型ソリューションで処理される場合、大規模な計算クラスタが常に利用できるとは限らない。このため、大規模なモデルを特定のタスクに最適化するためには、許容できる時間とメモリ容量（例えば、PC 上）で、高速な学習方法が重要になる。そのためにはさらに、事前に学習させた重みの活用や、ファインチューニング戦略や転移学習が必要となる。

第三に、さまざまな画像処理問題は、問題定義、ユーザー制御、あるいは画像注釈の形態が多様である。これらの問題に対処する場合、画像拡散アルゴリズムは、例えば、ノイズ除去プロセスの制約、多頭の注意の活性化の編集など、「手続き的」な方法で規制することができるが、これらの手作りのルールの動作は、基本的に人間の指示によって規定されている。深度から画像、ポーズから人間などの特定のタスクを考慮すると、これらの問題は本質的に生の入力をオブジェクトレベルまたはシーンレベルの理解に解釈する必要があり、手作りの手続き的方法は実現性が低くなる。多くのタスクで学習された解を得るためには、end-to-end の学習が不可欠である。

![Canny エッジマップを用いた Stable Diffusion の制御](2023-05-22-14-31-44.png)
図1：**Canny エッジマップを用いた Stable Diffusion の制御**
Canny エッジマップを入力し、右の画像を生成する際には、ソース画像は使用しない。出力は、デフォルトのプロンプト「a high-quality, detailed, and professional image」で実現されている。このプロンプトは、画像の内容やオブジェクト名について何も言及しないデフォルトのプロンプトとして、本稿で使用されている。本論文の図の多くは高解像度の画像であり、拡大すると最も見やすくなる。

本論文では、大規模な画像拡散モデル（Stable Diffusion など）を制御し、タスク固有の入力条件を学習させる end-to-end のニューラルネットワーク構造である ControlNet を紹介する。ControlNet は、大規模拡散モデルの重みを「訓練されたコピー」と「ロックされたコピー」にクローン化する。ロックされたコピーは数十億枚の画像から学習したネットワーク能力を保持し、訓練可能なコピーはタスク固有のデータセットで訓練して条件制御を学習させる。訓練可能なニューラルネットワークブロックとロックされたニューラルネットワークブロックは、「ゼロコンボリューション」と呼ばれる独自のタイプのコンボリューション層で接続されており、コンボリューションの重みはゼロから最適化されたパラメータまで学習しながら徐々に大きくなっていく。生産に適した重みが保持されるため、さまざまなスケールのデータセットで頑丈な学習が可能である。ゼロコンボリューションは深層特徴に新たなノイズを加えないため、ゼロから新しいレイヤーをトレーニングするのに比べ、拡散モデルをファインチューニングするのと同じくらい速く訓練できる。

Canny エッジ、Hough ライン、ユーザーの走り書き、人間のキーポイント、セグメンテーションマップ、形状法線、深度など、様時ざまな条件のデータセットを用いて、複数の ControlNet を訓練している。また、ControlNet を小規模データセット（サンプル数5万以下、千以下）と大規模データセット（サンプル数数百万）の両方で実験した。また、深度画像のようないくつかのタスクでは、PC（Nvidia RTX 3090TI 1台）でControlNetを訓練すると、1TB のGPU メモリと数千 GPU 時間を持つ大規模計算クラスターで訓練した商用モデルと同等の結果が得られることを示した。

## 2 関連した研究

### 2.1 HyperNetwork と Neural Network の構造

HyperNetwork は、より大きなニューラルネットワークの重みに影響を与える小さなリカレントニューラルネットワークを訓練する自然言語処理[^14]に由来している。HyperNetwork の成功例は、敵対的生成ネットワーク[^1] [^10]を用いた画像生成や、他の機械学習タスク[^51]でも報告されている。これらアイデア[^15]に触発され、Stable Diffusion[^44]に小さなニューラルネットワークを取り付け、その出力画像の芸術的スタイルを変更する方法を提供した。このアプローチは、いくつかの HyperNetworks の事前訓練された重みを提供した後、より人気を博した。[^28]ControlNet と HyperNetwork は、ニューラルネットワークの動作に影響を与える方法において類似している。

ControlNet は「ゼロコンボリューション」と呼ばれる特殊なタイプの畳み込み層を使用している。初期のニューラルネットワーク研究[^31] [^47] [^32]では、ガウス分布で重みを初期化することの合理性や、重みを0で初期化することで発生しうるリスクなど、ネットワークの重みの初期化について広く議論されてきた。また、最近では、[^37]学習効果を高めるために、拡散モデルのいくつかの畳み込み層の初期重みを変化させる方法が議論されており、これはゼロコンボリューションのアイデアと共通している（両者のコードには「zero_module」という関数が含まれている）。初期の畳み込み層を操作することは、ProGan[^21]や StyleGAN[^22]、Noise2Noise[^33]や[^65]でも取り上げられている。Stability のモデルカード[^55]も、ニューラル層におけるゼロウェイトの使用について言及している。

### 2.2 拡散確率モデル

確率拡散モデルは[^52]で提案された。画像生成の成功結果は、先ず小規模で報告され[^25]、その後比較的大規模に報告された[^9]。このアーキテクチャは、Denoising Diffusion Probablistic Model（DDPM）[^17]、Denoising Diffusion Implicit Model（DDIM）[^53]、score-based diffusion[^54]等の重要な訓練やサンプリング手法によって改良されてきた。画像拡散法は、がその色を直接学習データとして用いることができ、その場合、高解像度画像を扱う際に計算能力を節約する戦略を検討したり[^53] [^50] [^26]、ピラミッドベースや多段法[^18] [^43]を直接用いたりしている研究が多い。これらの手法は、基本的にニューラルネットワークのアーキテクチャとしてU-net[^45]を使用している。拡散モデルの学習に必要な計算量を削減するために、潜像[^11]の考え方に基づき、Latent Diffusion Model（LDM）[^44]というアプローチが提案され、さらに Stable Diffusion へと拡張された。

### 2.3 Text-to-Image Diffusion

拡散モデルは、テキストから画像への生成タスクに適用することで、最先端の画像生成結果を得ることができる。これは、CLIP [^41]のような事前に訓練された言語モデルを用いて、テキスト入力を潜在ベクトルに符号化することで実現されることが多い。例えば、Glide [^38]は、画像生成と編集の両方をサポートするテキストガイド付き拡散モデルである。Disco Diffusion は、テキストプロンプトを処理するための[^9]のクリップガイド付き実装である。 Stable Diffusion は、テキストから画像への生成を実現するために、潜在拡散[^44]の大規模な実装である。Imagen [^49]は、潜像を用いず、ピラミッド構造を用いて直接ピクセルを拡散させるテキストから画像への構造である。

### 2.4 事前学習された拡散モデルのパーソナライズ、カスタマイズ、コントロール

最先端の画像拡散モデルはテキストから画像への手法が主流であるため、拡散モデルの制御を強化する最も直接的な方法は、しばしばテキスト誘導型[^38] [^24] [^2] [^3] [^23] [^43] [^16]である。この種の制御は、CLIP の特徴を操作することによっても実現できる[^43]。画像拡散処理自体は、色レベルの詳細な変化を実現するためのいくつかの機能を提供することができる[^35]（Stable Diffusion のコミュニティでは、これを img2img と呼んでいる）。画像拡散アルゴリズムは、結果をコントロールする重要な方法として、インペインティングを自然にサポートしている[^43] [^2]。Textual Inversion[^12]や DreamBooth[^46]は、同じトピックやオブジェクトを含む小さな画像セットを使って、生成結果のコンテンツをカスタマイズ（またはパーソナライズ）するために提案されている。

### 2.5 画像から画像への変換

ControlNet と画像から画像への変換は、いくつかの重複したアプリケーションを持つが、その動機は本質的に異なることを指摘したいと思う。画像から画像への変換は、異なる領域の画像間のマッピングを学習することを目的としており、ControlNet は、タスクに特化した条件で拡散モデルを制御することを目的としている。

Pix2Pix [^20]は画像間変換の概念を提示し、初期の手法は条件付き生成ニューラルネットワーク[^20] [^69] [^60] [^39] [^8] [^63] [^68]が主流であった．transformers や Vision Transformers（ViT）が普及した後、自己回帰法[^42] [^11] [^7]を用いた成功例が報告されている。また、マルチモデル法が様々な翻訳タスク[^64] [^29] [^19] [^40]からロバストな生成器を学習できることを示す研究もある。

画像間変換において現在最も有力な手法を議論する。Taming Transformer [^11]は、画像生成と画像間変換の両方を行う機能を持つビジョン変換器である。Palette[^48]は、拡散ベースの画像間変換フレームワークを統一したものである。PITI[^59]は、生成された結果の品質を向上させる方法として、大規模な事前学習を利用する拡散ベースの画像間翻訳手法である。スケッチガイド拡散のような特定の分野では、[^58]は拡散プロセスを操作する最適化ベースの方法である。これらの手法は実験で検証されている。

## 3 手法

ControlNet は、あらかじめ訓練された画像拡散モデルをタスクに応じた条件で拡張することができるニューラルネットワークアーキテクチャである。3.1節で ControlNet の本質的な構造と各パーツの動機付けを紹介する。3.2節では、Stable Diffusion を例に、ControlNet を画像拡散モデルに適用する方法を詳しく説明する。3.3節では学習目標と一般的な学習方法について詳述し、3.4節では1台のノートパソコンでの学習や大規模計算機クラスタでの学習など、極端なケースでの学習を改善するいくつかのアプローチについて説明する。最後に、セクション3.5で、異なる入力条件を持ついくつかの ControlNet の実装の詳細を記載する。

### 3.1 ControlNet

ControlNet は、ニューラルネットワークブロックの入力条件を操作することで、ニューラルネットワーク全体の挙動をさらに制御する。ここで、「ネットワークブロック」とは、resnet ブロック、conv-bn-relu ブロック、マルチヘッドアテンションブロック、transformer ブロックなど、ニューラルネットワークを構築する際によく使われる単位としてまとめられたニューラル層の集合を指す。

2次元の特徴を例にとると、$\{h,w,c\}$ を高さ、幅、チャンネル番号とする特徴マップ $x\in\mathbb{R}^{h\times w\times c}$ が与えられたとき、パラメータ $\Theta$ を設定したニューラルネットワークブロック $\mathcal{F}(-;\Theta)$ は、$x$ を以下の特徴マップ $y$ に変換する。

$$y=\mathcal{F}(x; \Theta)\tag{1}$$

この手順を図2-(a)に視覚化した。

$\Theta$ の全パラメータをロックし、訓練可能なコピー $\Theta_c$ にクローン化する。コピーされた $\Theta_c$ は、外部の条件ベクトル $c$ で学習される。本論文では、オリジナルと新しいパラメータを「ロックドコピー」「トレーナブルコピー」と呼ぶことにする。元の重みを直接学習するのではなく、このようなコピーを作成する動機は、データセットが小さい時の過学習を避け、数十億の画像から学習した大規模モデルの生産可能な品質を維持するためである。

ニューラルネットワークブロックは、「ゼロコンボリューション」と呼ばれる固有な型のコンボリューション層、即ち重みとバイアスの両方がゼロで初期化された1×1コンボリューション層によって接続されている。このゼロコンボリューションを $\mathcal{Z}(-;-)$ と呼び、2つのパラメータ $\{\Theta_{z1},\Theta_{z2}\}$ を用いて、図2-(b)に示すように、

$$y_c=\mathcal{F}(x;\Theta)+\mathcal{Z}(\mathcal{F}(x+\mathcal{Z}(c;\Theta_{z1});\Theta_c);\Theta_{z2})\tag{2}$$

というControlNetの構造を構成し、$y_c$はニューラルネットワークブロックの出力となる。

![ControlNet](2023-05-29-09-37-07.png)
図2：**ControlNet**
任意のニューラルネットワークブロックに ControlNet を適用するアプローチを示す。$x,y$ はニューラルネットワークの深層特徴量である。「$+$」は特徴量の追加を意味する。「$c$」はニューラルネットワークに追加したい条件である。「ゼロコンボリューション」は、重みもバイアスも0で初期化した1×1コンボリューション層である。

ゼロコンボリューション層の重みとバイアスはともにゼロとして初期化されるため、最初のトレーニングステップでは、

$$\left\{\begin{array}{l}\mathcal{Z}(c;\Theta_{z1})=\bold{0} \\ \mathcal{F}(x+\mathcal{Z}(c;\Theta_{z1});\Theta_c)=\mathcal{F}(x;\Theta_c)=\mathcal{F}(x;\Theta) \\ \mathcal{Z}(\mathcal{F}(x+\mathcal{Z}(c;\Theta_{z1});\Theta_c);\Theta_{z2})=\mathcal{Z}(\mathcal{F}(x;\Theta_c);\Theta_{z2})=\bold{0}\end{array}
\right.\tag{3}$$

となり、これを

$$y_c=y\tag{4}$$

と変換すると、式(2, 3, 4)は、最初のトレーニングステップにおいて、トレーナブル・ロックドコピー両方のニューラルネットワークブロックのすべての入力と出力が ControlNet が存在しない場合と一致することが示される。つまり、ControlNet をいくつかのニューラルネットワークブロックに適用した場合、最適化の前に、深層ニューロンの特徴に影響を与えることはない。ニューラルネットワークブロックの能力、機能、品質は完全に維持され、それ以上の最適化は（そして層をゼロから訓練するのと比較して）ファインチューニングのように速くなる。

ここでは、ゼロコンボリューション層の勾配計算を簡単に説明する。重み $W$ とバイアス B を持つ1×1畳み込み層を考え、任意の空間位置$p$とチャネルワイズインデックス $I$ で、入力マップ $I\in\mathbb{R}^{h\times w\times c}$ が与えられた場合、フォワードパスは

$$\mathcal{Z}(I;\{W,B\})_{p,i}=B_i+\sum^c_jI_{p,i}W_{i,j}\tag{5}$$

と書け、ゼロコンボリューションは $W=0$、$B=0$（最適化前）なので、$I_{p,i}$ がゼロでない場所では、勾配は

$$\left\{\begin{array}{l}\frac{\partial\mathcal{Z}(I;\{W,B\})_{p,i}}{\partial B_i}=1\\
\frac{\partial\mathcal{Z}(I;\{W,B\})_{p,i}}{\partial I_{p,i}}=\sum^c_jW_{i,j}=0\\
\frac{\partial\mathcal{Z}(I;\{W,B\})_{p,i}}{\partial W_{i,j}}=I_{p,i}\not=0\end{array}\right.\tag{6}$$

になり、ゼロコンボリューションが特徴に勾配を引き起こす可能性があることがわかりる。項$I$がゼロになる場合、重みとバイアスの勾配は影響を受けない。特徴量 $I$ がゼロでない限り、最初の勾配降下反復で重み $W$ はゼロでない行列に最適化される。注目すべきは、我々の場合、特徴項はデータセットからサンプリングされた入力データまたは条件ベクトルであり、これは当然、$I$ がゼロでないことを保証する。たとえば、総合損失関数 $\mathcal{L}$ と学習率 $\beta_{lr}\not=0$ の古典的な勾配降下を考えると、「外側」の勾配 $\partial\mathcal{L}/\partial\mathcal{Z}(I;\{W,B\})$ がゼロでない場合、

$$W^*=W-\beta_{lr}\cdot\frac{\partial\mathcal{L}}{\partial\mathcal{Z}(I;\{W,B\})}\odot\frac{\partial\mathcal{Z}(I;\{W,B\})}{\partial W}\not=0\tag{7}$$

が成り立つ。ここで $W^*$ は1勾配降下ステップ後の重み、$\odot$ はアダマール積を表す。このステップのあと、

$$\frac{\partial\mathcal{Z}(I;W^*,B)}{\partial I_{p,i}}=\sum^c_jW^*_{i,j}\not=0\tag{8}$$

でゼロでない勾配が得られ、ニューラルネットワークの学習が開始される。このように、ゼロコンボリューションは、ゼロから最適化されたパラメータへと学習しながら徐々に成長するユニークなタイプの接続層となる。

![Stable DiffusionのControlNet](2023-05-29-11-12-12.png)
図3：**Stable DiffusionのControlNet**
灰色のブロックがStable Diffusion 1.5（同じU-Netアーキテクチャを使っているのでSD V2.1）の構造で、青いブロックがControlNet。

### 3.2 Image Diffusion Model 内の ControlNet

ここでは、Stable Diffusion[^44]を例に、ControlNet を使ってタスクに応じた条件で大規模な拡散モデルを制御する手法を紹介する。

Stable Diffusion は、数十億枚の画像で学習させた大規模なテキストから画像への拡散モデルである。このモデルは基本的に、エンコーダー、ミドルブロック、スキップ接続されたデコーダーを持つ U-net である。 エンコーダーとデコーダーの両方が12ブロックあり、フルモデルでは25ブロック（ミドルブロックを含む）ある。そのうち、8ブロックはダウンサンプリングまたはアップサンプリングのコンボリューション層、17ブロックはメインブロックで、それぞれ4つの resnet 層と2つの Vision Transformer（ViT）が含まれている。各 Vit は、いくつかのクロスアテンションおよび/またはセルフアテンションメカニズムを含んでいる。テキストは OpenAI CLIP によってエンコードされ、拡散時間ステップは位置エンコードによってエンコードされる。

Stable Diffusion では、VQ-GAN [^11]と同様の前処理を行い、512×512画像のデータセット全体を、64×64の小さな「潜在画像」に変換して安定化学習を行う。 そのため、ControlNet が画像ベースの条件を64×64の特徴空間に変換して、畳み込みサイズに合わせる必要がある。 我々は、4×4のカーネルと2×2のストライドを持つ4つの畳み込み層からなる小さなネットワーク $\mathcal{E}(\cdot)$（ReLUで活性化、チャンネルは16、32、64、128、ガウス重みで初期化、フルモデルと共同で学習）を用いて、画像空間の条件c_iを、

$$c_f=\mathcal{E}(c_i)\tag{9}$$

の特徴マップにエンコードする（$c_f$ は変換後の特徴マップ）。このネットワークは、512×512の画像条件を64×64の特徴量マップに変換する。

図3に示すように、U-net の各レベルを制御するために、ControlNet を使用する。 ControlNet の接続方法は計算効率が良いことに注意したい。元の重みがロックされているため、トレーニングに元のエンコーダーの勾配計算が必要ない。これにより、元のモデルに対する勾配計算の半分を回避できるため、トレーニングのスピードアップとGPUメモリの節約が可能になる。ControlNet で安定した拡散モデルをトレーニングする場合、GPUメモリは約23%増、各トレーニング反復で34%の時間増で済む（Nvidia A100 PCIE 40G1台でテスト）。

具体的には、12個の符号化ブロックと安定拡散の中間ブロック1個の学習用コピーを ControlNet で作成する。12個のブロックは4つの解像度（64×64, 32×32, 16×16, 8×8）で、それぞれ3個のブロックを持つ。出力は、U-net の12個のスキップコネクションと1個のミドルブロックに加えられる。 SD は典型的な U-net 構造であるため、この ControlNet アーキテクチャは他の拡散モデルでも使用できる可能性がある。

### 3.3 訓練

画像拡散モデルは、画像を段階的にノイズ除去してサンプルを生成することを学習する。ノイズ除去は、画素空間または学習データから符号化された「潜在」空間で行われる。Stable Diffusion では、潜在的な画像を学習領域として使用する。この文脈では、「画像」、「ピクセル」、「ノイズ除去」という用語はすべて「知覚的潜在空間」[^44]における対応する概念を指している。

画像 $z_0$ が与えられたとき、拡散アルゴリズムは画像に徐々にノイズを加え、ノイズ画像 $z_t$ を生成する。 $t$ が十分に大きい場合、画像は純粋なノイズに近似する。時間ステップ $t$、テキストプロンプト $c_t$ 、およびタスク固有の条件 $c_f$ を含む一連の条件が与えられると、画像拡散アルゴリズムは、ノイズ画像 $z_t$ に加えられるノイズを予測するためのネットワーク $\epsilon_\theta$ を、

$$\mathcal{L}=\mathbb{E}_{z_0,t,c_t,c_f,\epsilon～\mathcal{N}(0,1)}[||\epsilon-\epsilon_\theta(z_t,t,c_t,c_f)||^2_2]\tag{10}$$

で学習する（ここで $\mathcal{L}$ は拡散モデル全体の学習目的である）。この学習目的は、拡散モデルのファインチューニングに直接利用することができる。

学習中、テキストプロンプト $c_t$ の $50\%$ をランダムに空文字列に置き換える。これにより、ControlNet が入力条件マップ（Canny エッジマップや人間の落書きなど）から意味内容を認識することが容易になる。 これは主に、SD モデルにとってプロンプトが見えない場合、エンコーダはプロンプトの代わりとして入力制御マップからより多くの意味内容を学習する傾向があるためである。

### 3.4 訓練の改善

特に、計算装置が非常に限られている場合（ノートPCなど）や非常に強力な場合（大規模なGPUが利用できる計算クラスタなど）、ControlNet のトレーニングを改善するためのいくつかの戦略について説明する。我々の実験では、これらの戦略のいずれかが使用されている場合、実験設定の中で言及する。

**小規模学習**：計算機が限られている場合、ControlNet と Stable Diffusion の接続を部分的に解除することで、収束を早めることができることがわかった。デフォルトでは、図3のように ControlNet を「SD Middle Block」と「SD Decoder Block 1,2,3,4」に接続している。デコーダー1,2,3,4へのリンクを外し、ミドルブロックのみを接続することで、学習速度を約1.6倍に向上できることがわかる（RTX 3070TIノートPC GPUでテスト）。モデルが結果と条件の間に合理的な関連性を示した場合、それらの切断されたリンクは、正確な制御を促進するために、継続的なトレーニングで再び接続することができる。

**大規模学習**：ここでいう大規模学習とは、強力な計算クラスタ（Nvidia A100 80G以上8台）と大規模データセット（少なくとも100万枚の学習画像ペア）の両方が利用可能な状況を指す。これは通常、Cannyで検出されたエッジマップのように、データが容易に入手できるタスクに適用される。この場合、過学習のリスクは比較的低いので、まずControlNet を十分に大きな反復回数（通常は50kステップ以上）訓練し、次に Stable Diffusion のすべての重みを解除してモデル全体を全体として共同訓練することができる。そうすることで、より問題に特化したモデルになると考えられる。

### 3.5 実装

大規模な拡散モデルを様々な方法で制御するために、画像ベースの条件を変えた ControlNet の実装をいくつか紹介する。

**Canny Edge**：Cannyエッジ検出器[^5]（ランダムな閾値）を用いて、インターネットから3Mのエッジ画像とキャプションのペアを取得する。モデルは Nvidia A100 80G で 600GPU時間かけて学習させる。ベースモデルは Stable Diffusion 1.5である。（図4も参照）。

**Canny Edge (Alter)**：上記のCannyエッジデータセットの画像解像度をランク付けし、1k, 10k, 50k, 500kのサンプルでいくつかのサブセットをサンプリングした。同じ実験設定で、データセット規模の効果を検証する。（図22も参照）。

**Hough Line**：学習ベースの深層Hough変換[^13]を用いてPlaces2[^66]から直線を検出し、BLIP[^34]を用いてキャプションを生成します。600kのエッジ画像とキャプションのペアを得る。上記のCannyモデルを出発点とし、Nvidia A100 80Gで150GPU時間かけてモデルを学習する。（図5も参照）。

**HED Boundary**：HED境界検出[^62]を用いて、インターネットから3Mのエッジ画像とキャプションのペアを取得する。モデルはNvidia A100 80Gで300GPU時間かけて学習させる。ベースモデルはStable Diffusion 1.5である。（図7も参照）。

**User Sketching**：HED境界検出[^62]と一連の強力なデータ補強（ランダムな閾値、ランダムな割合の走り書きのマスクアウト、ランダムな形態変換、ランダムな非最大抑制）を組み合わせて、画像から人間の走り書きを合成する。インターネットから500kの落書き画像とキャプションのペアを入手する。上記のCannyモデルを出発点として、Nvidia A100 80Gで150GPU時間かけてモデルを訓練する。なお、より「人間に近い」合成方法[^57]も試したが、この方法は単純なHEDよりもはるかに遅く、目に見える改善とはならなかった。（図6も参照）。

**Human Pose (Openpifpaf)**：学習ベースのポーズ推定法[^27]を用いて、「人間が写っている画像は、全身のキーポイントの少なくとも30%が検出されていなければならない」というシンプルなルールを用いて、インターネットから人間を「見つける」ことができます。 その結果、80kのポーズ画像とキャプションのペアが得られた。 なお、学習条件として、人間の骨格が可視化されたポーズ画像を直接使用している。モデルは、Nvidia RTX 3090TIで400GPU時間かけて学習させる。ベースモデルはStable Diffusion 2.1（図8も参照）。

**Human Pose (Openpose)**：学習ベースのポーズ推定法[^6]を用いて、上記のOpenpifpafの設定と同じルールでインターネットから人間を探すことができる。その結果、200k個のポーズ画像とキャプションのペアを得ることができた。なお、学習条件として、人間の骨格を可視化したポーズ画像を直接使用している。Nvidia A100 80Gを使用し、300GPU時間かけて学習させた。その他の設定は上記のOpenpifpafと同じである。（図9参照）。

**Semantic Segmentation (COCO)**：COCO-Stuffデータセット[^4]はBLIP[^34]によってキャプションが付けられた。164Kのセグメンテーション画像とキャプシャンのペアを得ている。モデルはNvidia RTX3090TIで400GPU時間かけて学習している。ベースモデルは、Stable Diffusion 1.5である。（図12も参照）。

**Semantic Segmentation (ADE20K)**：ADE20Kデータセット[^67]にBLIP[^34]でキャプションを付けた。 164Kのセグメンテーション画像とキャプシャンのペアを得ている。モデルはNvidia A100 80Gで200GPU時間かけて学習される。ベースモデルは、Stable Diffusion 1.5である。（図11参照）。

**Depth (large-scale)**：Midas [^30]を用いて、インターネットから3Mの深度画像とキャプシャンのペアを取得する。モデルはNvidia A100 80Gで500GPU時間かけて学習させる。ベースモデルはStable Diffusion 1.5である。（図23,24,25も参照）。

**Depth (small-scale)**：上記の深度データセットの画像解像度をランク付けして、200k組のサブセットをサンプリングする。このセットは、モデルの訓練に必要な最小限のデータセットサイズの実験に使用される。(図14も参照)。

**Normal Maps**：DIODEデータセット[^56]はBLIP[^34]によってキャプションが付けられた。25,452の通常画像とキャプシャンのペアを得る。モデルはNvidia A100 80Gの100GPU時間で学習される。ベースモデルはStable Diffusion 1.5である。(図13も参照)。

**Normal Maps (extended)**：我々はMidas [^30]を使って深度マップを計算し、次に法線距離から法線マップを実行して「粗い」法線マップを達成する。 上記の法線モデルを出発点として、Nvidia A100 80Gで200GPU時間かけてモデルを訓練する。（図23,24,25も参照）。

**Cartoon Line Drawing**：インターネット上の漫画イラストから線画を抽出するために、漫画線画抽出法[^61]を使用する。漫画画像を人気順にソートすることで、上位1M個の線画・漫画とキャプションのペアを得る。 モデルはNvidia A100 80Gで300GPU時間かけて学習させる。ベースモデルは、Waifu Diffusion （stable diffusion [^36]からコミュニティが開発した興味深い変動モデル）である。（図15も参照）。

## 4 実験

### 4.1 実験設定

本論文の結果はすべてCFGスケール9.0で達成されている。サンプラーはDDIMを使用。デフォルトで20ステップを使用している。モデルをテストするために3種類のプロンプトを使用する：

1. プロンプトなし：空文字列「」をプロンプトとして使用する。
2. デフォルトのプロンプト：Stable diffusionは基本的にプロンプトで学習するため、空文字列はモデルにとって予期せぬ入力となる可能性があり、プロンプトが提供されない場合、SDはランダムなテクスチャマップを生成する傾向がある。 より良い設定は、「an image」、「a nice image」、「a professional image」などの無意味なプロンプトを使用することである。 私たちの設定では、デフォルトのプロンプトとして「a professional, detailed, high-quality image」を使用している。
3. 自動プロンプト：完全自動化パイプラインの最先端最大化品質を検証するため、自動画像キャプション手法（例：BLIP [^34]）を用いて、「default prompt」モードで得られた結果を用いてプロンプトを生成する試みも行っている。生成されたプロンプトは、再び拡散に使用される。
4. ユーザープロンプト：ユーザーがプロンプトを出す。

### 4.2 定性的結果

図4、5、6、7、8、9、10、11、12、13、14、15に定性的結果を示す。

### 4.3 Ablation Study

図20は、ControlNetを使用せずに学習させたモデルとの比較である。このモデルは、StabilityのDepth-to-Imageモデルと全く同じ方法（SDにチャンネルを追加してトレーニングを続ける）でトレーニングされている。

図21はその学習過程を示している。ここで注目したいのは、モデルが突然入力条件に追従できるようになる「突然の収束現象」である。これは、学習率を1e-5とした場合、5000ステップから10000ステップまでの学習過程で起こりうる現象である。

図22は、データセットの規模を変えて学習させたCanny-edgeベースのControlNetである。

### 4.4 従来手法との比較

図14は、Stability社のDepth-to-Imageモデルとの比較である。

図17はPITI[^59]との比較ある。

図18は、スケッチガイド付き拡散[^58]との比較ある。

図19は、タミングトランス[^11]との比較ある。

### 4.5 学習済みモデルの比較

図23、24、25に、さまざまな事前学習済みモデルの比較を示す。

### 4.6 その他のアプリケーション

図16は、拡散過程をマスクすれば、ペンを使った画像編集にモデルが使えることを示している。

図26は、オブジェクトが比較的単純な場合、モデルは細部の制御を比較的正確に実現できることを示している。

図27は、ControlNetを50%の拡散反復にのみ適用した場合、ユーザーは入力形状に従わない結果を得ることができることを示している。

## 5 制約

図28は、意味的な解釈が誤っている場合、モデルが正しいコンテンツを生成することが困難な場合があることを示している。

## 付録

図29は、エッジ検出やポーズ抽出などを行うための本論文の全ソース画像である。

## 参考資料

![CannyエッジによるStable Diffusionの制御](2023-05-29-14-23-17.png)
図4：**CannyエッジによるStable Diffusionの制御**
「自動プロンプト」は、ユーザープロンプトを使用せず、デフォルトの結果画像に基づいてBLIPが生成したものである。Cannyエッジ検出のためのソース画像については、付録も参照のこと。

![Hough lines(M-LSD)によるStable Diffusionの制御](2023-05-29-14-25-56.png)
図5：**Hough lines(M-LSD)によるStable Diffusionの制御**
ライン検出のためのソース画像については、付録も参照のこと。

![落書きによるStable Diffusionの制御](2023-05-29-14-28-17.png)
図6：**落書きによるStable Diffusionの制御**
この落書きは[^58]のもの。

![HED boundary mapによるStable Diffusionの制御](2023-05-29-14-30-40.png)
図7：**HED boundary mapによるStable Diffusionの制御**
HED境界検出のためのソース画像については、付録も参照のこと。

![Openpifpaf poseによるStable Diffusionの制御](2023-05-29-14-32-31.png)
図8：**Openpifpaf poseによるStable Diffusionの制御**
Openpifpafのポーズ検出のソース画像については、付録も参照のこと。

![OpenposeによるStable Diffusionの制御](2023-05-29-14-34-25.png)
図9：**OpenposeによるStable Diffusionの制御**
Openposeのポーズ検出のソース画像については、付録も参照のこと。

![human poseでのStable Diffusionの制御による同一人物の異なるポーズの生成](2023-05-29-14-35-43.png)
図10：**human poseでのStable Diffusionの制御による同一人物の異なるポーズの生成**
画像はチェリーピックではない。 Openposeのポーズ検出のためのソース画像については、付録も参照のこと。

![ADE20KのセグメンテーションマップによるStable Diffusionの制御](2023-05-29-14-40-16.png)
図11：**ADE20K[^67]のセグメンテーションマップによるStable Diffusionの制御**
すべての結果は、デフォルトのプロンプトで生成されている。セグメンテーションマップ抽出のためのソース画像については、付録も参照のこと。

![COCO-StuffのセグメンテーションマップによるStable Diffusionの制御](2023-05-29-14-43-04.png)
図12：**COCO-Stuff[^4]のセグメンテーションマップによるStable Diffusionの制御**

![DIODE normal mapによるStable Diffusionの制御](2023-05-29-14-44-29.png)
図13：**DIODE[^56]normal mapによるStable Diffusionの制御**

![深度ベースControlNetとStable DiffusionV2深度対画像の比較](2023-05-29-14-47-34.png)
図14：**深度ベースControlNetとStable DiffusionV2深度対画像の比較**
なお、この実験では、必要な計算資源を最小限に抑えるため、DepthベースのControlNetを比較的小規模に学習させている。また、比較的大規模に学習させた、より強力なモデルも提供している。

![cartoon line drawingsでのStable Diffusion(anime weights)の制御](2023-05-29-14-50-39.png)
図15：**cartoon line drawingsでのStable Diffusion(anime weights)の制御**
線画は入力であり、対応する「グランドトゥルース」は存在しない。 このモデルは、アーティスティック・クリエーション・ツールで使用することができる。

![Masked Diffusion](2023-05-29-14-53-01.png)
図16：**Masked Diffusion**
Cannyエッジモデルは、マスクされた部分の画像を拡散させることで、画像コンテンツのペンによる編集をサポートすることができる。どの拡散モデルも当然Masked Diffusionをサポートしているので、他のモデルも画像操作に利用される可能性が高い。

![PITIとの比較](2023-05-29-14-56-15.png)
図17：**PITI[^59]との比較**
なお、このタスクでは、「壁」「紙」「コップ」の意味的な整合性を扱うことが難しい。

![スケッチガイド付き拡散との比較](2023-06-02-09-55-10.png)
図18：**スケッチガイド付き拡散[^58]との比較**
個の入力は、彼らの論文で最も困難なケースの1つである。

![タミングトランスフォーマーとの比較](2023-06-02-09-59-12.png)
図19：**タミングトランスフォーマー[^11]との比較**
この入力は、彼らの論文で最も困難なケースの1つである。

![アブレーション研究](2023-06-02-10-02-30.png)
図20：**アブレーション研究**
ControlNetの構造を、Stable Diffusionが拡散モデルに条件を追加するデフォルトの方法として使用している標準的な方法と比較している。

![突然の収束現象](2023-06-02-10-04-17.png)
図21：**突然の収束現象**
ゼロコンボリューションを使用しているため、ニューラルネットワークはトレーニングの間中、常に高品質の画像を予測する。訓練ステップのある時点で、モデルは突然、入力条件に適応するように学習する。これを「突然の収束現象」と呼んでいる。

![異なる規模でのトレーニング](2023-06-02-10-06-55.png)
図22：**異なる規模でのトレーニング**
Canny-edgeを用いたControlNetを、様々なデータセットを用いて、様々な実験設定で学習させたものを示す。

![6種類の検出方法の比較と対応する結果](2023-06-02-10-08-12.png)
図23：**6種類の検出方法の比較と対応する結果**
HEDマップからモルフォロジー変換でスクリブルマップを抽出したもの。

![（続き）6種類の検出方法の比較と対応する結果](2023-06-02-10-09-53.png)
図24：**（続き）6種類の検出方法の比較と対応する結果**

![（続き）6種類の検出方法の比較と対応する結果](2023-06-02-10-11-36.png)
図25：**（続き）6種類の検出方法の比較と対応する結果**

![シンプルなオブジェクトの例](2023-06-02-10-12-15.png)
図26：**シンプルなオブジェクトの例**
拡散内容が比較的単純な場合、このモデルは内容物を操作するための非常に正確な制御を実現することができる。

![粗いレベルの制御](2023-06-02-10-13-44.png)
図27：**粗いレベルの制御**
ユーザーが入力した形状を画像に残したくない場合、最後の50%の拡散反復をControlNetを使わない標準的なSDに置き換えるだけで良い。結果として、画像検索と同様の効果が得られるが、それらの画像は生成されたものである。

![制約](2023-06-02-10-15-49.png)
図28：**制約**
入力画像の意味を誤って認識した場合、強いプロンプトを出したとしても、その悪影響を排除するのは難しい。

![付録：エッジ検出、セマンティックセグメンテーション、ポーズ抽出などのためのすべてのオリジナルソース画像](2023-06-02-10-17-40.png)
図29：**付録：エッジ検出、セマンティックセグメンテーション、ポーズ抽出などのためのすべてのオリジナルソース画像**
なお、一部の画像には著作権がある場合がある。

[^1]:Y.Alaluf, O.Tov, R.Mokady, R.Gal, and A.H.Bermano. Hyperstyle: Stylegan inversion with hypernetworks for real image editing. 2021.

[^2]:O.Avrahami, D.Lischinski, and O.Fried. Blended diffusion for text-driven editing of natural images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18208–18218, 2022.

[^3]:T.Brooks, A.Holynski, and A.A.Efros. In structpix2pix: Learning to follow image editing instructions, 2022.

[^4]:H.Caesar, J.Uijlings, and V.Ferrari. Coco-stuff: Thing and stuff classes in context, 2016.

[^5]:J.Canny. A computational approach to edge detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, PAMI-8(6):679–698, 1986.

[^6]:Z.Cao, G.Hidalgo Martinez, T.Simon, S.Wei, and Y.A.Sheikh. Openpose: Realtime multiperson 2d pose estimation using part affinity fields. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.

[^7]:H.Chen, Y.Wang, T.Guo, C.Xu, Y.Deng, Z.Liu, S.Ma, C.Xu, C.Xu, and W.Gao. Pre-trained image processing transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12299–12310, 2021.

[^8]:Y.Choi, M.Choi, M.Kim, J.-W.Ha, S.Kim, and J.Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8789–8797, 2018.

[^9]:P.Dhariwal and A.Nichol. Diffusion models beat gans on image synthesis. CoRR, 2105, 2021.

[^10]:T.M.Dinh, A.T.Tran, R.Nguyen, and B.-S.Hua. Hyperinverter: Improving stylegan inversion via hypernetwork. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11389–11398, 2022.

[^11]:P.Esser, R.Rombach, and B.Ommer. Taming transformers for high-resolution image synthesis. CoRR, 2012, 2020.

[^12]:R.Gal, Y.Alaluf, Y.Atzmon, O.Patashnik, A.H.Bermano, G.Chechik, and D.Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.

[^13]:G.Gu, B.Ko, S.Go, S.-H.Lee, J.Lee, and M.Shin. Towards light-weight and real-time line segment detection. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

[^14]:D.Ha, A.M.Dai, and Q.V.Le. Hypernetworks. In International Conference on Learning Representations, 2017.

[^15]:Heathen. github.com/automatic1111/stable-diffusion-webui/discussions/2670 , hypernetwork style training, a tiny guide, 2022.

[^16]:A.Hertz, R.Mokady, J.Tenenbaum, K.Aberman, Y.Pritch, and D.Cohen-Or. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

[^17]:J.Ho, A.Jain, and P.Abbeel. Denoising diffusion probabilistic models. NeurIPS, 2020.

[^18]:J.Ho, C.Saharia, W.Chan, D.J.Fleet, M.Norouzi, and T.Salimans. Cascaded diffusion models for high fidelity image generation. CoRR, 2106:15282, 2021.

[^19]:X.Huang, A.Mallya, T.-C.Wang, and M.-Y.Liu. Multimodal conditional image synthesis with product-of-experts gans. arXiv preprint arXiv:2112.05130, 2021.

[^20]:P.Isola, J.-Y.Zhu, T.Zhou, and A.A.Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 1125–1134, 2017.

[^21]:T.Karras, T.Aila, S.Laine, and J.Lehtinen. Progressive growing of gans for improved quality, stability, and variation, 2017.

[^22]:T.Karras, S.Laine, and T.Aila. A style-based generator architecture for generative adversarial networks, 2018.

[^23]:B.Kawar, S.Zada, O.Lang, O.Tov, H.Chang, T.Dekel, I.Mosseri, and M.Irani. Imagic: Text-based real image editing with diffusion models. arXiv preprint arXiv:2210.09276, 2022.

[^24]:G.Kim, T.Kwon, and J.C.Ye. Diffusionclip: Text-guided diffusion models for robust image manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and PatternRecognition, pages 2426–2435, 2022.

[^25]:D.P.Kingma, T.Salimans, B.Poole, and J.Ho. Variational diffusion models. 2107:00630, 2021.

[^26]:Z.Kong and W.Ping. On fast sampling of diffusion probabilistic models. CoRR, 2106, 2021.

[^27]:S.Kreiss, L.Bertoni, and A.Alahi. OpenPifPaf: Composite Fields for Semantic Keypoint Detection and Spatio-Temporal Association. IEEE Transactions on Intelligent TransportationSystems, pages 1–14, March 2021.

[^28]:Kurumuz. [https://blog.novelai.net/novelai-improvements-on-stable-diffusion-e10d38db82ac], novelai improvements on stable diffusion, 2022.

[^29]:S.Kutuzova, O.Krause, D.McCloskey, M.Nielsen, and C.Igel. Multimodal variational autoencoders for semi-supervised learning: In defense of product-of-experts. arXiv preprint arXiv:2101.07240, 2021.

[^30]:K.Lasinger, R.Ranftl, K.Schindler, and V.Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. CoRR, abs/1907.01341, 2019.

[^31]:Y.Lecun, L.Bottou, Y.Bengio, and P.Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998.

[^32]:Y.LeCun, Y.Bengio, and G.Hinton. Deep learning. Nature, 521(7553):436–444, May 2015.

[^33]:J.Lehtinen, J.Munkberg, J.Hasselgren, S.Laine, T.Karras, M.Aittala, and T.Aila. Noise2noise: Learning image restoration without clean data, 2018.

[^34]:J.Li, D.Li, C.Xiong, and S.Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022.

[^35]:C.Meng, Y.He, Y.Song, J.Song, J.Wu, J.-Y.Zhu, and S.Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. In International Conference on Learning Representations, 2021.

[^36]:A.Mercurio. Waifu diffusion, 2022.

[^37]:A.Nichol and P.Dhariwal. Improved denoising diffusion probabilistic models, 2021.

[^38]:A.Nichol, P.Dhariwal, A.Ramesh, P.Shyam, P.Mishkin, B.McGrew, I.Sutskever, and M.Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.

[^39]:T.Park, M.-Y.Liu, T.-C.Wang, and J.Y.Zhu. Semantic image synthesis with spatially-adaptive normalization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2337–2346, 2019.

[^40]:G.Qian, J.Gu, J.S.Ren, C.Dong, F.Zhao, and J.Lin. Trinity of pixel enhancement: a joint solution for demosaicking, denoising and super-resolution. arXiv preprint arXiv:1905.02538, 1(3):4, 2019.

[^41]:A.Radford, J.W.Kim, C.Hallacy, A.Ramesh, G.Goh, S.Agarwal, G.Sastry, A.Askell, P.Mishkin, J.Clark, G.Krueger, and I.Sutskever. Learning transferable visual models from natural language supervision, 2021.

[^42]:A.Ramesh, M.Pavlov, G.Goh, S.Gray, C.Voss, A.Radford, M.Chen, and I.Sutskever. Zero-shot text-to-image generation. In International Conference on Machine Learning, pages 8821–8831. PMLR, 2021.

[^43]:A.Ramesh, P.Dhariwal, A.Nichol, C.Chu, and M.Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 2022.

[^44]:R.Rombach, A.Blattmann, D.Lorenz, P.Esser, and B.Ommer. High-resolution image synthesis with latent diffusion models, 2021.

[^45]:O.Ronneberger, P.Fischer, and T.Brox. U-net: Convolutional networks for biomedical image segmentation. In MICCAI (3), volume 9351 of Lecture Notes in Computer Science, pages 234–241. 2015.

[^46]:N.Ruiz, Y.Li, V.Jampani, Y.Pritch, M.Rubinstein, and K.Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. arXiv preprint arXiv:2208.12242, 2022.

[^47]:D.E.Rumelhart, G.E.Hinton, and R.J.Williams. Learning representations by back-propagating errors. Nature, 323(6088):533–536, Oct. 1986.

[^48]:C.Saharia, W.Chan, H.Chang, C.Lee, J.Ho, T.Salimans, D.Fleet, and M.Norouzi. Palette: Image-to-image diffusion models. In ACM SIGGRAPH 2022 Conference Proceedings, SIG-GRAPH ’22, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393379.

[^49]:C.Saharia, W.Chan, S.Saxena, L.Li, J.Whang, E.Denton, S.K.S.Ghasemipour, B.K.Ayan, S.S.Mahdavi, R.G.Lopes, et al. Photorealistic text-to-image diffusion models with deep language understanding. arXiv preprint arXiv:2205.11487, 2022.

[^50]:R.San-Roman, E.Nachmani, and L.Wolf. Noise estimation for generative diffusion models. CoRR, 2104, 2021.

[^51]:A.Shamsian, A.Navon, E.Fetaya, and G.Chechik. Personalized federated learning using hypernetworks. In International Conference on Machine Learning, pages 9489–9502. PMLR,2021.

[^52]:J.Sohl-Dickstein, E.A.Weiss, N.Maheswaranathan, and S.Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. CoRR, 1503, 2015.

[^53]:J.Song, C.Meng, and S.Ermon. Denoising diffusion implicit models. In ICLR. OpenReview.net, 2021.

[^54]:Y.Song, J.Sohl-Dickstein, D.P.Kingma, A.Kumar, S.Ermon, and B.Poole. Score-based generative modeling through stochastic differential equations. CoRR, 2011:13456, 2020.

[^55]:Stability. stable-diffusion-2-depth, [https://huggingface.co/stabilityai/stable-diffusion-2-depth], 2022.

[^56]:I.Vasiljevic, N.Kolkin, S.Zhang, R.Luo, H.Wang, F.Z.Dai, A.F.Daniele, M.Mostajabi, S.Basart, M.R.Walter, and G.Shakhnarovich. DIODE: A Dense Indoor and Outdoor DEpth Dataset. CoRR, abs/1908.00463, 2019.

[^57]:Y.Vinker, Y.Alaluf, D.Cohen-Or, and A.Shamir. Clipascene: Scene sketching with different types and levels of abstraction, 2022.

[^58]:A.Voynov, K.Abernan, and D.Cohen-Or. Sketch-guided text-to-image diffusion models. 2022.

[^59]:T.Wang, T.Zhang, B.Zhang, H.Ouyang, D.Chen, Q.Chen, and F.Wen. Pretraining is all you need for image-to-image translation, 2022.

[^60]:T.-C.Wang, M.-Y.Liu, J.-Y.Zhu, A.Tao, J.Kautz, and B.Catanzaro. High-resolution image synthesis and semantic manipulation with conditional gans. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 8798–8807, 2018.

[^61]:X.Xiang, D.Liu, X.Yang, Y.Zhu, and X.Shen. Anime2sketch: A sketch extractor for anime arts with deep networks. [https://github.com/Mukosame/Anime2Sketch], 2021.

[^62]:S.Xie and Z.Tu. Holistically-nested edge detection. In 2015 IEEE International Conference on Computer Vision (ICCV), pages 1395–1403, 2015.

[^63]:P.Zhang, B.Zhang, D.Chen, L.Yuan, and F.Wen. Cross-domain correspondence learning for exemplar-based image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5143–5153, 2020.

[^64]:Z.Zhang, J.Ma, C.Zhou, R.Men, Z.Li, M.Ding, J.Tang, J.Zhou, and H.Yang. M6-ufc: Unifying multi-modal controls for conditional image synthesis. arXiv preprint arXiv:2105.14211,2021.

[^65]:J.Zhao, F.Schäfer, and A.Anandkumar. Zero initialization: Initializing residual networks with only zeros and ones. CoRR, abs/2110.12661, 2021.

[^66]:B.Zhou, A.Lapedriza, A.Khosla, A.Oliva, and A.Torralba. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.

[^67]:B.Zhou, H.Zhao, X.Puig, S.Fidler, A.Barriuso, and A.Torralba. Scene parsing through ade20k dataset. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 5122–5130, 2017.

[^68]:X.Zhou, B.Zhang, T.Zhang, P.Zhang, J.Bao, D.Chen, Z.Zhang, and F.Wen. Cocosnet v2: Full-resolution correspondence learning for image translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11465–11475, 2021.

[^69]:J.-Y.Zhu, R.Zhang, D.Pathak, T.Darrell, A.A.Efros, O.Wang, and E.Shechtman. Toward multimodal image-to-image translation. Advances in neural information processing systems,30, 2017.33
