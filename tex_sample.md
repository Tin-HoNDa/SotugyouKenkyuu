# 数学記号記法

## 定義

|表記|$\LaTeX$|備考|
|:--:|:--:|:--:|
|$A=B$|`A = B`|不適切な誤解を招くがよく使われる|
|$A\stackrel{\mathrm{def}}{=}B$|`A \stackrel{\mathrm{def}}{=} B`|最も誤解がない|
|$A:=B$|`A := B`|工学部っぽい|
|$A\triangleq B$|`A \triangleq B`|最近情報系の論文でよく見る|
|$A\equiv B$|`A \equiv B`|理学部っぽい|
|$A\stackrel{\mathrm{def}}{\equiv}B$|`A \stackrel{\mathrm{def}}{\equiv} B`|同値関係と誤解されうる|
|$A:\Leftrightarrow B$|`A :\Leftrightarrow B`|圏論で見た|
|$A\stackrel{\mathrm{def}}{\Leftrightarrow}B$|`A \stackrel{\mathrm{def}}{\Leftrightarrow} B`|命題の定義に用いられる|

## 但し書き

|表記|$\LaTeX$|意味|
|:--:|:--:|:--:|
|$\text{let}$|`\text{let}`|（以下の数式において）この条件を課す|
|$\therefore$|`\therefore`|（前の数式の結論として）ゆえに|
|$\because$|`\because`|（前の数式が導出されたのは）なぜならば|
|$\text{where}$|`\text{where}`|（前の数式を説明するために）ただし|
|$\text{if}$|`\text{if}`|（前の数式は）以下の条件の下で|
|$\text{otherwise}$|`\text{otherwise}`|（ $\text{if}$ を受けて）それ以外|

「$x$ は $0$ 以上の実数とする。この $x$ は $x^2=3$ を満たす。ゆえに $x=\sqrt{3}$ である（なぜならば $x\geq0$ だから）」

$$\begin{aligned}
    \text{let}\,\,x\in\mathbb{R},x\geq0:\,\,x^2=3\\
    \therefore x=\sqrt{3}\quad(\because x\geq0)
\end{aligned}$$

「$y=\text{ReLU}(x)$ である。ただし $\text{ReLU(x)}$ は $x\geq0$ のとき $x$ をそのまま出力し、その他では $0$ をとる関数である。」

$$y=\operatorname{ReLU}(x)\quad\text{where}\,\,\operatorname{ReLU}(x)=\begin{cases}
    x&\text{if}\,\,x\geq0\\
    0&\text{otherwise}
\end{cases}$$

## 集合

### 基本的な集合

|集合|表記|$\LaTeX$|
|:--:|:--:|:--:|
|自然数の集合|$\mathbb{N}$|`\mathbb{N}`|
|整数の集合|$\mathbb{Z}$|`\mathbb{Z}`|
|実数の集合|$\mathbb{R}$|`\mathbb{R}`|
|複素数の集合|$\mathbb{C}$|`\mathbb{C}`|
|任意の体|$\mathbb{K}$|`\mathbb{K}`|

### 範囲を限定するとき

|集合|表記|$\LaTeX$|
|:--:|:--:|:--:|
|$0$ 以上の整数|$\mathbb{Z}_{+}$|`\mathbb{Z}_{+}`|
|$0$ より大きい整数|$\mathbb{Z}_{++}$|`\mathbb{Z}_{++}`|
|$0$ 以上の実数|$\mathbb{R}_{+}$|`\mathbb{R}_{+}`|
|$0$ より大きい実数|$\mathbb{R}_{++}$|`\mathbb{R}_{++}`|

|集合|表記|$\LaTeX$|
|:--:|:--:|:--:|
|$0$ より大きい整数|$\mathbb{Z}_{>0}$|`\mathbb{Z}_{>0}`|
|$0$ 以上の整数|$\mathbb{Z}_{\geqslant0}$|`\mathbb{Z}_{\geqslant0}`|
|$0$ より大きい自然数|$\mathbb{R}_{>0}$|`\mathbb{R}_{>0}`|
|$0$ 以上の自然数|$\mathbb{R}_{\geqslant0}$|`\mathbb{Z}_{\geqslant0}`|

### 集合の演算

|演算|表記|$\LaTeX$|
|:--:|:--:|:--:|
|和|$A\cup B$|`A \cup B`|
|積|$A\cap B$|`A \cap B`|
|差|$A\setminus B$|`A \setminus B`|

複数の集合 $A_1,A_2,\cdots,A_n$ で和や席をとるときは以下の表記が用いられる。

$$\bigcup_{i=1}^nA_i\stackrel{\textrm{def}}{=}A_1\cup A_2\cup\cdots\cup A_n\\\bigcap_{i=1}^nA_i\stackrel{\textrm{def}}{=}A_1\cap A_2\cap\cdots\cap A_n$$

### 集合の元と量化子

$a$ が集合 $A$ の元であるとき、$a$ は $A$ に **属する** という。

|表記|$\LaTeX$|意味|
|:--:|:--:|:--:|
|$a\in A$|`a \in A`|$a$ は集合 $A$ の元である|

「任意の実数に対して」のように指定する記号は **量化子**と呼ばれる。

|量化子|表記|$\LaTeX$|用法|意味|
|:--:|:--:|:--:|:--:|:--|
|全称量化|$\forall$|`\forall`|$\forall x\in X$|集合 $X$ のすべての元 $x$ について / 任意の $x$ について|
|存在量化|$\exists$|`\exists`|$\exists x\in X$|集合 $X$ のある元 $x$ について / $x$ が存在して|
|唯一存在量化|$\exists!$|`\exists!`|$\exists! x\in X$|集合 $X$ のある唯一の元 $x$ について / 唯一の $x$ が存在して|

### 包含関係

|表記|$\LaTeX$|意味|
|:--:|:--:|:--:|
|$A=B$|`A = B`|集合 $A,B$ は等しい|
|$A\subset B$|`A \subset B`|集合 $A$ は集合 $B$ の（真）部分集合である|
|$A\subsetneq B$|`A \subsetneq B`|集合 $A$ は集合 $B$ の真部分集合である|
|$A\subseteq B$|`A \subseteq B`|集合 $A$ は集合 $B$ の部分集合であるかまたは等しい|

### 外延表記と内包表記

#### 外延表記

|表記例|$\LaTeX$|
|:--:|:--:|
|$A=\{1,2,3\}$|`A = \{ 1, 2, 3 \}`|
|$A=\{1,2,\ldots,n\}$|`A = \{ 1, 2, \ldots, n \}`|

#### 内包表記

|表記例|$\LaTeX$|
|:--:|:--:|
|$A=\{x\mid1\leq x\leq n\}$|`A = \{ x \mid 1 \leq x \leq n \}`|

### 集合の直積

複数の集合からひとつずつ元を選んで作られる組を元とする新たな集合を定義するとき、この集合をもとの集合の **デカルト積**（Cartesian product）または **直積**（direct product）という。集合 $A,B$ の直積 $C$ を内包表記で定義すれば$$C=A×B\stackrel{\textrm{def}}{=}\{(a,b)∣a∈A,b∈B\}$$である。

同じ集合 $A$ の $n$ 個についての直積は $A^n$ と簡略表記される。

$$A^n\stackrel{\mathrm{def}}{=}\underbrace{A\times A\times\cdots\times A}_{n個}$$

### 写像（対応、写像、関数）

**対応**（correspondence）：集合 $A$ の元から集合 $B$ の元への対応関係のうちで、イメージ的には **多対多** のもの。

**写像**（map）：集合 $A$ の元から集合 $B$ の元への対応関係のうちで、イメージ的には **多対一** のもの。

関数（function）：写像の特殊ケース。

集合 $A$ から集合 $B$ への写像 $f$ を考えるとき、$A$ を始域、 $B$ を終域といい $f:A→B$ と書く。
特に $x∈A$ が $f(x)∈B$ に写されることまで明記するときは $f:A→B:x↦f(x)$ のように書く。

|表記|$\LaTeX$|
|:--:|:--:|
|$f:A\to B$|`f \colon A \to B`|
|$f:A\to B:x\mapsto f(x)$|`f \colon A \to B \colon x \mapsto f(x)`|

たとえば $f(x)=x^2$ も、より厳密に書きたいならば

$$
\begin{aligned}
    f&:\mathbb R\to\mathbb R_{\geqslant0}:x\mapsto x^2\\
    f&:\mathbb C\to\mathbb R:x\mapsto x^2
\end{aligned}
$$

のようになる。

#### 写像によく使われる文字

|表記|$\LaTeX$|
|:--:|:--:|
|$f$|`f`|
|$g$|`g`|
|$h$|`h`|
|$F$|`F`|
|$G$|`G`|
|$H$|`H`|
|$\gamma$|`\gamma`|
|$\Gamma$|`\Gamma`|
|$\phi$|`\phi`|
|$\varphi$|`\varphi`|
|$\psi$|`\psi`|

### 少し発展的な内容

#### 集合族

**集合族**（family of sets）：集合を集めた集合。筆記体（`\mathcal`）、ドイツ文字（`\mathfrak`）、花文字（`\mathscr`）等の飾り文字で表されることがある。

|文字|`\mathcal`|`\mathfrak`|`\mathscr`|
|:--:|:--:|:--:|:--:|
|A|$\mathcal{A}$|$\mathfrak{A}$|$\mathscr{A}$|
|B|$\mathcal{B}$|$\mathfrak{B}$|$\mathscr{B}$|
|C|$\mathcal{C}$|$\mathfrak{C}$|$\mathscr{C}$|
|D|$\mathcal{D}$|$\mathfrak{D}$|$\mathscr{D}$|
|F|$\mathcal{F}$|$\mathfrak{F}$|$\mathscr{F}$|
|O|$\mathcal{O}$|$\mathfrak{O}$|$\mathscr{O}$|
|P|$\mathcal{P}$|$\mathfrak{P}$|$\mathscr{P}$|
|V|$\mathcal{V}$|$\mathfrak{V}$|$\mathscr{V}$|
|X|$\mathcal{X}$|$\mathfrak{X}$|$\mathscr{X}$|
|Y|$\mathcal{Y}$|$\mathfrak{Y}$|$\mathscr{Y}$|
|Z|$\mathcal{Z}$|$\mathfrak{Z}$|$\mathscr{Z}$|

## 線形代数

### ベクトル空間

|ベクトルの表記|$\LaTeX$|受ける印象|
|:--:|:--:|:--:|
|$v$|`v`|数学寄り|
|$\bm v$|`\bm v`|情報寄り|
|$\vec v$|`\vec{v}`|物理寄り|

|表記|$\LaTeX$|意味|
|:--:|:--:|:--|
|$v\in V$|`v \in V`|$v$ はベクトル空間 $V$ の元である|
|$x\in\mathbb R^d$|`v \in \mathbb{R} ^ d`|$x$ は $d$ 次元実数ベクトル空間 $\mathbb R^d$ の元である|

### ベクトルの表記

$d$ 次元実数ベクトル空間 $\mathbb R^d$ の元は直積で定義されているから、$()$ の中に並べて書ける。横に並べて書くときは行ベクトルという。

$$x=(x_1,x_2,\ldots,x_d)\in\mathbb R^d$$

一方で、縦に並べて書くときは列ベクトルという。

$$
x=\left(
    \begin{matrix}
        x_1\\
        x_2\\
        \vdots\\
        x_d
    \end{matrix}
\right)\in\mathbb R^d
$$

### ベクトルの転置

行と列の並びの方向を入れ替える操作を **転置**（transpose）という。

$$\left(x_1,x_2,\ldots,x_d\right)^\mathrm T=\left(\begin{matrix}
    x_1\\x_2\\\vdots\\x_d
\end{matrix}\right)$$

### 内積

ベクトル同士の内積 $V\times V\to\mathbb R$ は中点 $\cdot$ か角括弧 $\left<\space\space,\space\space\right>$ を用いて表す。実数ベクトル同士であれば、$u\in\mathbb R^n$ と $v\in\mathbb R^n$ の内積は

$$\left<u,v\right>\stackrel{\textrm{def}}{=}u^\mathrm Tv=\sum^n_{i=1}u_iv_i$$

で定義される。

また、関数 $f(x),g(x)$ は無限次元ベクトル空間 $\mathscr X$ の元とみなすことができて、関数どうしの内積を以下のように積分で定義することもある。

$$\left⟨f,g\right⟩\stackrel{\textrm{def}}{=}\int f(x)g(x)dx$$

他にも、ヒルベルト空間 $\mathcal H$ や再生核ヒルベルト空間 $\mathcal K$ の内積というのも、機械学習のカーネル法を勉強していると出てくるが、どの空間での内積かを明記するときは下付き添字を用いて

$$\left<\cdot,\cdot\right>_\mathcal H,\left<\cdot,\cdot\right>_\mathcal K$$​

のようにする。

|$u,v\in V$ の内積|$\LaTeX$|備考|
|:--:|:--:|:--:|
|$u\cdot v$|`u \cdot v`|簡易的|
|$\left<u,v\right>$|`\left< u, v \right>`|一般的なので定義は都度要確認|
|$u^\mathrm Tv$|`u ^ \mathrm{T} v`|有限次元実数ベクトル空間 $V=\mathbb R^n$|
|$\int u(x)v(x)dx$|`\int u(x) v(x) dx`|無限次元実数ベクトル空間 $V=\mathscr X$|
|$\left<\cdot,\cdot\right>_\mathcal H$|`\left< \cdot, \cdot \right> _ \mathcal{H}`|ヒルベルト空間 $V=\mathcal H$|
|$\left<\cdot,\cdot\right>_\mathcal K$|`\left< \cdot, \cdot \right> _ \mathcal{K}`|再生核ヒルベルト空間 $V=\mathcal K$|

### ノルム

ベクトルの長さに相当する。

ベクトル $x\in V$ のノルム $\|\cdot\|:V\to\mathbb R$ は、普通は自分自身との内積の平方根によって与えられるユークリッドノルム（Euclid norm）を用いる。

$$\|x\|\stackrel{\textrm{def}}{=}\sqrt{\left<x,x\right>}$$

これは有限次元実数ベクトル $x\in\mathbb R^n$ であれば、

$$\|x\|=\sqrt{\sum^n_{i=1}x^2_i}=\sqrt{x^2_1+x^2_2+\cdots+x^2_n}$$

となる。有限次元実数ベクトル空間ではこの他に $p$-ノルムまたは $L^p$ ノルムと呼ばれるノルム $\|\cdot\|_p:V\to\mathbb R$ も定義される。$L^p$ ノルムは $p\in\mathbb R_{\geqslant1}$ について定義されていて、

$$\|x\|_p\stackrel{\textrm{def}}{=}\left(\sum^n_{i=1}\left|x_i\right|^p\right)^\frac{1}{p}$$

である。ただし $|\cdot|$ は絶対値記号である。これは $p=2$ のときユークリッドノルムに一致している。

|$x\in V$|$\LaTeX$|備考|
|:--:|:--:|:--|
|$\mid x\mid$|`\mid x \mid`|この表記は一般のノルムを指すのでどんなノルムかは都度要確認|
|$\mid x\mid$|`\mid x \mid`|バナッハ空間（バナッハ空間はベクトル空間とは限らない）|
|$\mid x\mid_\mathcal H$|`\mid x \mid _ \mathcal{H}`|ヒルベルト空間 $V=\mathcal H$|
|$\mid x\mid_\mathcal K$|`\mid x \mid _ \mathcal{K}`|再生核ヒルベルト空間 $V=\mathcal K$|

|$x\in\mathbb R^n$ のノルム|$\LaTeX$|備考|
|:--:|:--:|:--:|
|$\mid x\mid$|`\mid x \mid`|ユークリッドノルム|
|$\mid x\mid_2$|`\mid x \mid _ 2`|ユークリッドノルム|
|$\mid x\mid_p$|`\mid x \mid _ p`|$L_p$ ノルム|
|$\mid x\mid_\infin$|`\mid x \mid _ \infin`|$L^\infin$ ノルム（要素の絶対値が最大のもの）|
|$\mid x\mid_0$|`\mid x \mid _ 0`|$L^0$ ノルム（$0$ でない要素の数）|

### 行列（線形変換）

$n$ 次元実数ベクトル空間 $\mathbb{R}^n$ の元に対する線形変換 $\mathbb{R}^n→\mathbb{R}^m$ は **行列**（matrix）で表すことができる。**線形変換**（linear transform）または **線形写像**（linear map）は線形性を満たす写像のことである。

$n$ 次元実数ベクトル空間 $\mathbb{R}^d$ の元 $x$ に対する線形変換 $A$ は、一般に $m \times n$ 個の数を縦横に並べて$$A=\begin{pmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\a_{21}&a_{22}&\cdots&a_{2n}\\\vdots&\vdots&\ddots&\vdots\\a_{m1}&a_{m2}&\cdots&a_{mn}\end{pmatrix}\in\mathbb R^{m\times n}$$のように書く。$i$ 行 $j$ 列目で代表して$$A=A_{ij},A=a_{ij}$$のように定義することもあり、**単に $A_{ij}$ と言われたときは、それが行列なのか、行列の $i$ 行 $j$ 番目の要素なのかに注意する必要がある。**

特に $m=1$ のときの $f:\mathbb{R}^n\rightarrow \mathbb{R}$ は **線形汎変換**（linear functional）といい、$$f=\begin{pmatrix}a_1&a_2&\cdots&a_n\end{pmatrix}\in\mathbb{R}^n$$のように行ベクトルで表される。

行列を $m$ 個の線形汎変換 $f_{i}:\mathbb{R}^n\rightarrow\mathbb{R}$ を縦に並べたものとして考えるとき $$A=\begin{pmatrix}f_{(1)}\\f_{(2)}\\\vdots\\f_{(m)}\end{pmatrix}$$ と解釈することがあり、同様に $n$ 個の $m$ 次元ベクトル $x_{j}\in\mathbb{R}^m$ を横に並べたものとして考えるとき$$X=\begin{pmatrix}x_{(1)}&x_{(2)}&\cdots&x_{(n)}\end{pmatrix}$$と解釈することもある。

### 行列の転置

ベクトルに転置を定義したように行列にも転置が定義される。

$$A^\text{T}=\begin{pmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\a_{21}&a_{22}&\cdots&a_{2n}\\\vdots&\vdots&\ddots&\vdots\\a_{m1}&a_{m2}&\cdots&a_{mn}\end{pmatrix}^\text{T}=\begin{pmatrix}a_{11}&a_{21}&\cdots&a_{m1}\\a_{12}&a_{22}&\cdots&a_{m2}\\\vdots&\vdots&\ddots&\vdots\\a_{1n}&a_{2n}&\cdots&a_{mn}\end{pmatrix}\in\mathbb{R}^{n\times m}$$

### 零行列

零行列 $O\in\mathbb{R}^{n\times n}$ は正方行列のうちですべての成分が $0$ のものである。

|零行列|$\LaTeX$|
|:--:|:--:|
|$O$|`O`|

### 単位行列

単位行列 $I_n\in\mathbb{R}^{n\times n}$ は正方行列のうちで対角成分がすべて $1$、その他がすべて $0$ の行列である。

$$I_n=\begin{pmatrix}1&0&\cdots&0\\0&1&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&1\end{pmatrix}$$

単位行列のサイズが自明なときは単に $I$ とも書かれ、$I$ の代わりに文字 $E$ が用いられることもある。

また、単位行列の要素はクロネッカーのデルタ $\delta_{ij}$ で定義することができるので、単に $\delta_{ij}$ と書かれているものが単位行列を指していることもある。

$$I=\delta_{ij}\quad\text{where}\space\delta_{ij}=\begin{cases}1&\text{if}\,\,i=j\\0&\text{otherwise}\end{cases}$$

|単位行列|$\LaTeX$|
|:--:|:--:|
|$I$|`I`|
|$I_n$|`I _ n`|
|$E$|`E`|
|$E_n$|`E _ n`|
|$\delta_{ij}$|`\delta _ {ij}`|

### 逆行列

正方行列 $A\in\mathbb{R}^{n\times n}$ が正則ならば、逆行列が唯一つ存在し、それを $A^{-1}$ と表記する。

逆行列は以下の性質を満たす。

$$AA^{-1}=A^{-1}A=I$$

|逆行列|$\LaTeX$|
|:--:|:--:|
|$A^{-1}$|`A ^ {-1}`|

### 行列式

正方行列に対しては行列式（determinant）$\det(A)$ が定義できる。

$\det(A)$ は、$A\in\mathbb{R}^{n\times n}$ を、$n$ 次元ベクトルを $n$ 本、行ごとに並べたものとみなすとき、その $n$ 本のベクトルがつくる図形（$n=2$ ならば平行四辺形、$n=3$ ならば平行六面体、それ以上は超立体）の符号付き体積に一致する。$()$ は省略して $\det A$ と書くこともある。行列の「大きさ」を表現する概念の一つであることから、集合の大きさやノルムのように$$|A|$$と表現されることもある（ノルムとは異なり二十銭で表現されることはない）。

|行列式|$\LaTeX$|
|:--:|:--:|
|$\det(A)$|`\det(A)`|
|$\det A$|`\det A`|

### 行列の対角和（トレース）

正方行列に対しては対角成分の総和演算であるトレース（trace）が定義できる。

$$\operatorname{tr}(A)=\sum^n_{i=1}a_{ii}$$

|行列の対角和|$\LaTeX$|
|:--:|:--:|
|$\operatorname{tr}(A)$|`\operatorname{tr}(A)`|

### 行列の内積

行列はあくまで線形写像の表記の一種に過ぎず、$A\in\mathbb{R}^{m\times n}$ は $m\times n$ 次元のベクトルとみなしてもよい（行列の和とスカラー倍は、ベクトル空間の公理としてのベクトルの和とスカラー倍にそのまま用いることができる）。

ベクトルだとみなしたら内積やノルムを定義したくなるのが道理であり、行列 $A,B\in\mathbb{R}^{m\times n}$ にも内積 $\left<\cdot,\cdot\right>:\mathbb{R}^{m\times n}\times\mathbb{R}^{m\times n}\rightarrow\mathbb{R}$ が定義される。行列の内積はベクトル同様、位置が対応する成分どうしをかけたあとで、すべて足し合わせる。

$$\left<A,B\right>=\operatorname{tr}(A^\text{T}B)=\sum^m_{i=1}\sum^n_{j=1}a_{ij}b_{ij}$$

$\operatorname{tr}(A^\text{T}B)$ という書き方がよく好まれるが、正直わかりにくいと思う。Python で計算するときも `numpy.trace(A.T.dot(B))` とするより `numpy.sum(A * B)` で計算したほうが速い。

他にもベクトルの内積と同様に $A\cdot B$ と書かれることもある。

|行列の内積|$\LaTeX$|
|:--:|:--:|
|$A\cdot B$|`A \cdot B`|
|$\left<A,B\right>$|`\left< A, B \right>`|
|$\operatorname{tr}(A^\mathrm{T}B)$|`\operatorname{tr}(A ^ \mathrm{T} B)`|

### 行列のノルム

行列の内積から誘導されるノルムはフロベニウスノルム（Frobenius norm）と呼ばれ $\|\cdot\|_F:\mathbb{R}^{n\times n}\rightarrow\mathbb{R}$ で表記される。

$$\|A\|_F\stackrel{\textrm{def}}{=}\sqrt{\sum^m_{i=1}\sum^n_{j=1}a^2_{ij}}=\sqrt{\left<A,A\right>}$$

Python で計算するときは `numpy.linalg.norm(A)` で計算できる。

行列 $Y\in\mathbb{R}^{m\times n}$ を、行列 $X\in\mathbb{R}^{m\times p},W\in\mathbb{R}^{n\times p}$ で

$$\hat{Y}=XW^\text{T}$$

と近似したいとき、二乗誤差はフロベニウスノルムの二乗で書けるためよく見かける。たとえば $X$ が固定の最小二乗法ならば、目的関数 $f(W)$ は

$$f(W)=\|Y-\hat{Y}\|^2_F=\|Y-XW^\text{T}\|^2_F$$

と書ける。

|行列のノルム|$\LaTeX$|備考|
|:--:|:--:|:--:|
|$\mid A\mid$|`\mid A \mid`|一般的なノルムを表す表記のため定義は都度要確認|
|$\mid A\mid_F$|`\mid A \mid _ F`|フロベニウスノルム|

### 行列の随伴

複素数 $\mathbb{C}$ の **共役**（conjugate）の概念を複素数ベクトル $\mathbb{C}^n$ や複素数行列 $\mathbb{C}^{m\times n}$ に拡張したのが **随伴**（adjoint）である。

実数ベクトルは転置すればそのベクトルに対応した線形汎関数を得ることができたのだが、複素数ベクトルに同様の概念を考えようとすると転置するだけでは不十分で、転置した上で各要素で複素共役を取らねばならない（リースの表現定理より）。この操作を随伴といい $A^\ast$ や $A^\dag$ で表す。

|$A$ の随伴|$\LaTeX$|
|:--:|:--:|
|$A^\ast$|`A ^ \ast`|
|$A^\dag$|`A ^ \dag`|

$$A^\dag=\begin{pmatrix}a_{11}&a_{12}&\cdots&a_{1n}\\a_{21}&a_{22}&\cdots&a_{2n}\\\vdots&\vdots&\ddots&\vdots\\a_{m1}&a_{m2}&\cdots&a_{mn}\end{pmatrix}^\dag=\begin{pmatrix}\bar{a}_{11}&\bar{a}_{21}&\cdots&\bar{a}_{m1}\\\bar{a}_{12}&\bar{a}_{22}&\cdots&\bar{a}_{m2}\\\vdots&\vdots&\ddots&\vdots\\\bar{a}_{1n}&\bar{a}_{2n}&\cdots&\bar{a}_{mn}\end{pmatrix}\in\mathbb{R}^{n\times m}$$

### エルミート内積

2つの複素数ベクトル $x,y\in\mathbb{C}^n$ の内積 $\mathbb{C}^n\times\mathbb{C}^n\rightarrow\mathbb{C}$ は随伴を用いて

$$\left<x,y\right>\stackrel{\text{def}}{=}x^\dag y=\sum^n_{i=1}\bar{x}_iy_i$$

のように定義される。

### ブラ - ケット記法による内積

量子力学や量子情報理論における波動関数は無限次元複素数ベクトルで表現できるので、ある粒子の状態を記述する波動関数はその粒子の状態ベクトルとも呼ばれる。

量子力学で扱う粒子は、大抵は1次元から3次元の空間座標 $x$ に時間軸 $t$ を加えた $(x,t)\in\mathbb{R}^d$ のどこかの領域 $\Omega\in\mathbb{R}^d$ に存在するものと仮定する。粒子の状態を波動関数は文献によって $\psi(x),\psi(t),\psi(x,t)$ などと表記揺れがあるが、

1. ある時間における状態を固定して見たいので $\psi(x)$ と書いている
2. 量子コンピュータなど、粒子を観測する座標は固定して見ているので $\psi(t)$ と書いている
3. 粒子の状態が定常状態に達しており時間に依存しない $\psi(x)$ と書いている
4. 空間座標 $x$ と時刻 $t$ と切り離して考えたいので、あとで $\psi_t(x)$ などと書くつもりである
5. $(x,t)$ といちいち書くのが面倒くさいので単に $x$ と書いている

などが考えられるので文脈から把握する（物理学者は表記揺れに寛容なので察するしかない）。以下では $(x,t)$ といちいち書くのが面倒くさいので単に $x$ と書いている。

$\mathbb{R}^d$ 上に存在する粒子の波動関数が $\psi(x)$ で表現されるとき、その粒子が領域 $\Omega\in\mathbb{R}^d$ に存在する確率は、$$\int_\Omega\left|\psi(x)\right|^2dx$$ で表される。この確率は $\mathbb{R}^d$ 全体で積分したときに $1$ になるように規格化するので、$$\int_\mathbb{R^d}\left|\psi(x)\right|^2dx=1$$とする。

ここで $|\cdot|$ の記号が少しややこしいのだが、複素関数 $\psi\in\mathcal{H}$（$\mathcal{H}$ は複素ヒルベルト空間）について、$|\psi|$ と $|\psi(x)|$ で意味が異なる。$|\psi|$ は複素関数の関数ノルムを表しているため、

$$|\psi|^2=\left<\psi,\psi\right>=\int\overline{\psi(x)}\psi(x)dx$$

となる。一方で $|\psi(x)|$ は複素数 $\psi(x)\in\mathbb{C}$ の絶対値を表しているため、

$$|\psi(x)|^2=\overline{\psi(x)}\psi(x)$$

である。$|\psi|$ と $|\psi(x)|$ の間には

$$|\psi|^2=\left<\psi,\psi\right>=\int\overline{\psi(x)}\psi(x)dx=\int|\psi(x)|^2dx$$

の関係が成り立つ。特に $\mathbb{R}^d$ 全体での積分について、上の関係式の第二項の表記を $\left⟨\psi∣\psi\right⟩$ とし、第四項との関係を抜き出して、粒子の存在確率の式を

$$\left<\psi|\psi\right>\stackrel{\text{def}}{=}\int_\mathbb{R^d}|\psi(x)|^2dx$$

と書くことにする。もちろん $\left<\psi|\psi\right>=1$ である。

こう書くと $\left<\psi|\psi\right>$ を $\left<\psi|\right.$ と $\left.|\psi\right>$ に分けたくなる。なった人がいた。それで、

$$\begin{align*}\psi&=\left.|\psi\right>\\\psi^\dag&=\left<\psi|\right.\end{align*}$$

と書くことにして、$\left<\psi|\right.$ を **ブラベクトル**（bra vector）、$\left.|\psi\right>$ を **ケットベクトル**（ket vector）と呼ぶことになった（あるいはそれぞれ単にブラ、ケットともいう）。括弧は英語で「bracket」なのでそれを c（center） の部分で割ったというシャレだろうか。ブラベクトルが行ベクトルでケットベクトルが列ベクトルに対応している。随伴を取ると互いに入れ替わるので

$$\left.|\psi\right>^\dag=\left<\psi|\right.,\quad\left<\psi|\right.^\dag=\left.|\psi\right>$$

である。状態ベクトルはケットベクトルで表す。

量子力学では粒子の生成・消滅、座標・運動量、状態の時間発展など様々なものを **演算子**（operator）で表す。演算子は $\hat{A}$ のように上にハット記号をつけて表記する。演算子はケットベクトルには左から、ブラベクトルには右から作用する。すなわち

$$\begin{align*}\hat{A}\left.|\psi\right>&=\left.|\phi\right>\\\left<\psi|\right.\hat{A}&=\left<\phi|\right.\end{align*}$$

のように状態ベクトルの変化を表現する。

量子情報理論で特に重要なのは状態の時間発展を表す **時間発展演算子**(time evolution operator)である。シュレージンガー方程式から導かれる性質として、閉じた系の時間発展演算子はユニタリ演算子 $\hat{U}$ によって表される。すなわち、時刻 $0$ における状態ベクトル $\left.|\psi_0\right>$ と時刻 $t$ における状態ベクトル $\left.|\psi_t\right>$ の間には、

$$\hat{U}\left.|\psi_0\right>=\left.|\psi_t\right>$$

の関係が成り立つ。

汎用量子コンピュータは閉じた系であり、計算に上の式を利用している。量子コンピュータにおける状態ベクトルは量子ビットであり、**量子計算**（quantum calculation）とは量子ビットに所望の時間発展演算子に対応するユニタリ行列をかけることをいう。

### 量子ビット

古典情報理論で扱うビット（bit）は $0$ か $1$ かで情報を表したが、量子情報理論で登場する量子ビット(qubit: quantum bit の略)は $\left|0\right>$ と $\left|1\right>$ の重ね合わせ（線形和）で状態を表す。

$\left|0\right>,\left|1\right>\in\mathbb{C}^2$ は

$$\left|0\right>\stackrel{\textrm{def}}{=}\begin{pmatrix}1\\0\end{pmatrix},\quad\left|1\right>\stackrel{\textrm{def}}{=}\begin{pmatrix}0\\1\end{pmatrix}$$

と定義されており、複素数 $\alpha,\beta\in\mathbb{C}(|\alpha|^2+|\beta|^2=1)$ を用いて

$$\left|\psi\right>=\alpha\left|0\right>+\beta\left|1\right>=\begin{pmatrix}\alpha\\\beta\end{pmatrix}$$

と表現できる $\left|\psi\right>$ が $1$ 量子ビットの表現できる情報の範囲である。先程、量子状態の観測確率は状態ベクトルの自身とのエルミート内積で表されると言った。量子ビットは理論的には観測に失敗することを考慮していない。つまり確率 $1$ で $0$  か $1$ のどちらかの状態が観測されるものとするから、先に断りを入れたように

$$\left<\psi|\psi\right>=(\bar{\alpha}\quad\bar{\beta})\begin{pmatrix}\alpha\\\beta\end{pmatrix}=\bar{\alpha}\alpha+\bar{\beta}\beta=|\alpha|^2+|\beta|^2=1$$

を要請するのである。

量子ビットは観測したときに結果が $0$ か $1$ に定まる。量子ビットが保持しているのは「どちらが何%の確率で観測されるか」という情報である。 $0,1$ が観測される確率をそれぞれ $p_0,p_1$ とおくと、

$$p_0=|\alpha|^2\\p_1=|\beta|^2$$

である。

$$\left|0\right>=1\cdot\left|0\right>+0\cdot\left|1\right>$$

であるから、$\left|0\right>$ は $p_0=1,p_1=0$ で「必ず $0$ が観測される」ことを意味し、これは古典ビットの $0$ に対応する状態になる。同様にして $\left|1\right>$ は古典ビットの $1$ に対応する。

複数量子ビットは量子ビットどうしのテンソル積 $\otimes$（`\otimes`）で定義される。テンソルやテンソル積についてはテンソル代数の章で説明するが、

$$\left|z\right>=\left|x\right>\otimes\left|y\right>\quad\text{where}\left|x\right>=\begin{pmatrix}x^0\\x^1\end{pmatrix},\left|y\right>=\begin{pmatrix}y^0\\y^1\end{pmatrix}$$

とするとき、

$$z^{ij}\stackrel{\textrm{def}}{=}x^iy^j$$

により定義される。添え字が上付きなのは反変成分表示していることを意味する（下付きだと共変成分表示の意味になる）。反変とか共変というのが何を意味しているかわからない人は、量子情報理論ではあまり気にしなくてよく、ただのベクトルの成分表示だと思ってよい。

$z$ には添字が2つついているので、$ij$ 成分を $i$ 行 $j$ 列目に行列表示すると、

$$\left|z\right>=\begin{pmatrix}x^0y^0&x^0y^1\\x^1y^0&x^1y^1\end{pmatrix}$$

となる。これが $2$ 量子ビットで表現可能な情報量である。いまは添字が２つだからいいが、$3$ 量子ビット以上になると添字が3つ以上になり行列では書けないので、量子情報理論の文脈では縦に並べ直すことが多い。

$$\left|z\right>=\begin{pmatrix}x^0y^0\\x^0y^1\\x^1y^0\\x^1y^1\end{pmatrix}$$

これは表記の仕方の問題なので本質的には意味は変わらない。さて、この規則に従って

$$\left|00\right>\stackrel{\textrm{def}}{=}\left|0\right>\otimes\left|0\right>=\begin{pmatrix}1\\0\\0\\0\end{pmatrix},\qquad\left|01\right>\stackrel{\textrm{def}}{=}\left|0\right>\otimes\left|1\right>=\begin{pmatrix}0\\1\\0\\0\end{pmatrix}\\\left|10\right>\stackrel{\textrm{def}}{=}\left|1\right>\otimes\left|0\right>=\begin{pmatrix}0\\0\\1\\0\end{pmatrix},\qquad\left|11\right>\stackrel{\textrm{def}}{=}\left|1\right>\otimes\left|1\right>=\begin{pmatrix}0\\0\\0\\1\end{pmatrix}$$

とおく。$\left|0\right>,\left|1\right>$ のときの議論と同様にして、$\left|00\right>,\left|01\right>,\left|10\right>,\left|11\right>$ はそれぞれ古典ビットの $00,01,10,11$ に対応する。

$$\left|z\right>=z^{00}\left|00\right>+z^{01}\left|01\right>+z^{10}\left|10\right>+z^{11}\left|11\right>$$

と規定展開できる。$\left|00\right>$ の観測確率 $p_{00}$ は

$$p_{00}=|z^{00}|^2$$

3量子ビット以降についても以下同様で、一般に $n$ 量子ビットは $2^n$ 通りのすべての可能性の観測確率を保持しているため、複数量子ビットに対する操作が可能になれば莫大な情報を内部的に保持した演算が可能になる。これが量子コンピュータが期待されている所以である。

[数学記号記法一覧（集合・線形代数）](https://zenn.dev/wsuzume/articles/b0b3a51cac5d7fe4555b#%E9%9B%86%E5%90%88%E3%81%AE%E6%BC%94%E7%AE%97)

[数学記号記法一覧（解析学・テンソル解析）](https://zenn.dev/wsuzume/articles/d3e88a408dc235)
