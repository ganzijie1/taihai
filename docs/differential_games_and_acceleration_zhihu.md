# 微分博弈：从 HJI 方程到 GPU 加速

> 本文讨论连续时间对抗决策的数学结构、常用解法和工程加速。公式采用知乎 Markdown/LaTeX 写法。重点不是堆砌名词，而是说明：每一种算法究竟在解什么、为什么慢，以及怎样在不破坏均衡含义的前提下提速。

## 1. 微分博弈研究什么

最优控制只有一个决策者；微分博弈则有两个或更多决策者，每个人的动作都会改变同一套动力系统。

设系统状态为 $x(t)\in\mathbb{R}^n$，双方控制分别为 $u(t)$ 与 $v(t)$：

$$
\dot{x}(t)=f\bigl(t,x(t),u(t),v(t)\bigr),\qquad x(0)=x_0.
$$

在二人零和博弈中，玩家一最小化损失，玩家二最大化同一个损失：

$$
J_{t,x}(u,v)=g\bigl(x(T)\bigr)
+\int_t^T \ell\bigl(s,x(s),u(s),v(s)\bigr)\,\mathrm{d}s.
$$

这里：

- $f$ 是状态演化规律；
- $\ell$ 是即时损失；
- $g$ 是终端损失；
- $T$ 是博弈时域。

微分博弈真正困难的地方不是多一个控制变量，而是“我会预判你在预判我”。策略空间、信息结构和行动顺序不同，均衡也会不同。

## 2. 先说清楚策略与信息

### 2.1 开环与反馈策略

开环策略在初始时刻给出整个控制轨迹：

$$
u(t)=\mu(t;x_0),\qquad v(t)=\nu(t;x_0).
$$

反馈策略则根据当前状态调整：

$$
u(t)=\mu\bigl(t,x(t)\bigr),\qquad v(t)=\nu\bigl(t,x(t)\bigr).
$$

反馈策略能够响应中途扰动，通常更符合闭环决策，但求解代价也高得多，因为它要求在整个状态空间上求策略，而不是只求一条轨迹。

### 2.2 同时行动与领导者—追随者

若双方同时选择最优反应，常用 Nash 均衡。若领导者先承诺策略、追随者观察后响应，则是 Stackelberg 博弈：

$$
\min_u J_L\bigl(u,v^*(u)\bigr),\qquad v^*(u)\in\mathrm{argmin}_{v}J_F(u,v).
$$

这是一个双层优化问题。把 Stackelberg 问题直接当作同时行动 Nash 问题，往往会得到不同结论。

### 2.3 非预见策略

在连续时间对抗中，策略不能利用未来信息。若对手的两条控制在 $[t,s]$ 上相同，那么本方策略在这段时间内也必须给出相同响应。这种“非预见性”是价值函数和动态规划成立的重要条件。

## 3. 从动态规划推导 HJI 方程

定义零和博弈的下值函数：

$$
V^-(t,x)=\inf_{u}\sup_{v}J_{t,x}(u,v).
$$

在很短的时间 $\Delta t$ 内，动态规划原理给出：

$$
V^-(t,x)=\inf_u\sup_v
\left\{
\ell(t,x,u,v)\Delta t
+V^-\bigl(t+\Delta t,x+f(t,x,u,v)\Delta t\bigr)
\right\}.
$$

对第二项作一阶展开，减去 $V^-(t,x)$，再除以 $\Delta t$ 并令 $\Delta t\to 0$，得到 Hamilton–Jacobi–Isaacs 方程：

$$
-\partial_t V^-(t,x)
=\inf_u\sup_v
\left\{
\ell(t,x,u,v)+\nabla_x V^-(t,x)^\top f(t,x,u,v)
\right\}.
$$

终端条件为：

$$
V^-(T,x)=g(x).
$$

定义 Hamiltonian：

$$
H^-(t,x,p)=\inf_u\sup_v
\left\{\ell(t,x,u,v)+p^\top f(t,x,u,v)\right\},
$$

则方程写成：

$$
-\partial_t V^-(t,x)=H^-(t,x,\nabla_x V^-).
$$

上值函数交换极小与极大次序：

$$
H^+(t,x,p)=\sup_v\inf_u
\left\{\ell(t,x,u,v)+p^\top f(t,x,u,v)\right\}.
$$

如果满足 Isaacs 条件：

$$
H^-(t,x,p)=H^+(t,x,p),
$$

上下值一致，博弈具有值。若不满足，就不能悄悄把 $\inf\sup$ 与 $\sup\inf$ 互换。

经典光滑解经常不存在，因此 HJI 通常在黏性解意义下理解。稳定、单调、一致的数值格式是收敛到正确黏性解的核心。

## 4. 随机微分博弈

若状态受随机扰动：

$$
\mathrm{d}x_t=f(t,x_t,u_t,v_t)\,\mathrm{d}t
+\sigma(t,x_t,u_t,v_t)\,\mathrm{d}W_t,
$$

生成元会多出二阶扩散项：

$$
-\partial_t V
=\inf_u\sup_v
\left\{
\ell+\nabla V^\top f
+\frac{1}{2}\operatorname{tr}
\left(\sigma\sigma^\top\nabla^2 V\right)
\right\}.
$$

这是一类二阶 HJBI 方程。扩散能提高解的正则性，但引入 Hessian、线性系统和更严格的稳定性限制，计算量通常更大。

若还存在突发事件，可加入跳过程：

$$
\mathrm{d}x_t=f\,\mathrm{d}t+\sigma\,\mathrm{d}W_t
+\kappa(t,x_{t^-},z)\,N(\mathrm{d}t,\mathrm{d}z).
$$

相应方程包含非局部积分项：

$$
\mathcal{I}[V](x)
=\int
\left[
V(x+\kappa(x,z))-V(x)
-\nabla V(x)^\top\kappa(x,z)\mathbf{1}_{\{|z|<1\}}
\right]\nu(\mathrm{d}z).
$$

跳强度、跳幅和控制如果相互依赖，必须共同进入 Hamiltonian，而不能先在外部随机生成事件，再假装它与策略无关。

## 5. 非零和与多玩家博弈

非零和情形中，每个玩家都有自己的价值函数：

$$
J_i(u_1,\ldots,u_N)
=g_i(x_T)+\int_t^T \ell_i(x_s,u_{1,s},\ldots,u_{N,s})\,\mathrm{d}s.
$$

反馈 Nash 均衡对应一组耦合 HJB 方程：

$$
-\partial_t V_i
=\min_{u_i}
\left\{
\ell_i(x,u_i,u_{-i}^*)
+\nabla V_i^\top f(x,u_i,u_{-i}^*)
\right\},
\qquad i=1,\ldots,N.
$$

每个最优策略又依赖所有价值函数：

$$
u_i^*(t,x)
\in\mathrm{argmin}_{u_i}
\left\{
\ell_i+\nabla V_i^\top f
\right\}.
$$

这类耦合系统常用策略迭代、最佳反应、虚拟博弈或 Newton 类方法求解。玩家数增加时，直接求解会迅速失控；若玩家可交换且数量很大，平均场博弈通常更合适。

## 6. 线性二次微分博弈：可验证的基准

考虑线性系统：

$$
\dot{x}=Ax+B_1u+B_2v,
$$

以及零和二次型目标：

$$
J=\frac{1}{2}x(T)^\top S_Tx(T)
+\frac{1}{2}\int_0^T
\left(x^\top Qx+u^\top R_1u-v^\top R_2v\right)\,\mathrm{d}t.
$$

令价值函数为：

$$
V(t,x)=\frac{1}{2}x^\top P(t)x,
$$

最优反馈为：

$$
u^*=-R_1^{-1}B_1^\top Px,\qquad v^*=R_2^{-1}B_2^\top Px.
$$

$P(t)$ 满足广义 Riccati 方程：

$$
-\dot{P}
=A^\top P+PA+Q
-PB_1R_1^{-1}B_1^\top P
+PB_2R_2^{-1}B_2^\top P.
$$

线性二次模型很重要，因为它给出解析或高精度基准，可用于检查符号、控制方向、时间反演、梯度和加速后端。复杂模型如果连 LQ 基准都过不了，就不应直接解释现实。

## 7. 常用数值路线

### 7.1 网格法与水平集法

对低维 HJI，可在状态网格上使用迎风差分、ENO/WENO、Lax–Friedrichs 数值 Hamiltonian 和 TVD Runge–Kutta 时间推进。优点是语义清晰、可求整个反馈策略和可达集；缺点是网格点数随维数指数增长：

$$
N_{\text{grid}}=m^d,
$$

其中每维 $m$ 个节点、状态维数为 $d$。这就是维数灾难。

### 7.2 半拉格朗日法

半拉格朗日离散沿特征线回溯：

$$
V_k(x)
=\min_u\max_v
\left\{
\Delta t\,\ell(x,u,v)
+\mathcal{I}[V_{k+1}]
\bigl(x+\Delta t f(x,u,v)\bigr)
\right\},
$$

$\mathcal{I}$ 表示插值。它通常允许比显式 CFL 限制更大的时间步，但插值、边界处理和控制优化会成为瓶颈。

### 7.3 策略迭代与 Howard 型方法

固定策略时，HJI 退化为线性或较容易的 PDE；求出价值函数后再更新策略：

$$
(u^{k+1},v^{k+1})
\in\mathrm{argmin}_{u}\,\mathrm{argmax}_{v}
\left\{\ell+\nabla V^k{}^\top f\right\}.
$$

策略迭代常比朴素值迭代收敛快，但每轮成本更高。实际效率取决于线性求解器、预条件器和策略切换的稳定性。

### 7.4 直接配点与最优控制转录

将状态、控制和动力学约束离散成非线性规划：

$$
\min_U\max_V\;J(X,U,V)\quad\text{s.t.}\quad F(X,U,V)=0.
$$

再使用 SQP、内点法、iLQGames 或 DDP 类算法。它适合求局部开环轨迹和滚动时域策略，但一般不等于求出了全局反馈 Nash 均衡。

### 7.5 BSDE、神经 PDE 与深度虚拟博弈

高维随机博弈可把 HJB/HJBI 与 BSDE 联系起来，用神经网络近似价值函数、梯度或控制。深度虚拟博弈把 $N$ 人问题拆成并行的单人最佳反应，再反复更新。优点是避开全张量网格；缺点是只得到近似解，训练损失小不自动等于策略可利用度小。

## 8. 加速的四个层次

### 8.1 层次一：先减少不必要的求解

最有效的加速往往不是换 GPU，而是减少必须求解的问题数量：

- 跨时间步和相邻情景热启动；
- 已收敛路径活动掩码；
- 事件发生前后使用不同精度；
- 对价值函数或策略建立可信代理模型；
- 用滚动时域只重算受新信息影响的窗口；
- 用误差估计器决定何时细化网格。

若迭代映射为 $z^{k+1}=G(z^k)$，阻尼更新为：

$$
z^{k+1}=(1-\alpha)z^k+\alpha G(z^k),
\qquad 0<\alpha\le 1.
$$

Anderson 加速则利用最近若干残差的线性组合，近似构造更好的搜索方向。它常能显著减少固定点轮数，但必须设置病态保护和回退策略。

### 8.2 层次二：向量化、批量化与编译

GPU 擅长大批量、规则、算术密集任务，不擅长被 Python 逐点调用的小算子。应把玩家、路径、情景、空间点和控制候选尽可能批量化：

$$
X\in\mathbb{R}^{B\times N_x\times d},
$$

其中 $B$ 同时容纳多路径或多情景。

实施重点是：

- 将时间循环写入编译图，而不是在 Python 中逐步调度；
- 融合梯度、Hamiltonian、投影和残差计算；
- 避免每步把标量从 GPU 取回 CPU；
- 固定形状，减少 JIT 重编译；
- 对稀疏线性系统使用适合结构的专用求解器。

JAX 官方文档强调，公平计时必须分离编译与执行，并在异步设备上调用 `block_until_ready()`；否则测到的可能只是任务提交时间。

### 8.3 层次三：更好的线性与非线性求解器

隐式 HJI/HJBI、策略评估和 Newton 步通常归结为：

$$
A(z^k)\,\delta z=-F(z^k).
$$

可使用：

- 稀疏直接法：适合中小规模、重复结构；
- Krylov 法：GMRES、BiCGSTAB、MINRES；
- 几何或代数多重网格：处理椭圆型和扩散主导问题；
- 块预条件：按玩家、时间、空间或物理场拆分；
- Jacobian-vector product：避免显式形成完整 Jacobian；
- inexact Newton：早期迭代不必把线性子问题解得过精。

对控制极值引起的非光滑性，可采用半光滑 Newton、策略迭代或主动集方法。盲目使用普通 Newton 可能在策略切换面附近振荡。

### 8.4 层次四：降低维数

维数灾难不能靠单纯堆 GPU 根治。可考虑：

- 稀疏网格与自适应稀疏网格；
- 低秩张量和张量列车；
- POD、平衡截断和流形降维；
- 特征线或粒子法；
- 神经隐式表示与 neural operator；
- 利用对称性、守恒量和相对坐标降维；
- 平均场极限替代大量近似可交换玩家。

任何降维都必须检查被删除方向是否恰好承载稀有但关键的策略切换。

## 9. 不同问题应选择不同加速路线

| 问题结构 | 首选路线 | 主要风险 |
|---|---|---|
| 2 至 4 维零和可达性 | 单调网格、水平集、GPU/JAX | 网格边界与数值耗散 |
| 中等维随机零和 | 半拉格朗日、稀疏网格、域分解 | 插值误差与二阶项成本 |
| 多玩家非零和 | 最佳反应、策略迭代、并行虚拟博弈 | 收敛到局部或错误均衡 |
| 实时反馈 | 滚动时域、iLQGames/DDP、热启动 | 只近似有限窗口均衡 |
| 高维 HJI | DeepReach、BSDE、神经隐式表示 | 缺少全局误差与安全保证 |
| 大量近似同质玩家 | MFG/GMFG | 有限玩家回代误差 |
| 脉冲或混合博弈 | QVI、事件驱动离散、主动集 | 切换面和 Zeno 行为 |

## 10. 一个可执行的工程流程

### 第一步：建立小型真值问题

至少准备三个基准：

1. 有解析解的 LQ 零和博弈；
2. 低维可用细网格求解的追逃博弈；
3. 与正式模型结构相同、但规模缩小的耦合博弈。

### 第二步：先剖析，再选后端

同时报告叶函数耗时和包含子调用耗时。后者会重叠，不能直接相加。对 GPU 还应报告 kernel 数、平均 kernel 时长、同步点、数据传输和显存峰值。

### 第三步：逐层优化

推荐顺序是：

1. 消除重复计算并加入热启动；
2. 批量化与算子融合；
3. 改善固定点、Newton 和线性求解器；
4. 再考虑 GPU、多 GPU 与神经近似；
5. 每一步均与未优化基线做相同精度的消融。

### 第四步：验收的不只是速度

至少检查：

$$
\varepsilon_{\mathrm{PDE}}
=\left\|\partial_t V+H(x,\nabla V)\right\|,
$$

$$
\varepsilon_{\mathrm{dyn}}
=\max_k\left\|x_{k+1}-\Phi(x_k,u_k,v_k)\right\|,
$$

以及策略可利用度：

$$
\operatorname{Exploit}_1
=J_1(u_1,u_2)-\inf_{\tilde u_1}J_1(\tilde u_1,u_2).
$$

非零和博弈应对每个玩家分别计算可利用度。零和问题还应检查上下值差：

$$
\varepsilon_{\mathrm{Isaacs}}
=\left|V^+(t,x)-V^-(t,x)\right|.
$$

最终结果接近并不充分；价值函数、策略、状态轨迹和事件时点也应处于容许误差内。

## 11. 当前全模型剖析给出的现实提醒

在一个包含多种群 MFG、滚动微分博弈和动态宏观闭合的正式仿真中，30 秒现场采样得到 2983 个有效样本。滚动微分博弈的包含子调用占比约 25.3%，其中候选策略评价的叶函数占比约 18.4%。这说明实际瓶颈不是“微分方程太难”这一句空话，而是候选策略组合被重复评价。

因此该模型的优先项不是立刻换成神经网络，而是：

1. 把候选策略维统一批量化；
2. 缓存与候选无关的状态转移和约束项；
3. 用热启动和可信域减少无效候选；
4. 在相同随机扰动下比较候选，降低蒙特卡洛方差；
5. 只有在这些工作完成后，再判断 GPU 或代理模型的真实收益。

这个结论不能机械推广到所有微分博弈，但它说明了一个普遍原则：优化前必须知道时间花在 PDE、线性代数、策略枚举、自动微分，还是 Python 调度上。

## 12. 常见误区

**误区一：用了 GPU 就一定更快。** 小网格、短向量和大量同步会让 GPU 比编译后的 CPU 更慢。

**误区二：求出一条最优轨迹就是反馈均衡。** 轨迹优化通常只给定初值和局部邻域有效，不等于整个状态空间上的反馈解。

**误区三：训练损失很小就是 HJI 解。** 还需要边界条件、PDE 残差、可利用度和独立轨迹回代。

**误区四：把所有玩家合成一个社会规划者。** 这样会消除战略互动，得到合作最优而不是 Nash 均衡。

**误区五：只比较终局概率。** 中间状态偏差可能互相抵消；模型一旦换政策或换冲击，误差会重新暴露。

**误区六：加大超时时间等于加速。** 延长执行窗口只能避免外层监控提前终止，不能减少任何一次浮点运算。

## 13. 结论

微分博弈是一套把动力系统、最优控制与战略互动统一起来的框架。零和反馈博弈的核心是 HJI 方程，随机问题增加二阶扩散项，跳变问题增加非局部算子，非零和问题则形成耦合 HJB 系统。

求解时没有万能算法：低维问题应优先保留可验证的网格法，高维问题才需要稀疏表示、BSDE 或神经 PDE；实时问题可用滚动时域近似，但必须承认它与无限时域反馈均衡的差别。

真正可靠的加速顺序是：先减少求解次数，再融合和批量化算子，然后改善非线性与线性求解器，最后才扩展到 GPU、多 GPU 或学习型代理。速度提升只有在 PDE 残差、策略可利用度、价值函数和事件结果同时通过门禁时才成立。

## 参考资料

1. Başar 与 Olsder，《Dynamic Noncooperative Game Theory》：动态非合作博弈、反馈 Nash 与追逃博弈的系统教材。[SIAM 页面](https://epubs.siam.org/doi/10.1137/1.9781611971132)
2. Botkin、Hoffmann 与 Turova，稳定 HJBI 数值格式，并展示二维至四维微分博弈算例。[SIAM Journal on Scientific Computing](https://epubs.siam.org/doi/10.1137/100801068)
3. Soravia，追逃博弈与 Isaacs 方程黏性解。[SIAM Journal on Control and Optimization](https://epubs.siam.org/doi/10.1137/0331027)
4. Bansal 与 Tomlin，DeepReach：面向高维 HJ 可达性的神经 PDE 方法。[arXiv](https://arxiv.org/abs/2011.02082)
5. Han 与 Hu，面向多玩家随机微分博弈的 Deep Fictitious Play。[arXiv](https://arxiv.org/abs/1903.09376)
6. Stanford ASL，基于 JAX 的 HJ reachability 实现。[GitHub](https://github.com/StanfordASL/hj_reachability)
7. Molu，LevelSetPy：基于 CuPy 的 GPU 加速双曲型 HJ/HJI 求解器。[arXiv](https://arxiv.org/abs/2507.11542)
8. JAX 官方文档，JIT、异步调度与正确基准测试方法。[JIT 文档](https://docs.jax.dev/en/latest/201/jit.html)；[基准测试](https://docs.jax.dev/en/latest/benchmarking.html)
