# 粒子物理方法在台海演化模型中的可用边界与接入方案

## 摘要

粒子物理的具体实体理论，例如标准模型粒子、规范相互作用、量子色动力学和散射振幅，没有可识别的台海社会经济对应物，不应直接写入模型。真正可借鉴的是高能物理、统计物理和计算物理共同发展出的多粒子表示、动理学方程、场论响应、重整化、粒子滤波和稀有事件采样方法。

本文把这些方法映射到现有的四方多种群平均场博弈、Graphon 与超图网络、信息传播、竞争风险、离散事件动态系统和大偏差模块。核心原则是：粒子方法只能承担微观近似、潜在状态识别、尺度聚合或尾部概率估计，不能凭空产生新的政治或军事因果关系。

## 1. 概念边界

本文中的“粒子”是计算粒子或异质主体。一个粒子可以代表一类居民、企业、军事单元、地区或一条带权模拟路径。其状态写为：

$$
X_i(t)=\left(s_i,w_i,k_i,m_i,q_i,r_i\right).
$$

其中，$s_i$ 为支持度或认知状态，$w_i$ 为财富或可支配资源，$k_i$ 为技能或生产能力，$m_i$ 为士气，$q_i$ 为库存或后勤状态，$r_i$ 为地区与网络类型。

以下概念不应直接接入：

- 把中国、台湾、美国和日本解释为四种基本粒子；
- 用电荷、色荷、希格斯质量等替代制度、资源或战斗力；
- 把量子叠加解释为政治主体同时选择多个现实行动；
- 在没有观测映射时使用量子振幅或费曼图生成终局概率。

这些类比无法校准，也不能通过消融或样本外验证提高可信度。

## 2. 相互作用粒子系统与平均场极限

对第 $i$ 个主体采用 McKean--Vlasov 型随机动力学：

$$
dX_i(t)=b\left(X_i(t),\mu_t^N,u_i(t),Z_t\right)dt
+\sigma\left(X_i(t),Z_t\right)dW_i(t)
+\gamma\left(X_i(t^-),Z_t\right)dJ_t.
$$

经验分布为：

$$
\mu_t^N=\frac{1}{N}\sum_{i=1}^{N}\delta_{X_i(t)}.
$$

$u_i$ 是主体行动，$Z_t$ 是共同战争与经济状态，$W_i$ 是个体噪声，$J_t$ 是领导中断、重大伤亡或联盟状态改变等共同跳过程。$N$ 增大时，经验分布可逼近连续 MFG 使用的分布 $\mu_t$。这一过程称为传播混沌意义下的平均场极限。

在现有模型中，该层不替代连续多种群 GMFG，而承担两项任务：第一，用有限网络回代检验平均场均衡；第二，在网络高度集中或共同噪声很强时识别平均场近似失效。

建议同时报告：

$$
E_N=\sup_{0\leq t\leq T}W_2\left(\mu_t^N,\mu_t\right),
$$

以及有限主体偏离收益：

$$
\varepsilon_N=\max_i\left[J_i(u_i^*,u_{-i}^*)-\inf_{u_i}J_i(u_i,u_{-i}^*)\right].
$$

$W_2$ 是二阶 Wasserstein 距离，$E_N$ 衡量分布误差，$\varepsilon_N$ 衡量平均场策略回代到有限网络后是否仍近似为 Nash 均衡。

## 3. Graphon 与超图上的粒子相互作用

普通平均场默认每个主体面对同一总体分布。现有模型具有信息、经济依赖、身份认同和伤亡传播四层有向时变网络，因此粒子 $i$ 感受到的局部场应写为：

$$
M_i^{(\ell)}(t)=\frac{1}{N}\sum_{j=1}^{N}G_{ij}^{(\ell)}(t)\,\psi_{\ell}\left(X_i(t),X_j(t)\right).
$$

$\ell$ 表示网络层，$G_{ij}^{(\ell)}$ 是有向权重，$\psi_{\ell}$ 是该层的交互函数。对平台、政府、军方和主要政党等重要玩家，单独保留 Major 状态；普通人口继续采用粒子群近似 Minor 分布。

群聊、组织动员、产业链联合故障或多域杀伤网属于高阶互动，可使用超边 $e$：

$$
M_i^{H}(t)=\sum_{e\ni i}w_e(t)\,\Psi_e\left(\{X_j(t):j\in e\}\right).
$$

该表达避免把真正的群体作用错误拆成若干独立双边作用。

## 4. Boltzmann 意见动力学

设个体意见 $x\in[-1,1]$，两名主体接触后的状态为：

$$
x'=x+\eta P(x,y,r_x,r_y)(y-x)+\xi,
$$

另一名主体按同一机制更新：

$$
y'=y+\eta P(y,x,r_y,r_x)(x-y)+\xi'.
$$

$P$ 是受身份、来源可信度、网络同质性和战争状态影响的说服核，$\eta$ 是互动强度，$\xi$ 与 $\xi'$ 是异质反应。若 $P$ 在异质群体间为负，可以表达反弹与极化。

对应的分布演化可写为弱形式：

$$
\frac{d}{dt}\int \varphi(x)f(x,t)dx
=\frac{1}{2}\iint E\left[\varphi(x')+\varphi(y')-\varphi(x)-\varphi(y)\right]f(x,t)f(y,t)dxdy.
$$

在单次互动很弱但频率很高的极限下，可得到 Fokker--Planck 方程：

$$
\partial_t f=-\partial_x\left(B[f]f\right)
+\frac{1}{2}\partial_{xx}\left(D[f]f\right).
$$

这比 SIR 更适合连续立场、极化、领导影响和选择性接触。SIR 仍适合描述“未知、接触、转发、遗忘”等离散信息状态，两者不能重复拥有同一认知变量。

## 5. 潜在意图的粒子滤波

各方真实意图、介入准备度、政策可信度和领导层风险不可直接观测。设潜在状态为 $Z_t$，公开观测为 $Y_t$：

$$
Z_t=F(Z_{t-1},a_{t-1})+\omega_t,
$$

观测方程为：

$$
Y_t=H(Z_t)+\nu_t.
$$

粒子滤波以加权样本近似后验：

$$
p(Z_t\mid Y_{1:t})\approx\sum_{n=1}^{N_p}w_t^{(n)}\delta_{Z_t^{(n)}}.
$$

权重更新为：

$$
\widetilde{w}_t^{(n)}=w_{t-1}^{(n)}p(Y_t\mid Z_t^{(n)}).
$$

归一化权重为：

$$
w_t^{(n)}=\widetilde{w}_t^{(n)}\big/\sum_m\widetilde{w}_t^{(m)}.
$$

有效样本量为：

$$
N_{\mathrm{eff}}=\frac{1}{\sum_n\left(w_t^{(n)}\right)^2}.
$$

当 $N_{\mathrm{eff}}$ 低于阈值时进行重采样。与 HMM 相比，粒子滤波能处理非线性、非高斯和连续潜在状态；与竞争风险模型的接口是把后验样本送入风险率，而不是用滤波后验直接冒充终局概率：

$$
\lambda_k(t)=E\left[\lambda_k(t\mid Z_t,X_t)\mid Y_{1:t}\right].
$$

## 6. 稀有事件粒子分裂

领导层中断、联盟保障同时失效、多个港口与机场同步瘫痪等事件概率低，普通蒙特卡洛常得到零命中。定义终局事件集合 $A$ 与反应坐标 $\zeta(X)$，建立递增阈值：

$$
z_0<z_1<\cdots<z_K.
$$

目标概率分解为：

$$
P(A)=\prod_{k=1}^{K}P\left(\zeta\geq z_k\mid \zeta\geq z_{k-1}\right).
$$

自适应多层分裂保留更接近目标事件的路径，复制并扰动这些路径，同时淘汰低分路径。为避免“为了得到极端结果而选取阈值”，阈值与反应坐标必须在冻结方案中预先登记，并用普通蒙特卡洛可覆盖的中等概率事件验证无偏性和置信区间覆盖率。

该方法应估计条件尾部概率和到达路径，不能提高事件本身的结构先验，也不能替代竞争风险模型。

## 7. 重整化与尺度一致性

将个体网络逐级聚合为社区、地区和四方宏观状态。设粗粒化算子为 $R_b$，微观演化算子为 $T$。理想的尺度一致性要求：

$$
R_b\,T_{\mathrm{micro}}\approx T_{\mathrm{coarse}}\,R_b.
$$

误差指标可写为：

$$
E_{mathrm{RG}}=\left\|R_bT_{\mathrm{micro}}(X)-T_{\mathrm{coarse}}R_b(X)\right\|.
$$

若误差随聚合快速增大，说明宏观方程丢失了关键中观结构，需要保留社区、平台或供应链节点；若误差稳定，则可用粗粒模型降低计算量。这里的重整化不是宣称社会系统具有严格物理普适类，而是一个可检验的降阶原则。

## 8. 渗流、临界现象与响应函数

通信、能源、供应链和交通网络可把节点可用率记为 $p$。网络最大连通分量比例为：

$$
S(p)=\frac{|C_{\max}(p)|}{|V|}.
$$

当 $p$ 穿越临界区时，$S(p)$ 可能非线性下降。模型不应预设一个普适阈值，而应在四方各自的多层网络上估计临界区、方向性和恢复过程。

对外部冲击 $h(t)$，宏观状态 $M(t)$ 的线性响应为：

$$
\chi(t,s)=\frac{\delta E[M(t)]}{\delta h(s)}.
$$

响应函数可用于识别伤亡、通胀、断供和领导事件对支持度与动员的滞后影响。若响应明显非线性，应回到脉冲、Filippov 或 DEDS 跳变层处理。

## 9. 与现有模块的状态所有权

为避免重复积分，建议采用以下唯一所有权：

| 状态 | 唯一主模块 | 粒子方法的角色 |
|---|---|---|
| 四方连续人口分布 | 多种群 GMFG / FP | 有限粒子近似与回代验证 |
| 有向社会交互 | 多层 Graphon / 超图 | 计算粒子局部场 |
| 连续意见与极化 | Boltzmann / FP 认知层 | 主演化器，可替代同状态的简单扩散项 |
| 潜在意图 | 状态空间模型 | 粒子滤波后验更新 |
| 终局发生时间 | 竞争风险 | 接收粒子后验或路径状态 |
| 离散制度与战役事件 | DEDS / 跳过程 | 作为共同跳生成元 |
| 极端尾部事件 | 大偏差与多层分裂 | 高效概率估计 |
| 硬容量运输 | 网络流求解器 | 粒子层只读取可用容量，不替代求解器 |

## 10. 推荐接入顺序

第一优先级是连续 GMFG 的有限粒子近似。它直接服务于已有的平均可利用度、最坏类型可利用度和有限网络回代偏离收益验证。

第二优先级是潜在意图粒子滤波。它应使用滚动时间验证，比较 HMM、扩展卡尔曼滤波和粒子滤波的对数预测密度、校准误差与退化率。

第三优先级是稀有事件粒子分裂。只对普通蒙特卡洛难以稳定估计的条件尾部使用，并保留冻结反应坐标、有效样本量和重复运行方差。

第四优先级是 Boltzmann 意见层。只有在微观接触或民调分布数据足以识别说服核时，才替换现有认知扩散器；否则保留为结构敏感性模型。

重整化和场论响应先作为诊断，不增加新的终局状态或自由校准系数。

## 11. 验证与消融

新增模块必须通过以下检查：

1. 粒子数收敛：比较 $N$、$2N$、$4N$ 下的分布距离和终局统计量。
2. 有限网络回代：检验平均场策略在真实或拟合网络上的偏离收益。
3. 滤波校准：报告对数预测密度、概率积分变换和有效样本量。
4. 稀有事件验证：在中等概率事件上与直接蒙特卡洛比较。
5. 一次一项消融：分别关闭粒子近似验证、粒子滤波、动理学意见层和多层分裂。
6. 反事实隔离：固定其他随机数种子，确保差异来自被关闭模块。
7. 不确定性传播：网络核、观测误差和结构参数必须共同进入置信区间。

模型发布时应同时报告统计误差、参数不确定性和结构不确定性，不能只给一个终局百分比。

## 12. 结论

粒子物理对台海模型的主要价值，不是提供新的实体因果理论，而是提供一组处理大量异质主体、非平衡分布、潜在状态和稀有跃迁的成熟数学方法。最可信的改进组合是：

$$
\text{有限粒子 GMFG 验证}
+\text{潜在意图粒子滤波}
+\text{稀有事件多层分裂}.
$$

Boltzmann 意见动力学可在数据允许时升级社会传播层；重整化、渗流和响应函数适合做降阶与临界性诊断。标准模型或量子场论的具体物理实体不应进入参数表，也不应成为提高或降低武力、和平及维持现状概率的依据。

## 参考资料

1. Bordenave, McDonald and Proutiere, *A particle system in interaction with a rapidly varying environment: Mean field limits and applications*: https://arxiv.org/abs/math/0701363
2. Kolliopoulos, Larsson and Zhang, *Propagation of chaos for point processes induced by particle systems with mean-field drift interaction*: https://arxiv.org/abs/2303.11426
3. During and Wolfram, *Opinion dynamics: inhomogeneous Boltzmann-type equations modelling opinion leadership and political segregation*: https://arxiv.org/abs/1505.07433
4. Kunsch, *Recursive Monte Carlo filters: Algorithms and theoretical analysis*: https://projecteuclid.org/journals/annals-of-statistics/volume-33/issue-5/Recursive-Monte-Carlo-filters-Algorithms-and-theoretical-analysis/10.1214/009053605000000426.full
5. Gerber and Chopin, *Convergence of sequential quasi-Monte Carlo smoothing algorithms*: https://projecteuclid.org/journals/bernoulli/volume-23/issue-4B/Convergence-of-sequential-quasi-Monte-Carlo-smoothing-algorithms/10.3150/16-BEJ834.full
6. Lopes and Lelievre, *Analysis of the Adaptive Multilevel Splitting method on the isomerization of alanine dipeptide*: https://arxiv.org/abs/1707.00950
