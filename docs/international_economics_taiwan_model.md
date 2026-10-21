# 国际经济学模型在台海演化系统中的应用

## 摘要

台海冲突不是封闭的四方战役，而是贸易、产业链、资本流动、汇率、制裁、技术扩散和联盟政策共同作用的开放经济系统。国际经济学模块的任务，是把跨境经济约束接入战役、财政、货币、金融、劳动、社会认知和政治决策的同一条模拟路径。

本文提出一个分工明确的组合：用结构引力和 PPML 识别贸易成本，用多区域多部门 CGE 清算数量与价格，用全球投入产出网络传播中间品冲击，用 Eaton--Kortum 与 Melitz 模型刻画技术来源和企业异质性，用国际收支与开放经济 DSGE 表达汇率、储备和融资约束，再用随机博弈描述制裁、出口管制和第三国响应。

## 1. 建模对象与边界

四个主要经济体记为：

$$
r\in\{\mathrm{CHN},\mathrm{TWN},\mathrm{USA},\mathrm{JPN}\}.
$$

其余经济体按贸易集团、资源类型和政治距离聚合。部门 $s$ 至少区分农业、能源、采矿、基础材料、一般制造、电子与芯片、交通运输、生产性服务和居民服务。

国际经济层读取战役损伤、港口和航线状态、制裁事件、政策选择与国内产能；向其他模块返回进口数量、价格指数、外汇需求、税基、企业退出、就业、库存和供应链可用率。它不独立决定战争终局。

## 2. 结构引力与 PPML

双边部门贸易流采用结构引力形式：

$$
E[X_{ij,s,t}]=\exp\left(\alpha_{i,s,t}+\delta_{j,s,t}-\theta_s\ln\tau_{ij,s,t}+\beta_s Z_{ij,t}\right).
$$

$X_{ij,s,t}$ 是出口地 $i$ 向进口地 $j$ 的部门贸易额，$\alpha$ 和 $\delta$ 吸收多边阻力及供需变化，$\tau$ 是综合贸易成本，$Z$ 包括协定、同盟、共同语言和地理等变量。

战争期贸易成本分解为：

$$
\ln\tau_{ij,s,t}=\ln\tau_{ij,s}^{0}+c_{ij,t}^{\mathrm{route}}+c_{ij,s,t}^{\mathrm{insurance}}+c_{ij,s,t}^{\mathrm{sanction}}+c_{ij,s,t}^{\mathrm{delay}}.
$$

航线风险、保险、制裁和延误分别进入成本。估计采用 PPML，不对贸易额先取对数，因此能保留零贸易流并缓解异方差造成的偏误。

结构引力只负责识别贸易成本和双边份额，不负责同时生成 GDP、工资和财政余额。WTO 的量化贸易研究表明，在基线份额和贸易成本一致时，结构引力、CGE 和 exact-hat 反事实具有可比基础；差异主要来自基线校准和模型结构。

## 3. Armington 多区域多部门 CGE

进口地区 $r$ 对同一产品的不同来源不是完全替代品。部门复合品为：

$$
Q_{r,s,t}=\left[\sum_j\alpha_{rj,s}^{1/\sigma_s}M_{rj,s,t}^{(\sigma_s-1)/\sigma_s}\right]^{\sigma_s/(\sigma_s-1)}.
$$

$\sigma_s$ 是来源替代弹性。能源、先进设备、芯片制造材料和特定军民两用零部件通常不能采用与一般消费品相同的替代弹性。

生产函数可写为资本、劳动和中间投入的嵌套结构：

$$
Y_{r,s,t}=A_{r,s,t}F_s\left(K_{r,s,t},L_{r,s,t},V_{r,s,t}\right).
$$

其中 $V$ 是由国内与进口中间品合成的投入束。市场清算满足：

$$
Y_{r,s,t}+M_{r,s,t}=C_{r,s,t}+I_{r,s,t}+G_{r,s,t}+E_{r,s,t}+\Delta S_{r,s,t}.
$$

$\Delta S$ 是库存变化。战时不得假设市场总能瞬时清算，因此 CGE 需要与现有非均衡数量调整和 DQ--SFC 库存层相连；CGE拥有价格与预算一致性，DQ--SFC拥有合同、库存和数量配给。

## 4. 全球与国内投入产出网络

静态投入产出关系为：

$$
x_t=A_t x_t+f_t.
$$

和平基线下有：

$$
x_t=(I-A_t)^{-1}f_t.
$$

但战争期存在产能、库存、物流和替代限制，不能机械使用 Leontief 逆矩阵。节点实际产出受到联合约束：

$$
x_{r,s,t}\leq\min\left\{K_{r,s,t}^{\mathrm{cap}},L_{r,s,t}^{\mathrm{eff}},I_{r,s,t}^{\mathrm{avail}},H_{r,s,t}^{\mathrm{log}}\right\}.
$$

四项分别是有效产能、有效劳动、中间品可用量和物流吞吐能力。国内贸易与国际贸易应放在同一多层网络中：国内边表示省区或城市间流动，国际边表示跨境流动。冲击可通过替代供应商、库存释放、配给和转运改变路径。

OECD 的 ICIO 数据提供跨国家、跨行业的生产、消费、投资和中间品流，可作为全球网络基线；中国、台湾、日本和美国的本地投入产出表用于细化四方内部网络。

## 5. Eaton--Kortum 技术与来源选择

对技术差异明显的部门，采购来源份额可写为：

$$
\pi_{ij,s,t}=\frac{T_{i,s,t}\left(c_{i,s,t}\tau_{ij,s,t}\right)^{-\theta_s}}{\sum_k T_{k,s,t}\left(c_{k,s,t}\tau_{kj,s,t}\right)^{-\theta_s}}.
$$

$T$ 表示技术能力，$c$ 是单位成本，$\theta_s$ 控制技术离散程度。该框架适合刻画能源设备、精密机械、化工材料、先进芯片和军工投入的来源切换。

国产替代不应设为固定比例，而应通过技术成熟度、良率、认证周期、单位成本、产能爬坡和物流风险共同改变 $T$、$c$ 与 $\tau$。替代能力还受到累计学习影响：

$$
T_{r,s,t+1}=T_{r,s,t}+\eta_{r,s}R_{r,s,t}+\kappa_{r,s}\ln\left(1+Q_{r,s,t}^{\mathrm{cum}}\right)-\delta_{r,s}^{T}T_{r,s,t}.
$$

$R$ 是研发投入，$Q^{\mathrm{cum}}$ 是累计产量。该式只描述能力演进，不保证任何投入可以在短期内替代。

## 6. Melitz 异质企业与出口退出

企业 $f$ 的生产率为 $\varphi_f$，出口目的地 $j$ 的利润为：

$$
\pi_{fj,t}^{\mathrm{export}}=R_{fj,t}(\varphi_f)-C_{f,t}-F_{fj,t}^{\mathrm{export}}.
$$

当利润低于零时企业退出该出口市场。冲突提高固定合规成本、保险与运输成本，也可能造成融资约束和需求下降。因此贸易冲击同时作用于出口强度边际和出口企业数量边际。

企业进入、退出和转产概率写为：

$$
P(a_{f,t}=a)=\frac{\exp\left(V_{f,t}^{a}/\mu_f\right)}{\sum_b\exp\left(V_{f,t}^{b}/\mu_f\right)}.
$$

$a$ 可取留在国内、维持出口、第三地转口、海外投资、军品转产或退出。该层适合企业数据较充分的芯片、机械、航运和关键材料行业；数据不足的部门继续采用代表性生产账户，避免制造无法识别的企业分布。

## 7. 国际收支、汇率与储备

国际收支恒等式为：

$$
CA_{r,t}+KA_{r,t}+\Delta R_{r,t}=0.
$$

$CA$ 是经常账户，$KA$ 是资本与金融账户，$R$ 是外汇储备。进口支付能力不仅取决于本币财政资源，还取决于可兑换外汇、结算网络和对手方风险。

风险调整后的利率平价为：

$$
i_{r,t}-i_t^*=E_t\left[e_{r,t+1}-e_{r,t}\right]+\rho_{r,t}^{\mathrm{risk}}+\rho_{r,t}^{\mathrm{sanction}}.
$$

资本管制和数量限制存在时，利率平价只作为影子价格条件，不应强制完全套利。储备缓冲按支付需求消耗：

$$
R_{r,t+1}=R_{r,t}+FX_{r,t}^{\mathrm{in}}-FX_{r,t}^{\mathrm{out}}-I_{r,t}^{\mathrm{defense}}.
$$

四方结构必须不同：美国具有美元融资与安全资产优势；日本是主要债权经济体但能源进口依赖高；台湾外汇缓冲与出口能力较强但港航集中；中国拥有大规模国内市场和制造体系，同时面临能源、关键技术、资本流动和地方债务约束。

## 8. 开放经济 DSGE 的跨期调整

家庭消费 Euler 条件可写为：

$$
1=E_t\left[\beta_r\frac{U_C(C_{r,t+1})}{U_C(C_{r,t})}\frac{1+i_{r,t}}{1+\pi_{r,t+1}}\right].
$$

企业投资受到调整成本：

$$
K_{r,t+1}=(1-\delta_r)K_{r,t}+I_{r,t}-\frac{\phi_r}{2}\left(\frac{I_{r,t}}{K_{r,t}}-\bar g_r\right)^2K_{r,t}.
$$

DSGE负责预期、跨期消费投资和货币政策传导，不重复计算 CGE 的部门市场清算。战争造成的巨大离散状态切换由 DEDS 和跳过程提供，DSGE仅在每个制度状态内描述连续调整。

## 9. 制裁、出口管制与联盟博弈

主要玩家的政策选择写为随机博弈：

$$
a_{r,t}^{*}=\arg\max_{a_{r,t}}E_t\left[\sum_{h=0}^{H}\beta_r^hU_r\left(S_{t+h},a_{t+h}\right)\right].
$$

行动包括关税、金融制裁、出口管制、保险限制、第三国执法、补贴、储备释放和物资援助。政府效用同时包含安全、国内福利、财政成本、联盟信誉和选举或支持度。

小型第三国和企业群体可使用主要玩家--次要玩家 MFG。它们对制裁的遵从不是固定比例，而取决于预期处罚、转口收益、网络位置和政治距离：

$$
P_{k,t}^{\mathrm{comply}}=\frac{1}{1+\exp\left[-V_{k,t}^{\mathrm{comply}}\right]}.
$$

## 10. 商品、能源与关键原料

粮食、石油、天然气和关键矿物具有库存与短期供给刚性，需要单独的部分均衡子层：

$$
S_{c,t+1}=S_{c,t}+Q_{c,t}+M_{c,t}-D_{c,t}-L_{c,t}.
$$

$S$ 是库存，$Q$ 是国内生产，$M$ 是进口，$D$ 是需求，$L$ 是运输和战损。现货价格由库存缺口、预期和风险溢价共同决定：

$$
\Delta\ln P_{c,t}=\alpha_c\left(D_{c,t}-Q_{c,t}-M_{c,t}\right)-\gamma_c S_{c,t}+\rho_{c,t}^{\mathrm{risk}}.
$$

该层向农业、工业、通胀与军事后勤返回实物可用量和价格，不能只把商品冲击写成 GDP 百分比。

## 11. 战争--贸易--政治闭环

完整因果链为：

战役损伤 → 节点容量与贸易成本 → 进口、产出与价格 → 财政、就业与财富 → 支持度和政策选择 → 动员、维修与联盟行动。

每月或每季更新顺序应固定：先处理 DEDS 事件与物理容量，再计算网络流和贸易成本；随后进行 CGE 与非均衡数量迭代，更新金融和国际收支；最后更新家庭、企业、政府与联盟策略，并把资源返回下一期战役模块。

这条链必须在每条蒙特卡洛路径内运行。禁止先独立模拟贸易，再把平均贸易损失乘到战役结果上。

## 12. 状态所有权

| 状态 | 唯一主模块 | 国际经济模块接口 |
|---|---|---|
| 双边贸易成本 | 结构引力与事件层 | PPML 估计基线和弹性 |
| 部门价格与数量清算 | 多区域 CGE | 接收贸易份额与容量 |
| 库存和合同配给 | DQ--SFC | 接收进口与价格 |
| 跨国中间品传播 | ICIO 网络 | 返回投入可用率 |
| 技术来源份额 | Eaton--Kortum | 返回供应商切换 |
| 企业进入退出 | Melitz 企业层 | 返回企业数与产能 |
| 汇率、储备与外汇缺口 | 国际收支和金融层 | 约束进口支付 |
| 跨期预期 | 开放经济 DSGE | 影响消费、投资与政策 |
| 硬物流容量 | 网络流求解器 | 约束可实现贸易流 |
| 战争终局 | 竞争风险和战役层 | 读取经济状态，不由贸易层单独决定 |

## 13. 数据与识别

建议数据包括 UN Comtrade 或海关双边产品贸易、WTO 关税与贸易政策、OECD ICIO、各方投入产出表、IMF 国际收支与国际金融统计、世界银行物流指标、能源和粮食统计、港口与航运数据、企业海关与财务微观数据。

估计顺序为：先用和平期和历史冲击估计贸易成本、替代弹性、投入产出系数和企业退出响应；再以疫情、自然灾害、制裁和航运中断做外部验证；最后才把参数带入台海情景。战时无法识别的参数必须使用冻结区间和结构敏感性分析。

## 14. 验证与消融

1. 引力层报告 PPML 拟合、零贸易流、固定效应和样本外预测。
2. CGE 层检查商品、要素、政府和国际收支恒等式残差。
3. ICIO 层检查国内外中间投入守恒和节点容量。
4. 企业层比较出口进入、退出与规模分布。
5. 金融层检查储备、外债、汇率和资本流闭合。
6. 分别关闭结构引力动态、企业异质性、金融制裁、技术替代和全球网络传播。
7. 比较静态 Leontief、容量约束网络和完整动态模型。
8. 对贸易成本、替代弹性、第三国遵从和技术爬坡做联合不确定性传播。

只有完整闭合、验证和消融通过后，国际经济模块才可以影响终局概率发布。

## 15. 结论

最适合现有台海模型的国际经济组合是：

推荐组合：PPML 结构引力 + 多区域多部门 CGE + ICIO 容量网络 + 国际收支约束。

Eaton--Kortum用于技术来源与替代，Melitz用于关键行业的企业进入退出，开放经济 DSGE用于跨期预期，随机博弈用于制裁和联盟政策。传统比较优势和要素禀赋理论保留为解释基准，但不能单独承担封锁、库存、金融制裁、企业退出和战时生产的预测。

## 参考资料

1. WTO, *Structural Gravity*: https://www.wto.org/english/res_e/reser_e/structural_gravity_e.htm
2. Bekkers, *The welfare effects of trade policy experiments in quantitative trade models*: https://www.wto.org/english/res_e/reser_e/ersd201902_e.htm
3. OECD, *Inter-Country Input-Output Tables*: https://www.oecd.org/en/data/datasets/inter-country-input-output-tables.html
4. OECD, *Economic security and vulnerabilities in international supply chains*: https://www.oecd.org/en/publications/economic-security-in-a-changing-world_4eac89c7-en/full-report/economic-security-and-vulnerabilities-in-international-supply-chains_dc88aefa.html
5. Eaton and Kortum, *Technology and Bilateral Trade*: https://www.nber.org/papers/w6253
6. Alvarez and Lucas, *General Equilibrium Analysis of the Eaton-Kortum Model*: https://www.nber.org/papers/w11764
7. Melitz, *The Impact of Trade on Intra-Industry Reallocations and Aggregate Industry Productivity*: https://www.nber.org/papers/w8881
8. Melitz and Redding, *Heterogeneous Firms and Trade*: https://www.nber.org/papers/w18652
9. IMF, *Geoeconomic Fragmentation and Commodity Markets*: https://www.imf.org/en/publications/wp/issues/2023/09/28/geoeconomic-fragmentation-and-commodity-markets-539614
