# V8.2 四方差异化农业、工业、发展与劳动经济学深化

## 1. 升级目的

V8.2 不再用同一经济方程仅替换四组系数，而是保留统一核算边界，在中国大陆、台湾、美国和日本内部使用不同的行为机制。新增机制在每个月进入生产、劳动、食品安全和价格清算，并通过 DQ-SFC、CGE、社会状态和战役模块反馈下一期。

## 2. 状态所有权

DQ-SFC 继续拥有商品库存、订单、合同价格、结算和总就业；动态 CGE 继续拥有资本存量、要素价格与市场清算；V6.2 继续拥有三产业就业份额。V8.2 只拥有农业预期与投入、食品交换权利、企业类型构成、岗位空缺与匹配、技能、疲劳、人力资本、技术能力和政策可信度。

该划分防止同一库存、资本或就业量在多个模块重复更新。

## 3. 中国大陆

中国大陆采用 Lewis--Fei--Ranis 与 Harris--Todaro 结构转型、国有与民营企业双重结构、地方财政影响、制造业学习、城乡迁移摩擦和动态政策承诺。国家统计局 2025 年公报给出的三产业增加值占比为 6.7%、35.6% 和 57.7%，城镇化率为 67.9%，城镇调查失业率均值为 5.2%。

## 4. 台湾

台湾采用小型开放经济、半导体关键节点、中小企业主体、低食品自给和快速老龄化结构。经济部资料显示中小企业占企业数 98% 以上并提供近八成就业；农业部 2025 年热量口径粮食自给率为 30.36%。模型因此分别表达企业数量分布、关键节点集中、进口通道和家庭食品获得能力。

## 5. 日本

日本采用超高龄劳动力、关系型供应链、进口食品和中间品依赖以及企业调整迟滞。农林水产省公布的 2025 年热量口径粮食自给率为 37%，核心农业从业者平均年龄约 69.2 岁。其农业劳动、土地供给和进口风险不能使用美国参数。

## 6. 美国

美国采用高岗位转换、市场融资、企业进入退出、任务创新和农业 TFP 模型。USDA 的农业生产账户显示 1948--2023 年农业 TFP 年均增长约 1.40%，因此农业供给主要通过技术、资本和中间投入变化，而不是劳动力数量扩张。

## 7. 农业预期与投入

农户预期收益按适应性预期更新：

$$E_tR_{t+1}=(1-\lambda_R)E_{t-1}R_t+\lambda_R R_t.$$

风险调整后的目标投入为：

$$x_t^{*}=\Pi_{[x_{\min},x_{\max}]}
\left(E_tR_{t+1}-\gamma_f\sigma_{R,t}-\gamma_c(1-C_t)\right).$$

其中，$C_t$ 为政策可信度。农业技术不按时间指数外推，而是向各方技术前沿收敛：

$$A_{t+1}^{A}=A_t^{A}+g_A C_t\left(\bar A^{A}-A_t^{A}\right)-\delta_A D_t.$$

## 8. 食品自给与交换权利

可获得食品由国内供给和受封锁影响的进口共同组成：

$$F_t^{\mathrm{avail}}=s_A F_t^{\mathrm{dom}}+(1-s_A)L_t(1-b_t)F_t^{\mathrm{imp}}.$$

家庭交换权利满足：

$$E_{t+1}^{f}=E_t^{f}+\lambda_E
\left(\omega_pp_t+\omega_gg_t+\omega_LL_t+\omega_FF_t^{\mathrm{avail}}-E_t^{f}\right).$$

食品安全指标为 $F_t^{\mathrm{avail}}E_t^{f}$，它能够区分“总量存在但家庭买不到”和“总量本身不足”。

## 9. 可持续农业剩余

Jorgenson 意义上的可持续剩余为：

$$Q_t^{\mathrm{sur}}=Y_{A,t}-C_{A,t}^{\min}-Z_t-F_t-L_t-\Delta S_t.$$

该量仅作为发展约束和政策诊断，不创建第二套食品库存。

## 10. 企业类型与产业组织

每方工业企业分为公共或大型既有企业、专业化民营企业和中小企业。企业类型份额按复制动态更新：

$$m_{k,t+1}=\frac{m_{k,t}\exp\left(\eta_m(\pi_{k,t}-\bar\pi_t)\right)}
{\sum_jm_{j,t}\exp\left(\eta_m(\pi_{j,t}-\bar\pi_t)\right)}.$$

企业生产率通过产出学习、融资和损害更新；关键节点集中损失为：

$$\ell_t^{\mathrm{conc}}=H_a
\left(\omega_DD_t+\omega_bb_t+\omega_L(1-L_t)\right).$$

$H_a$ 是各方的有界集中度先验，不冒充公开 HHI 观察。

## 11. 搜索匹配、技能与疲劳

劳动市场紧张度和岗位匹配为：

$$\theta_t=\frac{v_t}{u_t},\qquad
p_t=\Pi_{[0,1]}\left(\mu_t\theta_t^{\alpha}\right).$$

有效劳动为：

$$L_t^{\mathrm{eff}}=L_t^{\mathrm{account}}
M_tH_t\left(1-\omega_ff_t\right).$$

其中，总就业 $L_t^{\mathrm{account}}$ 仍由 DQ-SFC 持有，V8.2 只计算匹配楔子 $M_t$、技能 $H_t$ 和疲劳 $f_t$。

## 12. 发展能力与政策承诺

政策可信度由财政能力、支付系统、物流、恐慌和损害共同更新：

$$C_{t+1}=C_t+\lambda_C\left(C_t^{*}-C_t\right).$$

人力资本和技术能力满足：

$$H_{t+1}=H_t+\phi_HG_t-\delta_ff_t-\delta_DD_t-\delta_aa_t,$$

$$A_{t+1}=A_t+\phi_A S_tH_t(\bar A-A_t)-\delta_AD_t.$$

其中，$S_t$ 为国家能力，$a_t$ 为老龄人口份额。

## 13. 同路径闭环

V8.2 当月形成农业、工业、服务和有效劳动乘数，直接修改 CGE 节点可实现产出；随后 DQ-SFC 清算实际交易、库存、就业和合同通胀。实际 GDP、工业产出、食品压力与服务拥堵再进入财政、金融、社会、MFG、战役和下一月生产。

不存在仿真结束后按结论追加的概率修正。

## 14. 数据分类

台湾粮食自给率、日本粮食自给率、中国三产业结构、中国城镇失业率、台湾中小企业就业和美国农业 TFP 属于公开观察或其直接派生量。工业集中度、制度调整速度和战时融资可得性属于有界校准先验，必须接受区间敏感性和一次一项消融。

## 15. 预仿真验证

8 路径、60 个月、单情景完整继承链诊断通过。企业类型份额最大守恒误差为 $2.22\times10^{-16}$，最低交换权利为 0.344，最大疲劳为 0.118。该诊断只验证结构和数值边界，不作为终局概率结果。

## 16. 正式验证设计

正式设计包含完整基线和六项一次一项消融：农业深化、工业深化、发展深化、劳动深化、统计场和分析力学。每组使用相同的 128 条路径、360 个月、6 个贸易情景、5 个介入情景和随机种子，共 30 个情景单元。全部 210 个检查点通过后才允许发布概率。

## 17. 公开数据来源

- [中国国家统计局 2025 年统计公报](https://www.stats.gov.cn/english/PressRelease/202602/t20260228_1962661.html)
- [台湾 DGBAS 2025 年第四季度国民账户](https://eng.dgbas.gov.tw/News_Content.aspx?n=4438&s=235721)
- [台湾经济部 2025 中小企业白皮书发布资料](https://www.moea.gov.tw/MNS/Populace/news/News.aspx?kind=1&menu_id=40&news_id=121491)
- [台湾农业部农业统计](https://agrstat.moa.gov.tw/moasdweb/inquire/Indicator.aspx)
- [日本农林水产省 2025 年粮食自给率](https://www.maff.go.jp/j/press/kanbo/anpo/260807.html)
- [美国 USDA 农业生产率方法](https://ers.usda.gov/data-products/agricultural-productivity-in-the-united-states/methods)
- [美国 BLS 投入产出矩阵](https://www.bls.gov/emp/data/input-output-matrix.htm)
- [美国 BEA--BLS 综合生产账户](https://www.bea.gov/data/special-topics/integrated-bea-gdp-bls-productivity-account)
