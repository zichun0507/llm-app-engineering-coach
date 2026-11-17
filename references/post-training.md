# 训练与微调：PPO / DPO / GRPO / LoRA

> 什么时候读这个文件：用户问 PPO、RLHF、DPO、GRPO 的原理或选型，问 LoRA/QLoRA 怎么做、该不该微调、微调该给多少参数。

---

## 第一部分：PPO

### 1. 一句话本质

**PPO = 小步快跑的强化学习。每次更新只允许策略挪一小步，挪过头就强制拉回。**

下山比喻：普通策略梯度是"哪边低迈一大步"，运气好到山腰，运气不好滚下悬崖。PPO 是"迈一步，但落点不能离预期太远，超了就当没迈"。

学术定位：**TRPO 的低成本近似**——不算 Hessian，直接把新旧策略的概率比值裁一刀。

### 2. 为什么需要它

基础策略梯度：

```
∇J(θ) = E[ ∇log π_θ(a|s) · A(s,a) ]
```

含义：**动作比平均水平好就提高概率，差就降低。** 三个病：

| 问题 | 说明 |
|---|---|
| 步长敏感 | 学习率稍大，分布整体偏移，旧样本全废，性能断崖，且回不来 |
| 样本浪费 | on-policy，更新一次就得重采，大模型采样极贵 |
| 方差大 | 回报噪声大，梯度方向乱抖 |

**重要性采样**解决样本浪费：用旧策略采的样本估新策略，乘一个修正比：

```
r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)
J(θ) = E[ r_t(θ) · A_t ]
```

于是同一批数据能更新 K 轮。但 `r_t` 无约束时，新策略会把概率推到 100 倍，方差爆炸、估计失效。**PPO 的全部工作就是给它上锁。**

### 3. 裁剪目标

```
L_CLIP(θ) = E[ min( r_t·A_t , clip(r_t, 1-ε, 1+ε)·A_t ) ]
```

ε 通常 0.1–0.2。注意是**最大化**目标，训练时取负号。

**外层 min 的意义**：悲观估计，取两者更差的那个，保证目标函数是真实目标的下界。

### 4. 四种情况（讲原理时必须讲这张表）

| 优势 A | r 的位置 | min 选中谁 | 梯度行为 |
|---|---|---|---|
| A > 0 | r < 1+ε | r·A | 正常梯度，继续推高 |
| A > 0 | r > 1+ε | (1+ε)·A 常数 | **梯度为 0，不再奖励更激进** |
| A < 0 | r > 1-ε | r·A | 正常梯度，继续压低 |
| A < 0 | r < 1-ε | (1-ε)·A 常数 | **梯度为 0，不再额外惩罚** |

**一句话规律：安全方向不裁剪，危险方向才裁剪。**

- 好动作概率下降了 → 拉回来是对的，不拦
- 好动作概率涨过头 → 拦住
- 坏动作概率下降 → 不拦
- 坏动作概率涨过头 → 拦住

**数字例子**：A=1，r=3（概率提到 3 倍），ε=0.2 → 未裁剪 3.0，裁剪后 1.2。收益从 3 降到 1.2，且梯度截断。**"我不相信你一次能变好三倍。"**

### 5. 完整损失

```
L = -L_CLIP + c1·(V(s) - V_target)² - c2·S[π]
```

| 项 | 作用 | 经验系数 |
|---|---|---|
| Policy Loss | 让策略变好 | 主体 |
| Value Loss | 训练 Critic | c1 = 0.5 |
| Entropy Bonus | 鼓励探索 | c2 = 0.01（RLHF 常设 0） |

**为什么要 Critic**：A = Q - V，需要一个网络估 V(s)，用基线降方差。

**GAE**：

```
Â_t = Σ (γλ)^l · δ_{t+l},   δ_t = r_t + γV(s_{t+1}) - V(s_t)
γ=0.99, λ=0.95
```

λ 是偏差-方差旋钮：λ→0 只看一步（方差小偏差大），λ→1 看全程（反之）。

### 6. 训练循环

```
① Rollout：用当前策略跑一批轨迹，存 (s,a,logp_old,value,reward,done)
② 算 returns 与 GAE，adv 在 batch 内标准化
③ 复用这批数据跑 K 个 epoch、多个 mini-batch
④ 清空缓冲，回到 ①
```

**③ 是 PPO 的价值所在**：同一批数据更新 K 次（3–10）。clip 保证"更新 K 次也不会跑太偏"。

### 7. 核心代码

```python
def ppo_loss(logp_new, logp_old, adv, values, returns, entropy,
             clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
    # 重要性采样比值，用对数相减再 exp 更数值稳定
    ratio = torch.exp(logp_new - logp_old)

    # 未裁剪 vs 裁剪，取 min 得到悲观下界
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(surr1, surr2).mean()

    value_loss = F.mse_loss(values, returns)
    return policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()
```

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    # values 长度 T+1（含 bootstrap），rewards 长度 T
    adv = torch.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(len(rewards))):          # 必须倒序递推
        delta = rewards[t] + gamma * values[t+1] - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values[:-1]                     # 后者是 value 的目标
```

### 8. 在 LLM 里：RLHF 的四个模型

token 级 MDP：状态 = prompt + 已生成前缀，动作 = 下一个 token，奖励只在句末。

**必须加 KL 惩罚**，否则模型会 reward hacking（越长越啰嗦）：

```
r_t = -β · log( π_θ(a_t|s_t) / π_ref(a_t|s_t) )
r_last += RM_score(x, y)
```

β 通常 0.01–0.1。

| 模型 | 角色 | 显存 |
|---|---|---|
| Actor | 被训策略 | 全量 + 梯度 + 优化器 |
| Critic | 价值估计 | 通常同尺寸 |
| Reward Model | 打分 | 推理，可量化 |
| Reference | 算 KL | 推理，冻结 |

7B 跑 PPO 要撑起两个可训练副本 + 两个推理副本，80G 卡也紧。**这就是 DPO 出现的直接原因。**

### 9. 超参与坑

| 参数 | 常用 | 坑 |
|---|---|---|
| clip_eps | 0.1–0.2 | 太小不学，太大失去保护 |
| gamma | 0.99（语言任务常 1.0） | — |
| lam | 0.95 | — |
| K epochs | 3–10 | 太多导致策略偏离旧数据 |
| lr | 1e-6 ~ 5e-5 | 比 SFT 低 1–2 个数量级 |
| ent_coef | 0–0.01 | RLHF 常 0，怕说胡话 |
| max_grad_norm | 1.0 | 必开 |

六个高频坑：

1. 优势不归一化 → 梯度尺度剧烈波动（每 batch 减均值除标准差是标配）
2. `logp_old` 记成更新后的 → ratio 恒为 1，clip 失效，退化成 PG
3. KL 系数固定 → 用自适应 β（KL 超目标 1.5 倍就涨，低于一半就降）
4. Critic 训不好 → 优势全是噪声，Actor 一起废
5. Reward hacking → 分数涨人评降，回答变长、堆关键词。必须人抽检 + 长度惩罚 + 多 RM 集成
6. 只看 reward 曲线 → 必须同时盯 KL、回答长度、人评胜率

---

## 第二部分：PPO / DPO / GRPO 怎么选

| 维度 | PPO | DPO | GRPO |
|---|---|---|---|
| 需要 Critic | ✅ | ❌ | ❌（组内均值当基线） |
| 需要 RM | ✅ | 离线预打分 | ✅（可用规则奖励） |
| 采样 | on-policy 在线 | 离线偏好对 | on-policy，一组采样 |
| 显存 | 最高 | 最低 | 中 |
| 稳定性 | 调参难 | 稳 | 中 |
| 典型用途 | 老牌 RLHF、需在线探索 | 对齐快速迭代 | 数学/代码等可验证任务 |

**选型口径**：有可验证奖励（对错可判）→ GRPO；只有偏好数据 → DPO；需要在线探索、有充足算力 → PPO。

---

## 第三部分：LoRA

### 10. 一句话本质

**冻结原模型，只在旁边挂两个小矩阵，训练时只学它们，推理时焊回去。**

比喻：四十吨的钢琴音色已经很好，但你要一点爵士味。不能拆了重造（全量微调，要 100 张卡）。LoRA 是在琴弦旁加一组几十克的配重块——音色变了，钢琴本身没动，演奏时也不会变重。

三个性质：

| 性质 | 含义 | 价值 |
|---|---|---|
| 冻结基座 | 原权重不改 | 显存省 90%，不训崩基座 |
| 低秩增量 | 只学瘦长补丁 | adapter 几十 MB |
| 可合并 | 推理时 W+BA | **零额外延迟** |

### 11. 数学

```
W' = W + ΔW,   ΔW = B·A
A ∈ R^{r×d_in},  B ∈ R^{d_out×r},  r << d
```

参数量从 `d_out·d_in` 变成 `r·(d_in+d_out)`。

d=4096, r=8：65,536 vs 16,777,216，**0.39%**。

**两个魔鬼细节**：

1. **B 初始化为 0** → ΔW=0，训练开始时模型行为与原模型完全一致。两个都随机初始化会一上来就加噪声。
2. **缩放 α/r**：`h = Wx + (α/r)·BAx`。真正的强度是 α/r 这个比值。**改了 r 不改 α，等于偷偷改了有效学习率**——新手最常见。

### 12. 为什么低秩够用

**内在维度假设**：把模型投到极低维子空间做微调，效果几乎不掉。直觉：预训练已学全语言能力，微调只是"换个说话方式"，这种改动天然低维。

**这只是经验假设，不是定理。** 任务越"新"，需要的秩越高。

### 13. 超参

| 参数 | 常用 | 说明 |
|---|---|---|
| r | 8 / 16 / 32 / 64 | 补丁容量 |
| alpha | r 或 2r | 实际生效的是 α/r |
| target_modules | 全线性层 | **最影响效果的参数** |
| dropout | 0.05–0.1 | 小数据集必须有 |
| lr | 1e-4 ~ 3e-4 | **比全量高 10 倍** |

target_modules 三档（8B, r=16）：只贴 q,v ≈ 680 万；q,k,v,o ≈ 1400 万；**全线性层 ≈ 4200 万（0.5%）**。

**经验**：数据少任务轻 → 只贴 q,v；想要效果 → **全线性层**。FFN 承载大量"知识/风格"，只贴 attention 常常欠拟合。

### 14. 代码

```python
from peft import LoraConfig, get_peft_model, TaskType

cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    bias="none",
)
model = get_peft_model(base_model, cfg)
model.print_trainable_parameters()   # 约 0.5%

model.save_pretrained("./my-lora")          # 只存 adapter，几十 MB
model = model.merge_and_unload()            # 合并：零额外延迟
```

**生产基本用合并**——合并后就是普通模型，vLLM / TensorRT-LLM 直接吃。只有"一个基座服务多个租户多个 adapter"时才用动态加载（vLLM multi-LoRA）。

### 15. 一笔硬账（8B）

| 项 | 全量 | LoRA r=16 |
|---|---|---|
| 基座 | 16 GB | 16 GB（冻结） |
| 梯度 | 16 GB | ~0.08 GB |
| 优化器 | 64 GB | ~0.3 GB |
| 激活（梯度检查点） | ~20 GB | ~8 GB |
| **合计** | **约 110 GB** | **约 24 GB** |
| 硬件 | 2×A100 80G | **1×4090 24G** |
| 产出 | 16 GB | **约 80 MB** |

### 16. QLoRA

三件套：4-bit NF4 量化基座（16G→4G）+ 双重量化 + 分页优化器。

```python
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

效果：65B 单张 48G 可微调；8B 约 12G 可微调。**个人/小团队首选。**

### 17. 变体速查

| 变体 | 一句话 |
|---|---|
| DoRA | 方向/幅度分开适配，小 r 更好 |
| LoRA+ | A、B 用不同学习率，收敛更快 |
| rsLoRA | 缩放改 α/√r，大 r 更稳 |
| AdaLoRA | 自动分配每层的 r |
| PiSSA | 用 SVD 初始化，收敛更快 |
| VeRA | A、B 共享，只学缩放向量，极致省 |

**建议**：先跑 vanilla LoRA 基线，再考虑换变体。

### 18. 七个坑

1. 只贴 q,v → loss 降不下去，换全线性层
2. 改 r 不改 α → 有效学习率变了
3. lr 沿用 1e-5 → 几乎不学，提到 1e-4~3e-4
4. merge 用 fp16 → 精度损失，用 bf16
5. 小数据集不加 dropout → 验证集崩
6. 数据全是目标任务 → **照样会灾难性遗忘**，混 10–20% 通用数据
7. adapter 不版本化 → 线上效果变了查不到原因

### 19. 关键判断：LoRA 还是 RAG

> **LoRA 擅长"改造"，不擅长"灌输"。**

- 改语气、改格式、适配任务、领域术语对齐 → **LoRA**
- 塞进几十万字手册、让它知道最新政策 → **RAG**
- 学全新语言、换模态 → 继续预训练 / 全量微调

**一句话：微调教"怎么说话"，RAG 给"说什么"。** 想更新知识却选微调，是性价比最差的做法。
