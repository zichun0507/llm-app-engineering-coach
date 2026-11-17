# llm-app-engineering-coach

一个面向工程团队的 LLM 应用开发讲解/设计 skill。

它不是知识百科，而是一套**判断与决策的框架**：先诊断提问者水平，再选讲解模板，最后按主题取对应的 reference。

---

## 目录结构

```
skills/llm-app-engineering-coach/
├── SKILL.md                          # 主入口：诊断 → 模板 → 路由 → 质量红线
├── references/
│   ├── foundations.md                # Transformer、长上下文四本账、幻觉五层根因
│   ├── prompt-engineering.md         # 七部件、位置效应、生产模板、迭代方法
│   ├── tool-and-routing.md           # 工具描述七行模板、五种路由范式、三层混合路由
│   ├── rag-and-eval.md               # 六道工序、混合检索+重排、三层评测、LLM 裁判
│   ├── post-training.md              # PPO/DPO/GRPO、LoRA/QLoRA、显存硬账
│   └── skill-authoring.md            # 三级加载、description 写法、六个标准
└── evals/
    └── evals.json                    # 6 条真实测试 prompt（含期望输出要点）
```

**为什么这么分**：SKILL.md 控制在可快速加载的体量，细节下沉到 references。模型只在需要时读那一篇，不会因为一个问题把六篇全读进来。

---

## 安装

把整个 `llm-app-engineering-coach/` 文件夹放进你的 skills 目录即可，文件夹名必须与 `name` 字段一致。常见位置：

- OpenCode：`~/.config/opencode/skills/`
- 通用 Agent Skills 目录：`~/.agents/skills/`
- Cherry Studio：`$CHERRY_STUDIO_SKILLS_DIR/`

```bash
cp -r skills/llm-app-engineering-coach ~/.config/opencode/skills/
```

无需注册，客户端重新加载 skills 后即生效。

---

## 会触发它的说法（示例）

- "什么是 LoRA"
- "PPO 的 clip 到底在裁什么"
- "上下文长了为什么又贵又答不准"
- "工具描述怎么写"
- "我们 40 个工具模型老调错"
- "RAG 答不准要不要换大模型"
- "帮我写个提示词，从合同里抽条款"
- "帮我写个 skill"

**不触发**：纯代码实现、纯业务咨询、与 LLM 应用无关的通用编程问题。

---

## 怎么验证它有效

`evals/evals.json` 里有 6 条测试 prompt。跑两组对比：

| 组别 | 做法 |
|---|---|
| with_skill | 加载本 skill，逐条执行 |
| baseline | 不加载 skill，同样 prompt 直接执行 |

对比三件事：

1. **质量**：有没有结构、有没有给"为什么"、有没有落到可执行动作
2. **token**：skill 加载本身的开销
3. **时间**：有没有因为多读 reference 反而变慢

对照组里最值得看的是第 4 条（RAG 答不准要不要换模型）——没有 skill 时，模型很可能直接建议换模型；有 skill 时应该先让你测 recall@10。

---

## 怎么改

1. **改之前先看 evals**：任何修改都要能在这 6 条上体现差异
2. **先删再加**：删掉一条规则重跑，分数不掉就永久删掉
3. **description 单独调**：它是唯一触发机制，改动后要用 10 条真实用户说法验证触发率
4. **references 可以长，SKILL.md 不能胖**：正文超过 500 行就该再分一层

---

## 设计取舍（写这个 skill 时的几个决定）

| 决定 | 原因 |
|---|---|
| 诊断分三档而不是一套讲法 | 同一个"什么是 LoRA"，新手要比喻，专家要显存账 |
| 用表格而不是段落讲对比 | 对比类知识的段落版几乎不可读 |
| 每个 reference 末尾都有"坑" | 这个领域的价值在避坑，不在定义 |
| 明确写"不吹能力上限" | LLM 预测/准确率有真实天花板，给了虚假信心就是害人 |
| 把 persona 写进 skill | 多轮对话验证过，固定风格比自由发挥稳定 |
