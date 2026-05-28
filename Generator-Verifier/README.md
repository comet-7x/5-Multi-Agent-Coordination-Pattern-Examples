# Generator-Verifier 模式

## 什么是 Generator-Verifier？

Generator-Verifier 是一种多智能体协调模式，由两个角色组成：

- **Generator**：负责生成内容（代码、文章、方案等）
- **Verifier**：负责审查内容，给出评分和反馈

两者形成一个**反馈循环**：Generator 生成 → Verifier 审查 → 未通过则带着反馈重新生成，直到通过或达到最大迭代次数。

```
         ┌─────────────────────────────────────────────┐
         │                                             │
         ▼                                             │ 未通过，带反馈重新生成
      Generator  ──生成内容──▶  Verifier  ──打分──▶  是否通过？
                                                       │
                                                       │ 通过
                                                       ▼
                                                    返回结果
```

---

## 为什么需要这个模式？

单个 LLM 一次生成的结果质量参差不齐。Generator-Verifier 通过**职责分离**解决这个问题：

| 问题 | 解决方式 |
|---|---|
| 模型一次生成结果不稳定 | 多轮迭代，逐步逼近高质量结果 |
| 模型难以自我批评 | 用独立的 Verifier 做客观评审 |
| 不知道何时停止 | 通过评分阈值（score ≥ 0.9）量化"通过"标准 |

---

## 文件结构

```
Generator-Verifier/
├── config.yaml            # 模型参数配置（温度、流式等）
├── model_setup.py         # 读取配置，构建模型后端
└── generator_verifier.py  # 核心模式逻辑
```

`.env`（放项目根目录，不提交 git）：
```
MODEL_URL=http://your-model-endpoint/v1
MODEL_NAME=your-model-name
MODEL_API_KEY=your-api-key
```

---

## 快速开始

**安装依赖**
```bash
uv pip install -e ".[dev]"
```

**配置环境**
```bash
# 在项目根目录创建 .env
cp .env.example .env
# 填写 MODEL_URL、MODEL_NAME、MODEL_API_KEY
```

**运行示例**
```bash
cd Generator-Verifier
python generator_verifier.py
```

---

## 核心代码讲解

### 1. 结构化输出（Pydantic Models）

```python
class GeneratorOutput(BaseModel):
    output: str  # 生成的代码

class VerifierOutput(BaseModel):
    feedback: str  # 详细反馈，通过时写 PASS
    score: float   # 质量评分 0.0~1.0
```

使用 Pydantic 约束输出格式，确保 LLM 返回的 JSON 可以被可靠解析，而不是自由文本。

---

### 2. GeneratorVerifier 类

```python
class GeneratorVerifier:
    def __init__(self, generator: ChatAgent, verifier: ChatAgent) -> None:
        self.generator = generator
        self.verifier = verifier
```

持有两个 `ChatAgent`，分别扮演 Generator 和 Verifier 角色。
提供同步（`generate` / `verify`）和异步（`agenerate` / `averify`）两套接口。

---

### 3. 核心循环

```python
def generator_verifier_loop(task, max_iterations=3):
    # 第一步：初始生成
    current_output = generate(task)

    for i in range(max_iterations):
        # 第二步：验证
        current_verify = verify(current_output)

        if current_verify.score >= 0.9:   # 通过
            break

        if i < max_iterations - 1:        # 还有剩余次数
            # 第三步：带反馈重新生成
            current_output = generate(task + feedback)

    return 历史记录中分数最高的结果
```

循环结束后，`final_code` 返回的是**所有轮次中评分最高**的代码，而不是最后一轮：

```python
@property
def final_code(self) -> str:
    best_idx, _ = max(enumerate(self.verifier_outputs), key=lambda t: t[1].score)
    return self.generator_outputs[best_idx].output
```

---

### 4. 历史记录

`GeneratorVerifierOutput` 保存了完整的迭代历史：

```python
result = generator_verifier_loop(task)

# 查看每轮代码
for i, gen in enumerate(result.generator_outputs):
    print(f"第 {i+1} 轮代码：\n{gen.output}")

# 查看每轮评分和反馈
for i, ver in enumerate(result.verifier_outputs):
    print(f"第 {i+1} 轮评分：{ver.score}，反馈：{ver.feedback}")

# 取最优结果
print(result.final_code)
```

---

## 配置说明

**`config.yaml`** — 模型推理参数（可根据模型调整）：

```yaml
model:
  temperature: 0.1   # 低温度 = 更确定性的输出，适合代码生成
  stream: false      # 必须关闭，否则无法使用结构化输出
  extra_body:        # 模型特定参数，不同模型可能不需要
    chat_template_kwargs:
      enable_thinking: false
```

---

## 适用场景

Generator-Verifier 模式适合任何有**明确质量标准**的生成任务：

- 代码生成（本示例）
- 数学解题（Verifier 验证步骤正确性）
- 文案写作（Verifier 检查语气、字数、关键词）
- 数据抽取（Verifier 校验格式和完整性）
