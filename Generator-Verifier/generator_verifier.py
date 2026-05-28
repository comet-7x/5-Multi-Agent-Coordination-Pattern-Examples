import asyncio
import json

from pydantic import Field, BaseModel

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.responses import ChatAgentResponse

from model_setup import get_model_backend


class GeneratorOutput(BaseModel):
    output: str = Field(..., description="生成的完整 Python 代码字符串。")


class VerifierOutput(BaseModel):
    feedback: str = Field(..., description="代码审查的详细反馈；完全通过时写 PASS。")
    score: float = Field(
        ..., ge=0.0, le=1.0, description="代码质量评分，范围 0.0 到 1.0，完全通过为 1.0。"
    )


class GeneratorVerifierOutput(BaseModel):
    generator_outputs: list[GeneratorOutput] = Field(..., description="每轮生成的代码历史。")
    verifier_outputs: list[VerifierOutput] = Field(..., description="每轮验证的反馈历史。")

    @property
    def final_code(self) -> str:
        best_idx, _ = max(enumerate(self.verifier_outputs), key=lambda t: t[1].score)
        return self.generator_outputs[best_idx].output


class GeneratorVerifier:
    def __init__(self, generator: ChatAgent, verifier: ChatAgent) -> None:
        self.generator = generator
        self.verifier = verifier

    def generate(self, *args, **kwargs):
        return self.generator.step(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        return await self.generator.astep(*args, **kwargs)

    def verify(self, *args, **kwargs):
        return self.verifier.step(*args, **kwargs)

    async def averify(self, *args, **kwargs):
        return await self.verifier.astep(*args, **kwargs)


def create_generator(model_backend: BaseModelBackend) -> ChatAgent:
    system_message = BaseMessage.make_assistant_message(
        role_name="Generator",
        content=(
            "你是一个代码生成专家，根据需求生成高质量 Python 代码。\n"
            "返回 JSON，字段说明：\n"
            "- output: 完整的 Python 代码字符串。"
        ),
    )
    return ChatAgent(system_message=system_message, model=model_backend)


def create_verifier(model_backend: BaseModelBackend) -> ChatAgent:
    system_message = BaseMessage.make_assistant_message(
        role_name="Verifier",
        content=(
            "你是一个严格的代码审查专家，评估代码是否满足以下标准：\n"
            "1. 无语法错误\n"
            "2. 有完整的错误处理\n"
            "3. 有类型注解\n"
            "返回 JSON，字段说明：\n"
            "- feedback: 详细说明所有问题；如果完全通过，写 PASS。\n"
            "- score: 质量评分（0.0-1.0），完全通过为 1.0，每项缺陷扣分。"
        ),
    )
    return ChatAgent(system_message=system_message, model=model_backend)


def create_generator_verifier() -> GeneratorVerifier:
    model_backend = get_model_backend()
    return GeneratorVerifier(
        create_generator(model_backend),
        create_verifier(model_backend),
    )


_PASS_SCORE_THRESHOLD = 0.9


def _parse_generator(response) -> GeneratorOutput:
    if response.msg is None:
        raise ValueError("Generator 生成的响应消息为空")
    return GeneratorOutput(**json.loads(response.msg.content))


def _parse_verifier(response) -> VerifierOutput:
    if response.msg is None:
        raise ValueError("Verifier 生成的响应消息为空")
    return VerifierOutput(**json.loads(response.msg.content))


def _ensure_not_stream(response: object) -> ChatAgentResponse:
    if not isinstance(response, ChatAgentResponse):
        raise ValueError("模型的 stream 参数应设置为 false")
    return response


def generator_verifier_loop(task: str, max_iterations: int = 3) -> GeneratorVerifierOutput:
    """
    Generator-Verifier 模式
    -----------------------
    Generator 生成内容 → Verifier 打分并给出反馈 → 未通过则带着反馈重新生成
    循环直到通过验证或达到最大迭代次数。
    """
    generator_outputs: list[GeneratorOutput] = []
    verifier_outputs: list[VerifierOutput] = []
    gv = create_generator_verifier()

    user_msg = BaseMessage.make_user_message(role_name="User", content=task)
    current_output = _parse_generator(gv.generate(user_msg, response_format=GeneratorOutput))
    generator_outputs.append(current_output)

    passed = False
    for i in range(max_iterations):
        verify_msg = BaseMessage.make_user_message(
            role_name="User",
            content=f"请评审以下代码：\n\n{current_output.output}",
        )
        current_verify = _parse_verifier(gv.verify(verify_msg, response_format=VerifierOutput))
        verifier_outputs.append(current_verify)

        if current_verify.score >= _PASS_SCORE_THRESHOLD:
            print(f"✅ 第 {i + 1} 轮通过验证 (score={current_verify.score:.2f})")
            passed = True
            break

        print(
            f"❌ 第 {i + 1} 轮未通过 (score={current_verify.score:.2f})，"
            f"反馈：{current_verify.feedback[:100]}..."
        )
        if i < max_iterations - 1:
            revise_msg = BaseMessage.make_user_message(
                role_name="User",
                content=(
                    f"根据以下反馈修改代码：\n{current_verify.feedback}"
                    f"\n\n原代码：\n{current_output.output}"
                ),
            )
            current_output = _parse_generator(
                gv.generate(revise_msg, response_format=GeneratorOutput)
            )
            generator_outputs.append(current_output)

    if not passed:
        print("⚠️ 达到最大迭代次数，返回最佳尝试")

    return GeneratorVerifierOutput(
        generator_outputs=generator_outputs,
        verifier_outputs=verifier_outputs,
    )


async def agenerator_verifier_loop(task: str, max_iterations: int = 3) -> GeneratorVerifierOutput:
    generator_outputs: list[GeneratorOutput] = []
    verifier_outputs: list[VerifierOutput] = []
    gv = create_generator_verifier()

    user_msg = BaseMessage.make_user_message(role_name="User", content=task)
    current_output = _parse_generator(
        _ensure_not_stream(await gv.agenerate(user_msg, response_format=GeneratorOutput))
    )
    generator_outputs.append(current_output)

    passed = False
    for i in range(max_iterations):
        verify_msg = BaseMessage.make_user_message(
            role_name="User",
            content=f"请评审以下代码：\n\n{current_output.output}",
        )
        current_verify = _parse_verifier(
            _ensure_not_stream(await gv.averify(verify_msg, response_format=VerifierOutput))
        )
        verifier_outputs.append(current_verify)

        if current_verify.score >= _PASS_SCORE_THRESHOLD:
            print(f"✅ 第 {i + 1} 轮通过验证 (score={current_verify.score:.2f})")
            passed = True
            break

        print(
            f"❌ 第 {i + 1} 轮未通过 (score={current_verify.score:.2f})，"
            f"反馈：{current_verify.feedback[:100]}..."
        )
        if i < max_iterations - 1:
            revise_msg = BaseMessage.make_user_message(
                role_name="User",
                content=(
                    f"根据以下反馈修改代码：\n{current_verify.feedback}"
                    f"\n\n原代码：\n{current_output.output}"
                ),
            )
            current_output = _parse_generator(
                _ensure_not_stream(await gv.agenerate(revise_msg, response_format=GeneratorOutput))
            )
            generator_outputs.append(current_output)

    if not passed:
        print("⚠️ 达到最大迭代次数，返回最佳尝试")

    return GeneratorVerifierOutput(
        generator_outputs=generator_outputs,
        verifier_outputs=verifier_outputs,
    )


if __name__ == "__main__":
    task = "请生成一个计算两个数的乘积的函数，并添加错误处理。"

    result = generator_verifier_loop(task)
    print(result.final_code)

    result_async = asyncio.run(agenerator_verifier_loop(task))
    print(result_async.final_code)
