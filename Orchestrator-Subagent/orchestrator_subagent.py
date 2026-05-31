import asyncio
import json

from pydantic import Field, BaseModel

from camel.agents import ChatAgent
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.responses import ChatAgentResponse

from model_setup import get_model_backend


class SecurityReport(BaseModel):
    issues: list[str] = Field(..., description="发现的安全漏洞列表；无问题时填空列表。")
    severity: str = Field(..., description="整体安全风险级别：low / medium / high / critical。")


class CoverageReport(BaseModel):
    missing_tests: list[str] = Field(..., description="缺少测试的功能点列表；全部覆盖时填空列表。")
    coverage_score: float = Field(..., ge=0.0, le=1.0, description="测试覆盖率估计分数（0.0–1.0）。")


class StyleReport(BaseModel):
    violations: list[str] = Field(..., description="代码风格违规列表；无问题时填空列表。")
    style_score: float = Field(..., ge=0.0, le=1.0, description="代码风格评分（0.0–1.0）。")


class FinalReview(BaseModel):
    summary: str = Field(..., description="综合 Code Review 摘要。")
    action_items: list[str] = Field(..., description="需要修复的行动项，按优先级排序。")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="综合质量评分（0.0–1.0）。")
    approved: bool = Field(..., description="是否建议通过此 PR。")


class OrchestratorSubagentResult(BaseModel):
    security: SecurityReport
    coverage: CoverageReport
    style: StyleReport
    final_review: FinalReview


def create_security_agent(model_backend: BaseModelBackend) -> ChatAgent:
    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="SecurityReviewer",
            content=(
                "你是一个专注于安全漏洞检测的代码审查专家。\n"
                "分析代码中的安全问题（如 SQL 注入、越权访问、敏感信息泄露等）。\n"
                "返回 JSON，字段说明：\n"
                "- issues: 发现的安全漏洞列表；无问题时填空列表。\n"
                "- severity: 整体安全风险级别：low / medium / high / critical。"
            ),
        ),
        model=model_backend,
    )


def create_coverage_agent(model_backend: BaseModelBackend) -> ChatAgent:
    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="CoverageReviewer",
            content=(
                "你是一个专注于测试覆盖率分析的代码审查专家。\n"
                "识别缺少单元测试或集成测试的功能点。\n"
                "返回 JSON，字段说明：\n"
                "- missing_tests: 缺少测试的功能点列表；全部覆盖时填空列表。\n"
                "- coverage_score: 测试覆盖率估计分数（0.0–1.0）。"
            ),
        ),
        model=model_backend,
    )


def create_style_agent(model_backend: BaseModelBackend) -> ChatAgent:
    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="StyleReviewer",
            content=(
                "你是一个专注于代码风格检查的代码审查专家。\n"
                "检查命名规范、类型注解、注释质量、代码结构等风格问题。\n"
                "返回 JSON，字段说明：\n"
                "- violations: 代码风格违规列表；无问题时填空列表。\n"
                "- style_score: 代码风格评分（0.0–1.0）。"
            ),
        ),
        model=model_backend,
    )


def create_orchestrator(model_backend: BaseModelBackend) -> ChatAgent:
    return ChatAgent(
        system_message=BaseMessage.make_assistant_message(
            role_name="Orchestrator",
            content=(
                "你是代码审查编排者（Orchestrator）。\n"
                "收到安全、覆盖率、风格三份子报告后，综合生成最终 Code Review 报告。\n"
                "返回 JSON，字段说明：\n"
                "- summary: 综合 Code Review 摘要。\n"
                "- action_items: 需要修复的行动项，按优先级排序。\n"
                "- overall_score: 综合质量评分（0.0–1.0）。\n"
                "- approved: 是否建议通过此 PR（bool）。"
            ),
        ),
        model=model_backend,
    )


class OrchestratorSystem:
    def __init__(
        self,
        orchestrator: ChatAgent,
        security_agent: ChatAgent,
        coverage_agent: ChatAgent,
        style_agent: ChatAgent,
    ) -> None:
        self.orchestrator = orchestrator
        self.security_agent = security_agent
        self.coverage_agent = coverage_agent
        self.style_agent = style_agent


def create_orchestrator_system() -> OrchestratorSystem:
    model_backend = get_model_backend()
    return OrchestratorSystem(
        orchestrator=create_orchestrator(model_backend),
        security_agent=create_security_agent(model_backend),
        coverage_agent=create_coverage_agent(model_backend),
        style_agent=create_style_agent(model_backend),
    )


def _ensure_not_stream(response: object) -> ChatAgentResponse:
    if not isinstance(response, ChatAgentResponse):
        raise ValueError("模型的 stream 参数应设置为 false")
    return response


def _parse_security(response) -> SecurityReport:
    if response.msg is None:
        raise ValueError("SecurityReviewer 响应消息为空")
    return SecurityReport(**json.loads(response.msg.content))


def _parse_coverage(response) -> CoverageReport:
    if response.msg is None:
        raise ValueError("CoverageReviewer 响应消息为空")
    return CoverageReport(**json.loads(response.msg.content))


def _parse_style(response) -> StyleReport:
    if response.msg is None:
        raise ValueError("StyleReviewer 响应消息为空")
    return StyleReport(**json.loads(response.msg.content))


def _parse_final(response) -> FinalReview:
    if response.msg is None:
        raise ValueError("Orchestrator 响应消息为空")
    return FinalReview(**json.loads(response.msg.content))


def _build_synthesis_content(
    security: SecurityReport, coverage: CoverageReport, style: StyleReport
) -> str:
    return (
        "请综合以下三份报告，生成最终 Code Review：\n\n"
        f"【安全报告】\n"
        f"漏洞：{security.issues}\n"
        f"风险级别：{security.severity}\n\n"
        f"【覆盖率报告】\n"
        f"缺失测试：{coverage.missing_tests}\n"
        f"覆盖率评分：{coverage.coverage_score:.2f}\n\n"
        f"【风格报告】\n"
        f"违规项：{style.violations}\n"
        f"风格评分：{style.style_score:.2f}"
    )


def orchestrator_subagent_review(pr_code: str) -> OrchestratorSubagentResult:
    """
    Orchestrator-Subagent 模式（同步版本）
    ----------------------------------------
    Orchestrator 依次将 PR 代码发给三个专业子 Agent（安全、覆盖率、风格），
    收集各自结构化报告后综合生成最终 Code Review。
    """
    system = create_orchestrator_system()

    security = _parse_security(
        system.security_agent.step(
            BaseMessage.make_user_message(
                role_name="User", content=f"请检查以下代码的安全漏洞：\n\n{pr_code}"
            ),
            response_format=SecurityReport,
        )
    )
    print(f"🔒 安全审查完成：severity={security.severity}, issues={len(security.issues)}")

    coverage = _parse_coverage(
        system.coverage_agent.step(
            BaseMessage.make_user_message(
                role_name="User", content=f"请分析以下代码的测试覆盖情况：\n\n{pr_code}"
            ),
            response_format=CoverageReport,
        )
    )
    print(f"🧪 覆盖率审查完成：score={coverage.coverage_score:.2f}, missing={len(coverage.missing_tests)}")

    style = _parse_style(
        system.style_agent.step(
            BaseMessage.make_user_message(
                role_name="User", content=f"请检查以下代码的风格问题：\n\n{pr_code}"
            ),
            response_format=StyleReport,
        )
    )
    print(f"🎨 风格审查完成：score={style.style_score:.2f}, violations={len(style.violations)}")

    final = _parse_final(
        system.orchestrator.step(
            BaseMessage.make_user_message(
                role_name="User",
                content=_build_synthesis_content(security, coverage, style),
            ),
            response_format=FinalReview,
        )
    )
    print(f"📋 综合审查完成：overall_score={final.overall_score:.2f}, approved={final.approved}")

    return OrchestratorSubagentResult(
        security=security,
        coverage=coverage,
        style=style,
        final_review=final,
    )


async def aorchestrator_subagent_review(pr_code: str) -> OrchestratorSubagentResult:
    """
    Orchestrator-Subagent 模式（异步并行版本）
    -------------------------------------------
    三个子 Agent 并发执行，Orchestrator 收到全部报告后综合生成最终 Code Review。
    并发执行相比同步版本可显著降低总耗时。
    """
    system = create_orchestrator_system()

    security_resp, coverage_resp, style_resp = await asyncio.gather(
        system.security_agent.astep(
            BaseMessage.make_user_message(
                role_name="User", content=f"请检查以下代码的安全漏洞：\n\n{pr_code}"
            ),
            response_format=SecurityReport,
        ),
        system.coverage_agent.astep(
            BaseMessage.make_user_message(
                role_name="User", content=f"请分析以下代码的测试覆盖情况：\n\n{pr_code}"
            ),
            response_format=CoverageReport,
        ),
        system.style_agent.astep(
            BaseMessage.make_user_message(
                role_name="User", content=f"请检查以下代码的风格问题：\n\n{pr_code}"
            ),
            response_format=StyleReport,
        ),
    )

    security = _parse_security(_ensure_not_stream(security_resp))
    coverage = _parse_coverage(_ensure_not_stream(coverage_resp))
    style = _parse_style(_ensure_not_stream(style_resp))
    print(
        f"🔒 security={security.severity} | "
        f"🧪 coverage={coverage.coverage_score:.2f} | "
        f"🎨 style={style.style_score:.2f}"
    )

    final = _parse_final(
        _ensure_not_stream(
            await system.orchestrator.astep(
                BaseMessage.make_user_message(
                    role_name="User",
                    content=_build_synthesis_content(security, coverage, style),
                ),
                response_format=FinalReview,
            )
        )
    )
    print(f"📋 综合审查完成：overall_score={final.overall_score:.2f}, approved={final.approved}")

    return OrchestratorSubagentResult(
        security=security,
        coverage=coverage,
        style=style,
        final_review=final,
    )


if __name__ == "__main__":
    sample_pr_code = """\
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)

def process_data(data):
    result = data * 2
    return result
"""

    print("=== 同步版本（顺序执行子 Agent）===")
    result = orchestrator_subagent_review(sample_pr_code)
    print(f"\n最终摘要：{result.final_review.summary}")
    print(f"行动项：{result.final_review.action_items}")

    print("\n=== 异步并行版本（并发执行子 Agent）===")
    result_async = asyncio.run(aorchestrator_subagent_review(sample_pr_code))
    print(f"\n最终摘要：{result_async.final_review.summary}")
    print(f"行动项：{result_async.final_review.action_items}")
