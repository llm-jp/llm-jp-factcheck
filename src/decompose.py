import json
from logging import getLogger
from textwrap import dedent
from typing import Optional

from utils import client

logger = getLogger(__name__)

SYSTEM_PROMPT = dedent(
    """\
    You are provided with a document (or an utterance) with context.
    Your task is to decompose the document into atomic claims.
    Each claim represents one fact.
    Every claim should be context-independent, i.e., it should be understandable alone without the context.
    For example, pronouns should be replaced with the actual names.

    Example:
        Input:
            Context: What do you know about Mary?
            Document: She likes playing piano and doesn't like cookies.
        Output:
            Claims:
                - Mary likes playing piano.
                - Mary doesn't like cookies.

    Example:
        Input:
            Context: アメリカの初代大統領は誰ですか？
            Document: ジョージ・ワシントンです。
        Output:
            Claims:
                - アメリカの初代大統領はジョージ・ワシントンです。
    """
)

USER_PROMPT = dedent(
    """\
    Decompose the following document into atomic claims:
    ---
    [Context]
    {context}
    ---
    [Document]
    {document}
    """
)

TOOL = {
    "type": "function",
    "function": {
        "name": "createClaimList",
        "description": "Create a list of atomic claims, each representing one context-independent fact.",
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "description": "A list of claims.",
                    "items": {"type": "string"},
                },
            },
            "required": ["claims"],
        },
    },
}

TOOL_CHOICE = {"type": "function", "function": {"name": "createClaimList"}}


def decompose_document_into_claims(document: str, model: str, context: Optional[str] = None) -> list[str]:
    """Decompose a document into a list of claims.

    Args:
        document (str): A document.
        model (str): A model.
        context (str, optional): A context. Defaults to None.

    Returns:
        list[str]: A list of claims.
    """
    if document.strip() == "":
        return []

    ret = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(document=document, context=context)},
        ],
        tools=[TOOL],
        tool_choice=TOOL_CHOICE,
    )
    for tool_call in ret.choices[0].message.tool_calls:
        if tool_call.function.name == "createClaimList":
            try:
                claims = json.loads(tool_call.function.arguments).get("claims", [])
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON: {tool_call.function.arguments}")
            if not isinstance(claims, list) or not all(isinstance(claim, str) for claim in claims):
                raise ValueError(f"Invalid claims: {tool_call.function.arguments}")
            claims = [claim.strip() for claim in claims if claim.strip()]
            return claims
    raise ValueError("Failed to extract claims")


if __name__ == "__main__":
    # model = "gpt-4-1106-preview"
    # model = "gpt-35-turbo-1106"
    model = "gpt-4-0613"

    documents = [
        dedent(
            """\
            「地球環境を守る」というのは漠然としすぎたテーマで、人によってそれぞれ異なるものですが、できることをいくつか紹介します。
            まず自分自身が意識してできることとして、
            １，毎日出るゴミをゴミ箱に入れる
            ２，不要になった紙を分別する
            ３，マイバックを使う
            ４，自然界を大切にする
            などが挙げられます。
            """
        ),
        dedent(
            """\
            ダランベールの収束判定法とは数学の定理の証明の方法の1つです。ダランベールの収束判定法では，定理の主張をある不定数に対して，それがある限り「確かさをもって」証明します。
            ダランベールの定理は，複素変数の実数値関数の収束を議論するときに使います。定理の主張は、実数値関数がある複素数の絶対値より小さくなることはないというものです。
            """
        ),
        dedent(
            """\
            地震はナマズや他の動物によって引き起こされるものではない。地震は地球の地殻プレートの動きによって引き起こされる自然現象です。
            地震が発生する最も一般的な原因のひとつは、地球のマントルの上部にある地殻プレートの衝突だ。地殻プレートが互いに衝突すると、やがて圧力が高まり、マントル内の地殻プレートが分離して互いに動くようになる。この動きが地震波を引き起こし、地表で地震を引き起こす。
            """
        ),
        dedent(
            """\
            天皇誕生日は現在、2月23日です。日付は以下のように変遷しています。
            1947年2月23日：昭和天皇の誕生日に合わせて制定されました。当初の名称は「天皇誕生日」で、祝日ではなく天長節と同じ休日扱いでした。
            """
        ),
        dedent(
            """\
            DXとは、デジタルトランスフォーメーションのことです。デジタル技術によってビジネスや業界の構造を変革することです。デジタルトランスフォーメーションの目標は、効率を高め、顧客により良いサービスを提供することです。よりインテリジェントなシステム、よりクラウドベースのモデル、より接続された顧客エンゲージメントプラットフォームを活用することで、企業はビジネスを改善し、市場でより競争力を高めることができる。DXのメリットは、効率性の向上、競争上の優位性の構築、コスト削減、カスタマーエクスペリエンスの改善である。
            """
        ),
        dedent(
            """\
            There are a few reasons people easily procrastinate. Some of the most common reasons include:
            Poor management of time: People often assign too high a priority to tasks that are perceived as important, leading them to neglect tasks that are perceived as less important.
            """
        ),
        dedent(
            """\
            Python is a popular versatile and general-purpose programming language that supports multiple programming paradigms, including structured, procedural, functional, object-oriented, and data-driven programming. It is also relatively easy to learn and has a broad supportive ecosystem of libraries and tools, making it a robust and flexible language for a wide range of applications.
            """
        ),
        dedent(
            """\
            A quantum computer is a computer that simulates the behavior of quantum systems. It does so by using a special technique called quantum computing to perform calculations.
            Quantum computers can perform certain calculations that are beyond the reach of classical computers. In fact, they can perform certain complex calculations faster than any known classical computer. This fact has led to significant interest in developing quantum computers as a promising tool for a variety of applications, from cryptography to materials science.
            """
        ),
        dedent(
            """\
            Art is something that many people struggle with. It can take a lifetime or more to master. It's also subjective. Some people like to see art as a creative expression others see it as an intellectual exercise. Art can even be judged by art critics who can have an opinion on the artistic quality of the art work. Art is hard because it requires practice and a large amount of discipline to be able to produce quality art work.
            """
        ),
        dedent(
            """\
            Here are 3 reasons why you might consider visiting the United States:
            1. Access to unique and interesting experiences.
            2. The US has some of the most creative restaurants.
            3. Growing tourism industry.
            """
        )
    ]

    for document in documents:
        try:
            for i, claim in enumerate(decompose_document_into_claims(document, model=model)):
                print(f"- {i}. {claim}")
            print("---")
        except Exception as e:
            print(f"Error: {e}")
            print("---")
