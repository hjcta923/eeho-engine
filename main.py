import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import httpx
import re
import asyncio
import json
import traceback as tb
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import storage
from pinecone import Pinecone

app = FastAPI(title="EEHO AI Engine", version="1.1")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

# ============================================================
# 설정 (환경변수에서 로드)
# ============================================================
OC = os.environ.get("LAW_API_OC", "")
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "")

# Pinecone 초기화 (지연 로딩 - 서버 시작 시 외부 연결 실패 방지)
pc = None
pinecone_index = None


def get_pinecone_index():
    global pc, pinecone_index
    if pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return pinecone_index


# ============================================================
# [보강 ①] 판례 5단계 파싱 스키마 정의 (Pydantic)
# ─ 특허 명세서 대응: 각 필드의 추출 규칙, 필수/선택,
#   최소 길이, 신뢰도 점수를 정형화
# ============================================================

class ParsedField(BaseModel):
    """개별 파싱 필드의 구조"""
    content: str = Field(..., min_length=10, description="추출된 내용 (최소 10자)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="해당 필드의 파싱 신뢰도 (0.0~1.0)")

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        stripped = v.strip()
        if len(stripped) < 2:
            raise ValueError("필드 내용이 너무 짧습니다")
        return stripped


class StructuredPrecedent(BaseModel):
    """
    판례 5단계 구조화 스키마
    ─────────────────────────────────────────
    [특허 기술 명세 대응]
    - 각 필드는 판례 원문에서 추출해야 할 '하위 필수항목'이 정의됨
    - 판례 본문에 해당 내용이 부재할 경우 content에 "해당없음" 기재 필수
    - confidence는 LLM이 자체 판단하여 부여 (추출 근거의 명확성 기준)
    """

    사실관계: ParsedField = Field(
        ...,
        description=(
            "당사자 관계(납세자-과세관청), 거래 대상 자산의 유형 및 소재지, "
            "거래일자(취득/양도), 쟁점이 되는 핵심 행위(합가, 증여, 양도 등)를 "
            "반드시 포함하여 추출. 판례 본문에 명시되지 않은 항목은 '미기재'로 표시."
        )
    )

    납세자주장: ParsedField = Field(
        ...,
        description=(
            "납세자가 비과세/감면/경정청구 등을 주장한 근거 조문, "
            "본인이 충족한다고 주장하는 요건, 그 논리적 근거를 추출. "
            "납세자 주장이 명시되지 않은 약식 판례는 '해당없음'으로 기재."
        )
    )

    과세관청주장: ParsedField = Field(
        ...,
        description=(
            "과세관청(국세청/세무서)이 과세처분의 근거로 제시한 조문, "
            "납세자 요건 미충족 사유, 사실관계에 대한 과세관청의 해석을 추출. "
            "과세관청 주장이 명시되지 않은 경우 '해당없음'으로 기재."
        )
    )

    판단근거: ParsedField = Field(
        ...,
        description=(
            "심판원/법원이 최종 판단을 내린 법리적 근거. "
            "인용/기각 결론, 그 결론에 이른 핵심 논리(요건 충족/미충족 판단), "
            "선례 참조 여부를 포함하여 추출."
        )
    )

    관련법령: ParsedField = Field(
        ...,
        description=(
            "판례에서 직접 인용되거나 적용된 법령 조문을 '법령명 + 조항' "
            "형태로 추출. (예: '소득세법 제89조 제1항 제3호', "
            "'소득세법 시행령 제155조 제4항') "
            "복수 조문이 있으면 쉼표로 구분."
        )
    )

    종합신뢰도: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "5개 필드 전체에 대한 종합 파싱 신뢰도. "
            "판례 본문이 충분히 상세하고 모든 필드가 명확히 추출 가능하면 0.8 이상, "
            "약식 판례이거나 2개 이상 필드가 '해당없음'이면 0.5 이하로 부여."
        )
    )

    @field_validator("종합신뢰도")
    @classmethod
    def validate_overall_confidence(cls, v):
        return round(v, 2)


# ============================================================
# [보강 ①] 구조화된 파싱 프롬프트
# ─ 특허 명세서 대응: 추출 규칙/프롬프트/후처리를 구체적으로 기술
# ============================================================

STRUCTURED_PARSING_PROMPT = """당신은 대한민국 세법 판례 분석 전문가입니다.
아래 판례 원문을 읽고, 정확히 5개 필드로 구조화하세요.

═══ 필드별 추출 규칙 ═══

1. 사실관계 (필수 포함 항목):
   - 당사자 관계 (납세자와 과세관청의 관계)
   - 거래 대상 자산의 유형 (아파트, 토지, 분양권 등) 및 소재지
   - 핵심 일자 (취득일, 양도일, 합가일, 상속개시일 등)
   - 쟁점 행위 (양도, 합가, 증여, 상속 등 과세 원인이 된 행위)
   ※ 판례 본문에 명시되지 않은 항목은 '미기재'로 표시

2. 납세자주장 (필수 포함 항목):
   - 주장의 근거 조문 (예: 소득세법 시행령 제155조)
   - 본인이 충족한다고 주장하는 구체적 요건
   - 주장의 논리적 근거
   ※ 약식 판례로 납세자 주장이 없으면 content에 "해당없음 - 판례 본문에 납세자 주장 관련 기술 없음" 기재

3. 과세관청주장 (필수 포함 항목):
   - 과세처분의 근거 조문
   - 납세자가 미충족한다고 보는 구체적 요건
   - 과세관청의 사실관계 해석
   ※ 과세관청 주장이 없으면 content에 "해당없음 - 판례 본문에 과세관청 주장 관련 기술 없음" 기재

4. 판단근거 (필수 포함 항목):
   - 인용/기각 결론
   - 결론에 이른 핵심 법리 (어떤 요건이 충족/미충족인지)
   - 선례 참조 여부

5. 관련법령:
   - 판례에서 직접 인용된 법령을 '법령명 + 조항' 형태로 나열
   - 복수 조문은 쉼표로 구분
   - 예시: "소득세법 제89조 제1항 제3호, 소득세법 시행령 제155조 제4항"

═══ confidence (신뢰도) 부여 기준 ═══

각 필드마다 0.0~1.0 사이의 confidence 값을 부여하세요:
- 0.9~1.0: 판례 본문에 해당 내용이 명확히 기재되어 있어 그대로 추출 가능
- 0.7~0.8: 내용이 있으나 일부 추론이 필요
- 0.4~0.6: 단편적 정보만 존재하여 상당한 추론 필요
- 0.1~0.3: 거의 정보가 없어 "해당없음"에 가까움

종합신뢰도는 5개 필드의 가중 평균으로 부여하되:
- 모든 필드가 명확하면 0.8 이상
- 2개 이상 "해당없음"이면 0.5 이하

═══ 출력 형식 ═══

반드시 아래 JSON만 출력하세요. 다른 텍스트 없이 JSON만 출력하세요.

{
  "사실관계": {"content": "...", "confidence": 0.9},
  "납세자주장": {"content": "...", "confidence": 0.8},
  "과세관청주장": {"content": "...", "confidence": 0.7},
  "판단근거": {"content": "...", "confidence": 0.9},
  "관련법령": {"content": "...", "confidence": 0.95},
  "종합신뢰도": 0.85
}

═══ 판례 원문 ═══
"""


# ============================================================
# [보강 ②] 스키마 검증 + 1회 재시도 로직
# ============================================================

RETRY_PROMPT_TEMPLATE = """이전 응답이 스키마 검증에 실패했습니다.

검증 오류: {validation_error}

다음 규칙을 반드시 준수하여 다시 JSON을 생성하세요:
1. 모든 content 필드는 최소 10자 이상
2. 모든 confidence 값은 0.0~1.0 사이의 float
3. 종합신뢰도는 0.0~1.0 사이의 float
4. 내용이 없는 필드도 "해당없음 - 판례 본문에 관련 기술 없음"처럼 10자 이상으로 기재

원본 판례:
{body_text_truncated}

수정된 JSON만 출력하세요:
"""


async def parse_with_gemini(
    body_text: str,
    model: GenerativeModel,
    max_retries: int = 1
) -> dict:
    """
    Gemini를 통한 판례 구조화 파싱 (스키마 검증 + 재시도 포함)
    """
    prompt = STRUCTURED_PARSING_PROMPT + body_text[:15000]
    attempt = 0
    ai_text = ""

    while attempt <= max_retries:
        try:
            response = model.generate_content(prompt)
            ai_text = response.text.strip()

            ai_clean = re.sub(r'^```json\s*', '', ai_text)
            ai_clean = re.sub(r'\s*```$', '', ai_clean)
            parsed_json = json.loads(ai_clean)

            validated = StructuredPrecedent(**parsed_json)

            return {
                "status": "validated",
                "attempt": attempt + 1,
                "data": validated.model_dump(),
            }

        except json.JSONDecodeError as e:
            validation_error = f"JSON 파싱 실패: {str(e)}"

        except Exception as e:
            validation_error = str(e)

        attempt += 1
        if attempt <= max_retries:
            prompt = RETRY_PROMPT_TEMPLATE.format(
                validation_error=validation_error[:500],
                body_text_truncated=body_text[:10000]
            )

    return {
        "status": "validation_failed",
        "attempt": attempt,
        "validation_error": validation_error[:300],
        "raw_response": ai_text[:2000] if ai_text else "no response",
    }


# ============================================================
# [보강 ④] 벡터 DB 인덱싱 함수
# ─ Pinecone에는 '사실관계'와 '납세자주장' 필드만 결합하여 임베딩
# ============================================================

async def upsert_to_pinecone(
    prec_id: str,
    case_name: str,
    case_no: str,
    structured_data: dict,
) -> dict:
    """
    Pinecone 벡터 DB에 판례 임베딩 업서트
    - 사실관계 + 납세자주장만 결합하여 임베딩 (Targeted Semantic Search)
    - 나머지 3개 필드는 metadata로 저장
    """
    fields = structured_data.get("data", structured_data)

    embed_text_parts = []
    for field_name in ["사실관계", "납세자주장"]:
        field_data = fields.get(field_name, {})
        content = field_data.get("content", "") if isinstance(field_data, dict) else str(field_data)
        if content and content != "해당없음":
            embed_text_parts.append(f"[{field_name}] {content}")

    embed_text = "\n".join(embed_text_parts)

    if len(embed_text.strip()) < 20:
        return {"status": "skipped", "reason": "임베딩 대상 텍스트 부족"}

    metadata = {
        "사건번호": case_no,
        "사건명": case_name,
        "판례일련번호": prec_id,
    }

    for field_name in ["과세관청주장", "판단근거", "관련법령"]:
        field_data = fields.get(field_name, {})
        content = field_data.get("content", "") if isinstance(field_data, dict) else str(field_data)
        metadata[field_name] = content[:500]

    metadata["종합신뢰도"] = fields.get("종합신뢰도", 0.0)

    for field_name in ["사실관계", "납세자주장"]:
        field_data = fields.get(field_name, {})
        content = field_data.get("content", "") if isinstance(field_data, dict) else str(field_data)
        metadata[field_name] = content[:500]

    try:
        idx = get_pinecone_index()
        idx.upsert_records(
            namespace="tax_cases",
            records=[
                {
                    "id": f"prec_{prec_id}",
                    "_text": embed_text,
                    **metadata,
                }
            ],
        )
        return {"status": "ok", "id": f"prec_{prec_id}", "embed_text_length": len(embed_text)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================
# 유틸리티: 국가법령정보 API 리다이렉트 파서
# ============================================================

def parse_redirect(text: str) -> str | None:
    o_match = re.search(r"o:'([^']+)'", text)
    c_match = re.search(r"c:(\d+)", text)
    z_match = re.search(r"z=(\d+)", text)
    if o_match and c_match and z_match:
        o = o_match.group(1)
        c = int(c_match.group(1))
        z = int(z_match.group(1))
        path = o[:c] + o[c + z:]
        return f"http://www.law.go.kr{path}"
    pairs = re.findall(r"(\w+):'([^']*)'", text)
    if pairs:
        x = dict(pairs)
        rsu_match = re.search(r"return\s+([\w.+]+)", text)
        if rsu_match:
            parts = rsu_match.group(1).split("+")
            path = ""
            for p in parts:
                p = p.strip()
                if p.startswith("x."):
                    path += x.get(p[2:], "")
            if path.startswith("/"):
                return f"http://www.law.go.kr{path}"
    return None


async def call_law_api(url: str, max_hops: int = 10) -> dict:
    timeout = httpx.Timeout(30.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        current_url = url
        for i in range(max_hops):
            resp = await client.get(current_url, headers=HEADERS)
            try:
                data = resp.json()
                return data
            except Exception:
                pass
            next_url = parse_redirect(resp.text)
            if not next_url:
                raise Exception(f"hop {i}: no redirect in: {resp.text[:200]}")
            current_url = next_url
        raise Exception(f"exceeded {max_hops} hops")


# ============================================================
# API 엔드포인트
# ============================================================

@app.get("/")
def health():
    return {"service": "EEHO AI Engine", "version": "1.1", "status": "running"}


@app.get("/check-ip")
async def check_ip():
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.ipify.org?format=json")
        return {"outbound_ip": resp.json()["ip"]}


@app.get("/test-pipeline")
async def test_pipeline():
    """
    통합 테스트 파이프라인
    ─ Step 1: 판례 목록 조회
    ─ Step 2: 판례 본문 조회
    ─ Step 3: Gemini 구조화 파싱 (스키마 검증 + 재시도)
    ─ Step 4: GCS 저장
    ─ Step 5: Pinecone 벡터 업서트 (사실관계+납세자주장 타겟)
    """
    pipeline_log = {}

    # === STEP 1: 판례 목록 1건 ===
    try:
        list_data = await call_law_api(
            f"http://www.law.go.kr/DRF/lawSearch.do"
            f"?OC={OC}&target=prec&type=JSON&display=1&query=양도"
        )
        prec_container = list_data.get("PrecSearch", list_data)
        prec_list = prec_container.get("prec", [])
        if isinstance(prec_list, dict):
            prec_list = [prec_list]
        if not prec_list:
            return {"step": "1", "error": "empty", "raw": str(list_data)[:500]}
        first = prec_list[0]
        prec_id = first.get("판례일련번호")
        case_name = first.get("사건명", "")
        case_no = first.get("사건번호", "")
        pipeline_log["step1_판례목록조회"] = {
            "status": "ok", "id": prec_id,
            "사건명": case_name, "사건번호": case_no
        }
    except Exception as e:
        return {"step": "1", "error": str(e), "trace": tb.format_exc()[-500:]}

    await asyncio.sleep(2)

    # === STEP 2: 판례 본문 조회 ===
    try:
        detail_url = (
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target=prec&ID={prec_id}&type=JSON"
        )
        detail_data = await call_law_api(detail_url)
        body_text = json.dumps(detail_data, ensure_ascii=False, indent=2)

        if "일치하는" in body_text and len(body_text) < 200:
            prec_id_alt = first.get("id", prec_id)
            detail_url2 = (
                f"http://www.law.go.kr/DRF/lawService.do"
                f"?OC={OC}&target=prec&ID={prec_id_alt}&type=JSON"
            )
            detail_data = await call_law_api(detail_url2)
            body_text = json.dumps(detail_data, ensure_ascii=False, indent=2)
            pipeline_log["step2_본문조회"] = {
                "status": "ok", "본문길이": len(body_text),
                "note": f"판례일련번호({prec_id}) 실패 → id({prec_id_alt})로 재시도"
            }
        else:
            pipeline_log["step2_본문조회"] = {
                "status": "ok", "본문길이": len(body_text)
            }
    except Exception as e:
        return {"step": "2", "error": str(e), "log": pipeline_log}

    if len(body_text) < 100:
        pipeline_log["step3_구조화"] = {
            "status": "skipped",
            "reason": f"본문 {len(body_text)}자로 파싱 불가",
            "preview": body_text
        }
        return {"pipeline": pipeline_log}

    # === STEP 3: Gemini 구조화 파싱 (보강된 버전) ===
    try:
        vertexai.init(project=PROJECT_ID, location=GCP_LOCATION)
        model = GenerativeModel(GEMINI_MODEL)

        parse_result = await parse_with_gemini(body_text, model, max_retries=1)

        pipeline_log["step3_구조화"] = {
            "status": parse_result["status"],
            "attempt": parse_result.get("attempt"),
            "종합신뢰도": (
                parse_result["data"]["종합신뢰도"]
                if parse_result["status"] == "validated"
                else None
            ),
        }

        if parse_result["status"] == "validated":
            structured = parse_result["data"]
            pipeline_log["step3_구조화"]["structured"] = structured
        else:
            pipeline_log["step3_구조화"]["validation_error"] = parse_result.get("validation_error")
            return {"pipeline": pipeline_log}

    except Exception as e:
        pipeline_log["step3_구조화"] = {
            "status": "error", "error": str(e)
        }
        return {"pipeline": pipeline_log}

    # === STEP 4: GCS 저장 ===
    try:
        save_data = {
            "meta": {
                "판례일련번호": prec_id,
                "사건명": case_name,
                "사건번호": case_no,
                "수집일시": datetime.utcnow().isoformat(),
                "소스": "국가법령정보_판례",
                "파싱버전": "v1.1",
                "파싱시도횟수": parse_result.get("attempt", 1),
                "종합신뢰도": structured["종합신뢰도"],
            },
            "원본": detail_data,
            "구조화": structured,
        }
        gcs = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(f"예판/판례/{prec_id}.json")
        blob.upload_from_string(
            json.dumps(save_data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        pipeline_log["step4_GCS저장"] = {
            "status": "ok",
            "path": f"gs://{BUCKET_NAME}/예판/판례/{prec_id}.json"
        }
    except Exception as e:
        pipeline_log["step4_GCS저장"] = {"status": "error", "error": str(e)}

    # === STEP 5: Pinecone 벡터 업서트 ===
    try:
        upsert_result = await upsert_to_pinecone(
            prec_id=prec_id,
            case_name=case_name,
            case_no=case_no,
            structured_data=structured,
        )
        pipeline_log["step5_Pinecone업서트"] = upsert_result
    except Exception as e:
        pipeline_log["step5_Pinecone업서트"] = {"status": "error", "error": str(e)}

    return {"pipeline": pipeline_log}


# ============================================================
# 배치 파싱 엔드포인트 (대량 판례 처리용)
# ============================================================

@app.get("/batch-pipeline")
async def batch_pipeline(query: str = "양도", count: int = 5):
    """
    복수 판례를 순차적으로 파싱 → GCS 저장 → Pinecone 업서트
    """
    count = min(count, 20)
    results = []

    try:
        list_data = await call_law_api(
            f"http://www.law.go.kr/DRF/lawSearch.do"
            f"?OC={OC}&target=prec&type=JSON&display={count}&query={query}"
        )
        prec_container = list_data.get("PrecSearch", list_data)
        prec_list = prec_container.get("prec", [])
        if isinstance(prec_list, dict):
            prec_list = [prec_list]
    except Exception as e:
        return {"error": f"목록 조회 실패: {str(e)}"}

    vertexai.init(project=PROJECT_ID, location=GCP_LOCATION)
    model = GenerativeModel(GEMINI_MODEL)

    for item in prec_list:
        prec_id = item.get("판례일련번호")
        case_name = item.get("사건명", "")
        case_no = item.get("사건번호", "")
        entry = {"id": prec_id, "사건명": case_name}

        try:
            await asyncio.sleep(2)

            detail_data = await call_law_api(
                f"http://www.law.go.kr/DRF/lawService.do"
                f"?OC={OC}&target=prec&ID={prec_id}&type=JSON"
            )
            body_text = json.dumps(detail_data, ensure_ascii=False, indent=2)

            if len(body_text) < 100:
                entry["status"] = "skipped_short"
                results.append(entry)
                continue

            parse_result = await parse_with_gemini(body_text, model, max_retries=1)
            entry["parse_status"] = parse_result["status"]

            if parse_result["status"] == "validated":
                structured = parse_result["data"]
                entry["종합신뢰도"] = structured["종합신뢰도"]

                save_data = {
                    "meta": {
                        "판례일련번호": prec_id, "사건명": case_name,
                        "사건번호": case_no,
                        "수집일시": datetime.utcnow().isoformat(),
                        "소스": "국가법령정보_판례", "파싱버전": "v1.1",
                        "종합신뢰도": structured["종합신뢰도"],
                    },
                    "원본": detail_data,
                    "구조화": structured,
                }
                gcs = storage.Client()
                bucket = gcs.bucket(BUCKET_NAME)
                blob = bucket.blob(f"예판/판례/{prec_id}.json")
                blob.upload_from_string(
                    json.dumps(save_data, ensure_ascii=False, indent=2),
                    content_type="application/json"
                )

                upsert_result = await upsert_to_pinecone(
                    prec_id, case_name, case_no, structured
                )
                entry["pinecone"] = upsert_result["status"]
            else:
                entry["error"] = parse_result.get("validation_error", "")[:200]

        except Exception as e:
            entry["status"] = "error"
            entry["error"] = str(e)[:200]

        results.append(entry)

    return {
        "query": query,
        "requested": count,
        "processed": len(results),
        "results": results,
    }


# ============================================================
# 디버그 엔드포인트: 판례 검색 API 원본 응답 확인용
# ============================================================

@app.get("/debug-search")
async def debug_search():
    data = await call_law_api(
        f"http://www.law.go.kr/DRF/lawSearch.do"
        f"?OC={OC}&target=prec&type=JSON&display=1&query=양도소득세"
    )
    return {"raw": data}

@app.get("/debug-detail/{prec_id}")
async def debug_detail(prec_id: str):
    results = {}
    # JSON으로 시도
    try:
        data_json = await call_law_api(
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target=prec&ID={prec_id}&type=JSON"
        )
        results["json"] = data_json
    except Exception as e:
        results["json_error"] = str(e)
    # XML로 시도
    try:
        data_xml = await call_law_api(
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target=prec&ID={prec_id}&type=XML"
        )
        results["xml"] = data_xml
    except Exception as e:
        results["xml_error"] = str(e)
    return results
