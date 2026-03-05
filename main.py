import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Query
import httpx
import re
import asyncio
import json
import traceback as tb
from datetime import datetime, timedelta, timezone
from typing import Optional
from pydantic import BaseModel, Field, field_validator

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import storage
from pinecone import Pinecone
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

app = FastAPI(title="EEHO AI Engine", version="2.1")

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

# 이메일 알림 설정 (삭제된 예판 감지 시 발송)
ALERT_EMAIL_TO = os.environ.get("ALERT_EMAIL_TO", "hjcta923@gmail.com")
ALERT_EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM", "")  # 발신 Gmail 주소
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")  # Gmail 앱 비밀번호

KST = timezone(timedelta(hours=9))

# Pinecone 지연 로딩
pc = None
pinecone_index = None


def get_pinecone_index():
    global pc, pinecone_index
    if pinecone_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
    return pinecone_index


# ============================================================
# [핵심] 세목별 수집 쿼리 설정
# ─────────────────────────────────────────
# 각 세목에 대해 판례(prec)와 법령해석례(expc) 두 가지 target으로
# 수집합니다. 쿼리는 세목 핵심 키워드 + 실무 쟁점 키워드로 구성하여
# 누락을 최소화합니다.
#
# [수집 대상 세목]
# 1. 양도소득세 (소득세법)
# 2. 상속세 (상속세및증여세법)
# 3. 증여세 (상속세및증여세법)
# 4. 취득세 (지방세법)
# 5. 재산세 (지방세법)
# 6. 종합부동산세 (종합부동산세법)
# 7. 조세특례제한법 (위 세목 관련 과세특례)
# 8. 지방세특례제한법 (취득세·재산세 관련 과세특례)
# ============================================================

COLLECTION_QUERIES = {
    # ── 양도소득세 관련 ──
    "양도소득세": {
        "targets": ["prec", "expc"],
        "queries": [
            "양도소득세",
            "1세대1주택 비과세",
            "장기보유특별공제",
            "다주택 중과",
            "양도 비과세",
            "분양권 양도",
            "조정대상지역 양도",
        ],
    },
    # ── 상속세 관련 ──
    "상속세": {
        "targets": ["prec", "expc"],
        "queries": [
            "상속세",
            "상속 공제",
            "상속재산 평가",
            "상속세 과세가액",
            "배우자 상속공제",
        ],
    },
    # ── 증여세 관련 ──
    "증여세": {
        "targets": ["prec", "expc"],
        "queries": [
            "증여세",
            "증여재산 평가",
            "증여 공제",
            "부담부증여",
            "특수관계인 증여",
        ],
    },
    # ── 취득세 관련 ──
    "취득세": {
        "targets": ["prec", "expc"],
        "queries": [
            "취득세",
            "취득세 중과",
            "취득세 감면",
            "취득세 비과세",
        ],
    },
    # ── 재산세 관련 ──
    "재산세": {
        "targets": ["prec", "expc"],
        "queries": [
            "재산세",
            "재산세 과세",
            "재산세 감면",
        ],
    },
    # ── 종합부동산세 관련 ──
    "종합부동산세": {
        "targets": ["prec", "expc"],
        "queries": [
            "종합부동산세",
            "종부세",
            "종부세 합산",
        ],
    },
    # ── 조세특례제한법 (국세: 양도·상속·증여 관련 과세특례) ──
    "조세특례제한법": {
        "targets": ["prec", "expc"],
        "queries": [
            "조세특례제한법 양도",
            "조세특례제한법 상속",
            "조세특례제한법 증여",
            "조특법 감면",
        ],
    },
    # ── 지방세특례제한법 (지방세: 취득세·재산세 관련 과세특례) ──
    "지방세특례제한법": {
        "targets": ["prec", "expc"],
        "queries": [
            "지방세특례제한법 취득",
            "지방세특례제한법 재산",
            "지방세특례제한법 감면",
        ],
    },
}

# 일일 수집 시 쿼리당 최대 페이지 수 (API 부하 방지)
DAILY_MAX_PAGES_PER_QUERY = 2
DAILY_DISPLAY_PER_PAGE = 20
# 백필 수집 시 쿼리당 최대 페이지 수
BACKFILL_MAX_PAGES_PER_QUERY = 50
BACKFILL_DISPLAY_PER_PAGE = 100
# API 호출 간 대기 시간 (초)
API_CALL_DELAY = 1.5  # 백필 속도 개선을 위해 2.0 → 1.5초로 단축
# Gemini 호출 간 대기 시간 (초) — rate limit 방지
GEMINI_CALL_DELAY = 2.0  # 3.0 → 2.0초로 단축 (Gemini 2.5 Pro는 RPM 여유 있음)


# ============================================================
# [보강 ①] 판례 5단계 파싱 스키마 (Pydantic)
# ============================================================

class ParsedField(BaseModel):
    content: str = Field(..., min_length=10, description="추출된 내용 (최소 10자)")
    confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        stripped = v.strip()
        if len(stripped) < 2:
            raise ValueError("필드 내용이 너무 짧습니다")
        return stripped


class StructuredPrecedent(BaseModel):
    사실관계: ParsedField
    납세자주장: ParsedField
    과세관청주장: ParsedField
    판단근거: ParsedField
    관련법령: ParsedField
    종합신뢰도: float = Field(..., ge=0.0, le=1.0)

    @field_validator("종합신뢰도")
    @classmethod
    def validate_overall_confidence(cls, v):
        return round(v, 2)


# ============================================================
# 구조화 파싱 프롬프트
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
   ※ 약식 판례로 납세자 주장이 없으면 "해당없음 - 판례 본문에 납세자 주장 관련 기술 없음" 기재

3. 과세관청주장 (필수 포함 항목):
   - 과세처분의 근거 조문
   - 납세자가 미충족한다고 보는 구체적 요건
   - 과세관청의 사실관계 해석
   ※ 과세관청 주장이 없으면 "해당없음 - 판례 본문에 과세관청 주장 관련 기술 없음" 기재

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

═══ 판례/해석례 원문 ═══
"""


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
# Pinecone 업서트 (사실관계 + 납세자주장 타겟)
# ============================================================

async def upsert_to_pinecone(
    record_id: str,
    case_name: str,
    case_no: str,
    structured_data: dict,
    source_type: str = "판례",  # "판례" 또는 "법령해석례"
    tax_category: str = "",
) -> dict:
    """
    Pinecone tax_cases namespace에 업서트
    - text: 사실관계 + 납세자주장 결합 (Targeted Semantic Search)
    - metadata: 나머지 필드 + 세목 분류 + 소스 유형
    """
    fields = structured_data.get("data", structured_data)

    embed_text_parts = []
    for field_name in ["사실관계", "납세자주장"]:
        field_data = fields.get(field_name, {})
        content = field_data.get("content", "") if isinstance(field_data, dict) else str(field_data)
        if content and "해당없음" not in content:
            embed_text_parts.append(f"[{field_name}] {content}")

    embed_text = "\n".join(embed_text_parts)
    if len(embed_text.strip()) < 20:
        return {"status": "skipped", "reason": "임베딩 대상 텍스트 부족"}

    metadata = {
        "사건번호": case_no,
        "사건명": case_name,
        "record_id": record_id,
        "source_type": source_type,       # 판례 / 법령해석례
        "tax_category": tax_category,     # 양도소득세, 상속세 등
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
            records=[{
                "id": f"{source_type}_{record_id}",
                "text": embed_text,
                **metadata,
            }],
        )
        return {"status": "ok", "id": f"{source_type}_{record_id}", "embed_text_length": len(embed_text)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ============================================================
# 국가법령정보 API 유틸리티
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
# GCS 매니페스트 관리 (수집 완료 ID 추적 → 중복 방지)
# ─────────────────────────────────────────
# 매니페스트는 GCS에 JSON으로 저장되며,
# 이미 수집·파싱·저장이 완료된 판례/해석례의 ID 목록을 관리합니다.
# ============================================================

MANIFEST_PATH = "수집관리/collected_ids.json"


def load_manifest() -> dict:
    """
    GCS에서 수집 완료 매니페스트를 로드합니다.
    구조: {
        "prec": {"12345": "2026-03-05T03:00:00", ...},
        "expc": {"67890": "2026-03-05T03:00:00", ...},
        "last_updated": "2026-03-05T03:00:00"
    }
    """
    try:
        gcs = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(MANIFEST_PATH)
        if blob.exists():
            data = json.loads(blob.download_as_text())
            return data
    except Exception:
        pass
    return {"prec": {}, "expc": {}, "last_updated": None}


def save_manifest(manifest: dict):
    """매니페스트를 GCS에 저장합니다."""
    manifest["last_updated"] = datetime.now(KST).isoformat()
    gcs = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(MANIFEST_PATH)
    blob.upload_from_string(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        content_type="application/json"
    )


def is_already_collected(manifest: dict, target: str, record_id: str) -> bool:
    """해당 판례/해석례가 이미 수집되었는지 확인합니다."""
    return str(record_id) in manifest.get(target, {})


def mark_collected(manifest: dict, target: str, record_id: str):
    """수집 완료로 표시합니다."""
    if target not in manifest:
        manifest[target] = {}
    manifest[target][str(record_id)] = datetime.now(KST).isoformat()


# ============================================================
# [핵심] 판례/해석례 목록 검색 함수
# ─────────────────────────────────────────
# target별로 API 응답 구조가 다르므로 분리 처리합니다.
# - prec (판례): PrecSearch.prec[]
# - expc (법령해석례): ExpcSearch.expc[]
# ============================================================

async def search_cases(
    target: str,
    query: str,
    display: int = 20,
    page: int = 1,
) -> tuple[list[dict], int]:
    """
    국가법령정보 API에서 판례/해석례 목록을 검색합니다.

    Returns:
        (결과 리스트, 총 건수)
    """
    url = (
        f"http://www.law.go.kr/DRF/lawSearch.do"
        f"?OC={OC}&target={target}&type=JSON"
        f"&display={display}&page={page}"
        f"&query={query}"
    )

    data = await call_law_api(url)

    if target == "prec":
        container = data.get("PrecSearch", data)
        items = container.get("prec", [])
        total_str = container.get("totalCnt", "0")
    elif target == "expc":
        container = data.get("ExpcSearch", data)
        items = container.get("expc", [])
        total_str = container.get("totalCnt", "0")
    else:
        return [], 0

    if isinstance(items, dict):
        items = [items]

    try:
        total = int(total_str)
    except (ValueError, TypeError):
        total = len(items)

    return items, total


def extract_record_id(item: dict, target: str) -> str:
    """target에 따라 적절한 일련번호 필드를 추출합니다."""
    if target == "prec":
        return item.get("판례일련번호", "")
    elif target == "expc":
        return item.get("법령해석례일련번호", "")
    return ""


def extract_case_info(item: dict, target: str) -> dict:
    """target에 따라 사건명, 사건번호 등 기본 정보를 추출합니다."""
    if target == "prec":
        return {
            "record_id": item.get("판례일련번호", ""),
            "case_name": item.get("사건명", ""),
            "case_no": item.get("사건번호", ""),
        }
    elif target == "expc":
        return {
            "record_id": item.get("법령해석례일련번호", ""),
            "case_name": item.get("법령해석례명", item.get("사건명", "")),
            "case_no": item.get("사건번호", item.get("안건번호", "")),
        }
    return {}


# ============================================================
# [보강] 이메일 알림 시스템
# ─────────────────────────────────────────
# 삭제된 예판이 감지되면 hjcta923@gmail.com으로 알림 발송
# Gmail SMTP + 앱 비밀번호 사용
#
# [환경변수 설정 필요]
# ALERT_EMAIL_FROM: 발신 Gmail 주소 (예: hjcta923@gmail.com)
# GMAIL_APP_PASSWORD: Google 계정 → 보안 → 앱 비밀번호에서 생성
#   (https://myaccount.google.com/apppasswords)
# ALERT_EMAIL_TO: 수신 이메일 (기본: hjcta923@gmail.com)
# ============================================================

async def send_alert_email(subject: str, body_html: str) -> dict:
    """
    Gmail SMTP를 통해 알림 이메일을 발송합니다.
    환경변수 미설정 시 발송을 건너뛰고 로그만 남깁니다.
    """
    if not ALERT_EMAIL_FROM or not GMAIL_APP_PASSWORD:
        return {
            "status": "skipped",
            "reason": "ALERT_EMAIL_FROM 또는 GMAIL_APP_PASSWORD 미설정"
        }

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"EEHO AI Engine <{ALERT_EMAIL_FROM}>"
    msg["To"] = ALERT_EMAIL_TO

    # 텍스트 폴백 (HTML을 못 읽는 클라이언트용)
    text_body = body_html.replace("<br>", "\n").replace("</tr>", "\n")
    text_body = re.sub(r"<[^>]+>", "", text_body)
    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    try:
        await aiosmtplib.send(
            msg,
            hostname="smtp.gmail.com",
            port=587,
            start_tls=True,
            username=ALERT_EMAIL_FROM,
            password=GMAIL_APP_PASSWORD,
        )
        return {"status": "sent", "to": ALERT_EMAIL_TO}
    except Exception as e:
        return {"status": "error", "error": str(e)[:300]}


def build_deletion_alert_html(deleted_cases: list[dict]) -> str:
    """삭제된 예판 목록을 HTML 이메일 본문으로 변환합니다."""
    now_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M KST")

    rows = ""
    for case in deleted_cases:
        source = case.get("source_type", "")
        record_id = case.get("record_id", "")
        case_name = case.get("case_name", "")
        case_no = case.get("case_no", "")
        tax_cat = case.get("tax_category", "")
        rows += f"""
        <tr>
            <td style="padding:8px;border:1px solid #ddd;">{source}</td>
            <td style="padding:8px;border:1px solid #ddd;">{record_id}</td>
            <td style="padding:8px;border:1px solid #ddd;">{case_name}</td>
            <td style="padding:8px;border:1px solid #ddd;">{case_no}</td>
            <td style="padding:8px;border:1px solid #ddd;">{tax_cat}</td>
        </tr>
        """

    html = f"""
    <html>
    <body style="font-family:Arial,sans-serif;color:#333;">
        <div style="background:#004447;color:white;padding:16px 24px;border-radius:8px 8px 0 0;">
            <h2 style="margin:0;">⚠️ EEHO AI — 삭제된 예판 감지 알림</h2>
        </div>
        <div style="padding:20px;border:1px solid #ddd;border-top:none;">
            <p>아래 예판이 국가법령정보시스템에서 삭제(또는 비공개 전환)된 것으로 확인되었습니다.</p>
            <p style="color:#666;font-size:13px;">감지 시각: {now_str}</p>

            <table style="border-collapse:collapse;width:100%;margin:16px 0;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:8px;border:1px solid #ddd;text-align:left;">유형</th>
                    <th style="padding:8px;border:1px solid #ddd;text-align:left;">일련번호</th>
                    <th style="padding:8px;border:1px solid #ddd;text-align:left;">사건명</th>
                    <th style="padding:8px;border:1px solid #ddd;text-align:left;">사건번호</th>
                    <th style="padding:8px;border:1px solid #ddd;text-align:left;">세목</th>
                </tr>
                {rows}
            </table>

            <p><strong>조치 사항:</strong></p>
            <ul>
                <li>Pinecone tax_cases namespace에서 해당 벡터가 삭제 처리되었습니다.</li>
                <li>GCS 원본 파일은 보존되어 있습니다 (참고용).</li>
                <li>해당 예판이 실무상 중요했다면, 판단 근거가 변경되었을 수 있으므로 관련 사안을 재검토해 주세요.</li>
            </ul>

            <hr style="border:none;border-top:1px solid #eee;margin:20px 0;">
            <p style="color:#999;font-size:12px;">
                이 메일은 EEHO AI Engine이 자동으로 발송한 알림입니다.<br>
                삭제 감지는 /check-deletions 엔드포인트에 의해 수행됩니다.
            </p>
        </div>
    </body>
    </html>
    """
    return html


# ============================================================
# [보강] 삭제 예판 감지 + Pinecone 정리 함수
# ============================================================

async def verify_case_exists(record_id: str, target: str) -> bool:
    """
    국가법령정보 API에서 해당 판례/해석례가 아직 존재하는지 확인합니다.
    상세 조회 API를 호출하여 "일치하는" 에러 또는 빈 응답이면 삭제된 것으로 판단합니다.
    """
    try:
        detail_url = (
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target={target}&ID={record_id}&type=JSON"
        )
        detail_data = await call_law_api(detail_url)
        body_text = json.dumps(detail_data, ensure_ascii=False)

        # "일치하는 판례가 없습니다" 또는 빈 응답이면 삭제된 것
        if "일치하는" in body_text and len(body_text) < 300:
            return False
        if len(body_text) < 50:
            return False
        return True
    except Exception:
        # API 오류는 삭제가 아닌 일시적 오류로 간주 → 존재한다고 보수적 판단
        return True


async def remove_from_pinecone(record_id: str, source_type: str) -> dict:
    """Pinecone에서 해당 벡터를 삭제합니다."""
    try:
        idx = get_pinecone_index()
        vector_id = f"{source_type}_{record_id}"
        idx.delete(ids=[vector_id], namespace="tax_cases")
        return {"status": "deleted", "id": vector_id}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}


# ============================================================
# [핵심] 개별 판례/해석례 상세 조회 + 파싱 + 저장 + 업서트
# ============================================================

async def process_single_case(
    record_id: str,
    case_name: str,
    case_no: str,
    target: str,
    tax_category: str,
    model: GenerativeModel,
) -> dict:
    """
    단일 판례/해석례를 처리하는 통합 파이프라인:
    1. 상세 본문 조회 (law.go.kr API)
    2. Gemini 5단계 구조화 파싱
    3. GCS 원본 + 구조화 데이터 저장
    4. Pinecone 벡터 업서트
    """
    source_type = "판례" if target == "prec" else "법령해석례"
    result = {
        "record_id": record_id,
        "case_name": case_name[:60],
        "source_type": source_type,
        "tax_category": tax_category,
    }

    # ── Step 1: 상세 본문 조회 ──
    try:
        detail_url = (
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target={target}&ID={record_id}&type=JSON"
        )
        detail_data = await call_law_api(detail_url)
        body_text = json.dumps(detail_data, ensure_ascii=False, indent=2)

        # "일치하는 판례가 없습니다" 등 빈 응답 체크
        if ("일치하는" in body_text and len(body_text) < 300) or len(body_text) < 100:
            result["status"] = "skipped_empty"
            result["body_length"] = len(body_text)
            return result

        result["body_length"] = len(body_text)
    except Exception as e:
        result["status"] = "error_fetch"
        result["error"] = str(e)[:200]
        return result

    # ── Step 2: Gemini 구조화 파싱 ──
    await asyncio.sleep(GEMINI_CALL_DELAY)
    try:
        parse_result = await parse_with_gemini(body_text, model, max_retries=1)
        result["parse_status"] = parse_result["status"]
        result["parse_attempts"] = parse_result.get("attempt", 0)

        if parse_result["status"] != "validated":
            result["status"] = "parse_failed"
            result["validation_error"] = parse_result.get("validation_error", "")[:200]
            return result

        structured = parse_result["data"]
        result["종합신뢰도"] = structured["종합신뢰도"]
    except Exception as e:
        result["status"] = "error_parse"
        result["error"] = str(e)[:200]
        return result

    # ── Step 3: GCS 저장 ──
    try:
        # 판례와 해석례를 별도 경로로 저장
        gcs_folder = "판례" if target == "prec" else "법령해석례"
        gcs_path = f"예판/{gcs_folder}/{record_id}.json"

        save_data = {
            "meta": {
                "record_id": record_id,
                "사건명": case_name,
                "사건번호": case_no,
                "수집일시": datetime.now(KST).isoformat(),
                "소스": f"국가법령정보_{source_type}",
                "소스_target": target,
                "세목분류": tax_category,
                "파싱버전": "v2.0",
                "파싱시도횟수": parse_result.get("attempt", 1),
                "종합신뢰도": structured["종합신뢰도"],
            },
            "원본": detail_data,
            "구조화": structured,
        }

        gcs = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(save_data, ensure_ascii=False, indent=2),
            content_type="application/json"
        )
        result["gcs_path"] = f"gs://{BUCKET_NAME}/{gcs_path}"
    except Exception as e:
        result["gcs_status"] = "error"
        result["gcs_error"] = str(e)[:200]
        # GCS 실패해도 Pinecone은 시도

    # ── Step 4: Pinecone 업서트 ──
    try:
        upsert_result = await upsert_to_pinecone(
            record_id=record_id,
            case_name=case_name,
            case_no=case_no,
            structured_data=structured,
            source_type=source_type,
            tax_category=tax_category,
        )
        result["pinecone_status"] = upsert_result["status"]
    except Exception as e:
        result["pinecone_status"] = "error"
        result["pinecone_error"] = str(e)[:200]

    result["status"] = "success"
    return result


# ============================================================
# API 엔드포인트
# ============================================================

@app.get("/")
def health():
    return {
        "service": "EEHO AI Engine",
        "version": "2.0",
        "status": "running",
        "description": "예판 수집 엔진 (판례 + 법령해석례)",
        "tax_categories": list(COLLECTION_QUERIES.keys()),
    }


@app.get("/check-ip")
async def check_ip():
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.ipify.org?format=json")
        return {"outbound_ip": resp.json()["ip"]}


# ============================================================
# [핵심 엔드포인트] 일일 수집 (/collect-daily)
# ─────────────────────────────────────────
# Cloud Scheduler가 매일 새벽 3시(KST)에 호출합니다.
#
# 동작 방식:
# 1. GCS에서 수집 완료 매니페스트를 로드
# 2. 각 세목 × 쿼리 × target(prec/expc)에 대해 검색
# 3. 이미 수집된 건은 스킵
# 4. 미수집 건에 대해 상세조회 → 파싱 → GCS → Pinecone
# 5. 매니페스트 업데이트 후 GCS에 저장
#
# 일일 총 처리량 제한: max_total (기본 30건)
# → API 안정성과 Gemini 호출 비용을 고려한 보수적 설정
# ============================================================

@app.post("/collect-daily")
async def collect_daily(
    max_total: int = Query(default=30, description="일일 최대 처리 건수", le=100),
):
    """
    매일 새벽 3시 Cloud Scheduler가 호출하는 일일 수집 엔드포인트.
    모든 세목에 대해 판례(prec) + 법령해석례(expc)를 수집합니다.
    """
    start_time = datetime.now(KST)
    manifest = load_manifest()
    total_processed = 0
    total_skipped = 0
    total_new = 0
    results_by_category = {}
    errors = []

    # Gemini 모델 초기화 (전체 수집에서 1번만)
    vertexai.init(project=PROJECT_ID, location=GCP_LOCATION)
    model = GenerativeModel(GEMINI_MODEL)

    for category, config in COLLECTION_QUERIES.items():
        category_results = []

        for target in config["targets"]:
            for query in config["queries"]:
                if total_new >= max_total:
                    break

                try:
                    # 1페이지만 조회 (일일 수집이므로 최신 건만)
                    for page in range(1, DAILY_MAX_PAGES_PER_QUERY + 1):
                        if total_new >= max_total:
                            break

                        await asyncio.sleep(API_CALL_DELAY)
                        items, total_count = await search_cases(
                            target=target,
                            query=query,
                            display=DAILY_DISPLAY_PER_PAGE,
                            page=page,
                        )

                        if not items:
                            break

                        for item in items:
                            if total_new >= max_total:
                                break

                            record_id = extract_record_id(item, target)
                            if not record_id:
                                continue

                            # 중복 체크
                            if is_already_collected(manifest, target, record_id):
                                total_skipped += 1
                                continue

                            info = extract_case_info(item, target)
                            await asyncio.sleep(API_CALL_DELAY)

                            case_result = await process_single_case(
                                record_id=info["record_id"],
                                case_name=info["case_name"],
                                case_no=info["case_no"],
                                target=target,
                                tax_category=category,
                                model=model,
                            )

                            total_processed += 1
                            if case_result.get("status") == "success":
                                total_new += 1
                                mark_collected(manifest, target, record_id)

                            category_results.append(case_result)

                except Exception as e:
                    errors.append({
                        "category": category,
                        "target": target,
                        "query": query,
                        "error": str(e)[:300],
                    })

            if total_new >= max_total:
                break
        if total_new >= max_total:
            results_by_category[category] = category_results
            break

        results_by_category[category] = category_results

    # 매니페스트 저장
    try:
        save_manifest(manifest)
    except Exception as e:
        errors.append({"manifest_save_error": str(e)[:200]})

    end_time = datetime.now(KST)
    elapsed = (end_time - start_time).total_seconds()

    return {
        "job": "collect-daily",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "summary": {
            "total_processed": total_processed,
            "total_new_saved": total_new,
            "total_skipped_duplicates": total_skipped,
            "manifest_prec_count": len(manifest.get("prec", {})),
            "manifest_expc_count": len(manifest.get("expc", {})),
        },
        "results_by_category": {
            cat: [
                {k: v for k, v in r.items() if k != "error" or v}
                for r in results
            ]
            for cat, results in results_by_category.items()
        },
        "errors": errors if errors else None,
    }


# ============================================================
# [백필 엔드포인트] 초기 대량 수집 (/backfill)
# ─────────────────────────────────────────
# 서비스 최초 구축 시, 과거 판례/해석례를 대량으로 수집할 때 사용합니다.
# 특정 세목과 target을 지정하여 호출합니다.
#
# 사용 예시:
#   POST /backfill?category=양도소득세&target=prec&max_items=50
#   POST /backfill?category=상속세&target=expc&max_items=30
# ============================================================

@app.post("/backfill")
async def backfill(
    category: str = Query(..., description="세목 (예: 양도소득세, 상속세)"),
    target: str = Query(default="prec", description="prec(판례) 또는 expc(법령해석례)"),
    max_items: int = Query(default=100, description="최대 수집 건수 (기본 100, 최대 500)", le=500),
    start_page: int = Query(default=1, description="시작 페이지"),
):
    """
    특정 세목의 과거 판례/해석례를 대량 수집하는 백필 엔드포인트.
    초기 데이터 적재 시 사용합니다.
    """
    if category not in COLLECTION_QUERIES:
        return {
            "error": f"지원하지 않는 세목: {category}",
            "available": list(COLLECTION_QUERIES.keys()),
        }
    if target not in ["prec", "expc"]:
        return {"error": "target은 'prec' 또는 'expc'만 가능합니다"}

    config = COLLECTION_QUERIES[category]
    if target not in config["targets"]:
        return {"error": f"{category}에 {target} target이 설정되어 있지 않습니다"}

    start_time = datetime.now(KST)
    manifest = load_manifest()

    vertexai.init(project=PROJECT_ID, location=GCP_LOCATION)
    model = GenerativeModel(GEMINI_MODEL)

    total_new = 0
    total_skipped = 0
    results = []
    errors = []

    for query in config["queries"]:
        if total_new >= max_items:
            break

        for page in range(start_page, start_page + BACKFILL_MAX_PAGES_PER_QUERY):
            if total_new >= max_items:
                break

            try:
                await asyncio.sleep(API_CALL_DELAY)
                items, total_count = await search_cases(
                    target=target,
                    query=query,
                    display=BACKFILL_DISPLAY_PER_PAGE,
                    page=page,
                )

                if not items:
                    break  # 더 이상 결과 없음

                for item in items:
                    if total_new >= max_items:
                        break

                    record_id = extract_record_id(item, target)
                    if not record_id:
                        continue

                    if is_already_collected(manifest, target, record_id):
                        total_skipped += 1
                        continue

                    info = extract_case_info(item, target)
                    await asyncio.sleep(API_CALL_DELAY)

                    case_result = await process_single_case(
                        record_id=info["record_id"],
                        case_name=info["case_name"],
                        case_no=info["case_no"],
                        target=target,
                        tax_category=category,
                        model=model,
                    )

                    if case_result.get("status") == "success":
                        total_new += 1
                        mark_collected(manifest, target, record_id)

                    results.append(case_result)

            except Exception as e:
                errors.append({
                    "query": query, "page": page,
                    "error": str(e)[:300],
                })

    # 매니페스트 저장
    try:
        save_manifest(manifest)
    except Exception as e:
        errors.append({"manifest_save_error": str(e)[:200]})

    end_time = datetime.now(KST)

    return {
        "job": "backfill",
        "category": category,
        "target": target,
        "elapsed_seconds": round((end_time - start_time).total_seconds(), 1),
        "summary": {
            "total_new_saved": total_new,
            "total_skipped_duplicates": total_skipped,
            "manifest_total": len(manifest.get(target, {})),
        },
        "results": results,
        "errors": errors if errors else None,
    }


# ============================================================
# 수집 현황 조회 (/collection-status)
# ============================================================

@app.get("/collection-status")
async def collection_status():
    """현재까지의 수집 현황을 매니페스트에서 조회합니다."""
    manifest = load_manifest()

    prec_ids = manifest.get("prec", {})
    expc_ids = manifest.get("expc", {})

    return {
        "last_updated": manifest.get("last_updated"),
        "총_판례_수집건수": len(prec_ids),
        "총_해석례_수집건수": len(expc_ids),
        "총_수집건수": len(prec_ids) + len(expc_ids),
        "최근_수집_판례_5건": dict(list(prec_ids.items())[-5:]) if prec_ids else {},
        "최근_수집_해석례_5건": dict(list(expc_ids.items())[-5:]) if expc_ids else {},
    }


# ============================================================
# 쿼리별 건수 사전조사 (/survey)
# ─────────────────────────────────────────
# 실제 수집 전에 각 세목·쿼리별로 API에 몇 건의 판례/해석례가
# 존재하는지 사전 조사합니다. 수집 계획 수립 시 활용합니다.
# ============================================================

@app.get("/survey")
async def survey():
    """모든 세목·쿼리별 예상 판례/해석례 건수를 조사합니다."""
    survey_results = {}

    for category, config in COLLECTION_QUERIES.items():
        category_data = {}
        for target in config["targets"]:
            target_data = {}
            for query in config["queries"]:
                try:
                    await asyncio.sleep(1.0)
                    _, total = await search_cases(
                        target=target, query=query, display=1, page=1,
                    )
                    target_data[query] = total
                except Exception as e:
                    target_data[query] = f"error: {str(e)[:100]}"
            category_data[target] = target_data
        survey_results[category] = category_data

    return {"survey": survey_results}


# ============================================================
# 기존 테스트 엔드포인트 유지 (하위 호환)
# ============================================================

@app.get("/test-pipeline")
async def test_pipeline():
    """기존 통합 테스트 파이프라인 (하위 호환)"""
    pipeline_log = {}

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
        pipeline_log["step1"] = {"status": "ok", "id": prec_id, "사건명": case_name}
    except Exception as e:
        return {"step": "1", "error": str(e)}

    await asyncio.sleep(2)

    vertexai.init(project=PROJECT_ID, location=GCP_LOCATION)
    model = GenerativeModel(GEMINI_MODEL)

    result = await process_single_case(
        record_id=prec_id,
        case_name=case_name,
        case_no=case_no,
        target="prec",
        tax_category="양도소득세",
        model=model,
    )
    pipeline_log["process_result"] = result

    return {"pipeline": pipeline_log}


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
    try:
        data_json = await call_law_api(
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target=prec&ID={prec_id}&type=JSON"
        )
        results["json"] = data_json
    except Exception as e:
        results["json_error"] = str(e)
    return results


# ============================================================
# [핵심 엔드포인트] 삭제 예판 감지 (/check-deletions)
# ─────────────────────────────────────────
# Cloud Scheduler가 매일 새벽 4시(KST)에 호출합니다.
# (collect-daily 완료 후 약 1시간 뒤)
#
# 동작 방식:
# 1. GCS 매니페스트에서 수집 완료 ID 목록을 로드
# 2. 각 ID에 대해 법령정보 API 상세조회를 재호출
# 3. "일치하는 판례가 없습니다" 등의 응답이면 삭제된 것으로 판단
# 4. 삭제 확인된 건:
#    - Pinecone에서 벡터 삭제
#    - GCS 원본은 보존 (참고용)
#    - 매니페스트에서 제거 (→ 향후 중복수집 방지 목록에서도 제거)
#    - 이메일 알림 발송 (hjcta923@gmail.com)
#
# 부하 관리:
# - 전체 매니페스트를 매일 전수 검사하면 API 호출이 과다하므로,
#   max_check 파라미터로 1일 검사 건수를 제한합니다 (기본 100건).
# - 매니페스트의 가장 오래된 수집 건부터 순환 검사합니다.
# ============================================================

# GCS에 "마지막으로 검사한 위치"를 기록하는 파일
DELETION_CURSOR_PATH = "수집관리/deletion_check_cursor.json"


def load_deletion_cursor() -> dict:
    """삭제 검사 커서 로드 (어디까지 검사했는지 추적)"""
    try:
        gcs = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(DELETION_CURSOR_PATH)
        if blob.exists():
            return json.loads(blob.download_as_text())
    except Exception:
        pass
    return {"prec_offset": 0, "expc_offset": 0, "last_checked": None}


def save_deletion_cursor(cursor: dict):
    """삭제 검사 커서 저장"""
    cursor["last_checked"] = datetime.now(KST).isoformat()
    gcs = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)
    blob = bucket.blob(DELETION_CURSOR_PATH)
    blob.upload_from_string(
        json.dumps(cursor, ensure_ascii=False, indent=2),
        content_type="application/json"
    )


@app.post("/check-deletions")
async def check_deletions(
    max_check: int = Query(default=100, description="1회 검사 최대 건수", le=500),
):
    """
    수집된 예판이 삭제되었는지 확인하고,
    삭제된 건은 Pinecone에서 제거 + 이메일 알림을 발송합니다.
    """
    start_time = datetime.now(KST)
    manifest = load_manifest()
    cursor = load_deletion_cursor()

    deleted_cases = []
    checked_count = 0
    errors = []

    # ── 판례(prec)와 해석례(expc)를 순차 검사 ──
    for target in ["prec", "expc"]:
        offset_key = f"{target}_offset"
        all_ids = list(manifest.get(target, {}).keys())

        if not all_ids:
            continue

        # 커서 위치부터 시작
        offset = cursor.get(offset_key, 0)
        if offset >= len(all_ids):
            offset = 0  # 전체 1순환 완료 → 처음부터 다시

        ids_to_check = all_ids[offset:offset + max_check - checked_count]

        for record_id in ids_to_check:
            if checked_count >= max_check:
                break

            try:
                await asyncio.sleep(API_CALL_DELAY)
                exists = await verify_case_exists(record_id, target)
                checked_count += 1

                if not exists:
                    # ── 삭제 감지! ──
                    source_type = "판례" if target == "prec" else "법령해석례"

                    # GCS에서 메타데이터 조회 (사건명, 사건번호, 세목)
                    case_meta = {"record_id": record_id, "source_type": source_type}
                    try:
                        gcs_folder = "판례" if target == "prec" else "법령해석례"
                        gcs = storage.Client()
                        bucket = gcs.bucket(BUCKET_NAME)
                        blob = bucket.blob(f"예판/{gcs_folder}/{record_id}.json")
                        if blob.exists():
                            stored = json.loads(blob.download_as_text())
                            meta = stored.get("meta", {})
                            case_meta["case_name"] = meta.get("사건명", "")
                            case_meta["case_no"] = meta.get("사건번호", "")
                            case_meta["tax_category"] = meta.get("세목분류", "")

                            # GCS 파일명에 _삭제됨 표시 추가 (원본 보존)
                            deleted_blob = bucket.blob(
                                f"예판/{gcs_folder}/{record_id}_삭제됨.json"
                            )
                            stored["meta"]["삭제감지일시"] = datetime.now(KST).isoformat()
                            stored["meta"]["삭제여부"] = True
                            deleted_blob.upload_from_string(
                                json.dumps(stored, ensure_ascii=False, indent=2),
                                content_type="application/json"
                            )
                    except Exception as e:
                        case_meta["gcs_error"] = str(e)[:100]

                    # Pinecone에서 벡터 삭제
                    pinecone_result = await remove_from_pinecone(record_id, source_type)
                    case_meta["pinecone"] = pinecone_result

                    # 매니페스트에서 제거
                    manifest.get(target, {}).pop(str(record_id), None)

                    deleted_cases.append(case_meta)

            except Exception as e:
                errors.append({
                    "target": target, "record_id": record_id,
                    "error": str(e)[:200],
                })

        # 커서 업데이트
        cursor[offset_key] = offset + len(ids_to_check)

    # ── 이메일 알림 발송 (삭제된 건이 있을 때만) ──
    email_result = None
    if deleted_cases:
        subject = f"[EEHO AI] 삭제된 예판 {len(deleted_cases)}건 감지 ({start_time.strftime('%Y-%m-%d')})"
        body_html = build_deletion_alert_html(deleted_cases)
        email_result = await send_alert_email(subject, body_html)

    # 매니페스트 & 커서 저장
    try:
        save_manifest(manifest)
        save_deletion_cursor(cursor)
    except Exception as e:
        errors.append({"save_error": str(e)[:200]})

    end_time = datetime.now(KST)

    return {
        "job": "check-deletions",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_seconds": round((end_time - start_time).total_seconds(), 1),
        "checked_count": checked_count,
        "deleted_count": len(deleted_cases),
        "deleted_cases": deleted_cases if deleted_cases else None,
        "email": email_result,
        "cursor": cursor,
        "errors": errors if errors else None,
    }


# ============================================================
# 이메일 알림 테스트 엔드포인트
# ============================================================

@app.post("/test-email")
async def test_email():
    """이메일 발송 설정이 올바른지 테스트합니다."""
    test_cases = [{
        "source_type": "판례",
        "record_id": "TEST_12345",
        "case_name": "[테스트] 양도소득세 비과세 판례",
        "case_no": "조심2025서0001",
        "tax_category": "양도소득세",
    }]

    subject = "[EEHO AI] 이메일 알림 테스트"
    body_html = build_deletion_alert_html(test_cases)
    result = await send_alert_email(subject, body_html)

    return {
        "email_config": {
            "from": ALERT_EMAIL_FROM or "(미설정)",
            "to": ALERT_EMAIL_TO,
            "app_password": "설정됨" if GMAIL_APP_PASSWORD else "(미설정)",
        },
        "result": result,
    }
