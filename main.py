from fastapi import FastAPI
import httpx
import re
import asyncio
import json
import traceback as tb
from datetime import datetime

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import storage

app = FastAPI()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}

OC = "eeho-public-raw-api"
PROJECT_ID = "project-9fb5ee59-ec65-4d2a-aa6"
BUCKET_NAME = "eeho-tax-knowledge-base-01"

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

@app.get("/")
def health():
    return {"status": "EEHO AI Engine is running"}

@app.get("/check-ip")
async def check_ip():
    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.ipify.org?format=json")
        return {"outbound_ip": resp.json()["ip"]}

@app.get("/test-pipeline")
async def test_pipeline():
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
        pipeline_log["step1"] = {
            "status": "ok", "id": prec_id, "사건명": case_name, "사건번호": case_no
        }
    except Exception as e:
        return {"step": "1", "error": str(e), "trace": tb.format_exc()[-500:]}

    await asyncio.sleep(2)

    # === STEP 2: 판례 본문 조회 ===
    try:
        # 직접 URL 구성 (상세링크 사용하지 않고 정확한 파라미터로)
        detail_url = (
            f"http://www.law.go.kr/DRF/lawService.do"
            f"?OC={OC}&target=prec&ID={prec_id}&type=JSON"
        )
        detail_data = await call_law_api(detail_url)
        body_text = json.dumps(detail_data, ensure_ascii=False, indent=2)

        # "일치하는 판례가 없습니다" 체크
        if "일치하는" in body_text and len(body_text) < 200:
            # 판례일련번호 대신 ID 필드 사용 시도
            prec_id_alt = first.get("id", prec_id)
            detail_url2 = (
                f"http://www.law.go.kr/DRF/lawService.do"
                f"?OC={OC}&target=prec&ID={prec_id_alt}&type=JSON"
            )
            detail_data = await call_law_api(detail_url2)
            body_text = json.dumps(detail_data, ensure_ascii=False, indent=2)
            pipeline_log["step2"] = {
                "status": "ok", "본문길이": len(body_text),
                "note": f"판례일련번호({prec_id}) 실패, id({prec_id_alt})로 재시도"
            }
        else:
            pipeline_log["step2"] = {"status": "ok", "본문길이": len(body_text)}
    except Exception as e:
        return {"step": "2", "error": str(e), "log": pipeline_log}

    if len(body_text) < 100:
        pipeline_log["step3"] = {"status": "skipped", "reason": f"본문 {len(body_text)}자", "preview": body_text}
        return {"pipeline": pipeline_log}

    # === STEP 3: Gemini 구조화 ===
    ai_text = ""
    try:
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.0-flash-001")
        prompt = f"""당신은 대한민국 세법 전문가입니다. 아래 판례를 5가지 항목으로 구조화하세요.
반드시 JSON만 응답하세요. 다른 텍스트 없이 JSON만 출력하세요.

{{"사실관계":"...","납세자주장":"...","과세관청주장":"...","판단근거":"...","관련법령":"..."}}

=== 판례 ===
{body_text[:15000]}"""
        response = model.generate_content(prompt)
        ai_text = response.text.strip()
        ai_clean = re.sub(r'^```json\s*', '', ai_text)
        ai_clean = re.sub(r'\s*```$', '', ai_clean)
        structured = json.loads(ai_clean)
        pipeline_log["step3"] = {"status": "ok", "structured": structured}
    except Exception as e:
        pipeline_log["step3"] = {"status": "error", "error": str(e), "raw": ai_text[:500]}
        return {"pipeline": pipeline_log}

    # === STEP 4: GCS 저장 ===
    try:
        save_data = {
            "meta": {"판례일련번호": prec_id, "사건명": case_name, "사건번호": case_no,
                     "수집일시": datetime.utcnow().isoformat(), "소스": "국가법령정보_판례"},
            "원본": detail_data,
            "구조화": structured
        }
        gcs = storage.Client()
        bucket = gcs.bucket(BUCKET_NAME)
        blob = bucket.blob(f"예판/판례/{prec_id}.json")
        blob.upload_from_string(json.dumps(save_data, ensure_ascii=False, indent=2), content_type="application/json")
        pipeline_log["step4"] = {"status": "ok", "path": f"gs://{BUCKET_NAME}/예판/판례/{prec_id}.json"}
    except Exception as e:
        pipeline_log["step4"] = {"status": "error", "error": str(e)}

    return {"pipeline": pipeline_log}
