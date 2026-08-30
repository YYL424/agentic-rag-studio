"""
FastAPI 入口 — 企业知识管理系统 REST API

提供四组接口:
  1. /api/ingest   — 文档上传 & 入库 (支持 HITL 人机协同审核)
  2. /api/qa       — 智能问答 (同步 + SSE 流式)
  3. /api/admin    — 管理（统计、更新触发）
  4. /api/health   — 健康检查

现代化特性:
  - SSE 流式输出 (astream_events v2): 首 token 低延迟, 实时观察 Agent 推理
  - HITL: interrupt 挂起 → /api/ingest/review 人工审核后恢复
  - LangSmith 全链路追踪 (配置 API Key 后自动开启)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import ChangeType, DocumentChange, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.entity_resolver import EntityResolver
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService

logger = logging.getLogger("uvicorn")

vector_store: VectorStoreService | None = None
knowledge_graph: KnowledgeGraphService | None = None
workflows: dict[str, Any] = {}


def _setup_langsmith() -> None:
    """LangSmith 全链路追踪 (配置 API Key 后自动开启)"""
    if settings.langsmith_api_key and settings.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        logger.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    else:
        logger.info("LangSmith tracing disabled (set LANGSMITH_API_KEY + LANGSMITH_TRACING=true to enable)")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vector_store, knowledge_graph
    os.makedirs(settings.upload_dir, exist_ok=True)
    _setup_langsmith()
    vector_store = VectorStoreService()
    knowledge_graph = KnowledgeGraphService()
    dependency_status = {"vector_store": False, "knowledge_graph": False}
    try:
        await vector_store.init()
        dependency_status["vector_store"] = True
    except Exception as e:
        logger.warning(f"Vector store init failed: {e}")
    try:
        await knowledge_graph.init()
        dependency_status["knowledge_graph"] = True
    except Exception as e:
        logger.warning(f"Knowledge graph init failed: {e}")
    app.state.dependency_status = dependency_status
    workflows.update(
        build_knowledge_graph_workflow(vector_store=vector_store, knowledge_graph=knowledge_graph)
    )
    yield
    workflows.clear()
    await vector_store.close()
    await knowledge_graph.close()


app = FastAPI(
    title="AgentKnowledgeHub — Agentic RAG Knowledge Base",
    description="支持文档解析、知识图谱、增量更新、HITL 和 SSE 的 Agentic RAG 工程原型",
    version="2.0.0",
    lifespan=lifespan,
)


# ── Request / Response Models ────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_length)


class QuestionResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]
    retrieval_rounds: int


class IngestResponse(BaseModel):
    file_name: str
    file_id: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str
    thread_id: str = ""
    pending_review: list[dict[str, Any]] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    thread_id: str = Field(min_length=8, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    approved: bool = True
    note: str = ""


class ReviewResponse(BaseModel):
    thread_id: str
    status: str
    vectors_stored: int
    entities_stored: int


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str
    change_type: Literal["created", "modified", "deleted"] = "modified"


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    chunks_reprocessed: int
    chunks_unchanged: int
    success: bool
    processing_time_ms: float


# ── Ingest Endpoints ─────────────────────────────────────────

async def require_write_access(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """可选 API Key：本地开发可留空，部署时保护所有写接口。"""
    if settings.api_key and not (
        x_api_key and secrets.compare_digest(x_api_key, settings.api_key)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    original_name = Path(file.filename or "unknown").name
    suffix = Path(original_name).suffix.lower()
    if suffix not in DocParserAgent.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {suffix or 'none'}")

    upload_root = Path(settings.upload_dir).resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    save_path = upload_root / f"{uuid.uuid4().hex}{suffix}"
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    total = 0
    try:
        with save_path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_size_mb} MB limit",
                    )
                target.write(chunk)
    except Exception:
        save_path.unlink(missing_ok=True)
        raise
    return str(save_path), original_name


def _resolve_managed_path(file_path: str) -> str:
    """管理接口只能操作 upload_dir 内的文件，阻断任意本地文件访问。"""
    root = Path(settings.upload_dir).resolve()
    candidate = Path(file_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="file_path must be inside upload_dir") from e
    return str(resolved)

@app.post(
    "/api/ingest/upload",
    response_model=IngestResponse,
    tags=["文档入库"],
    dependencies=[Depends(require_write_access)],
)
async def upload_document(file: UploadFile = File(...)):
    """上传并解析文档，自动入库到向量库和知识图谱 (HITL 开启时需人工审核)"""
    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")
    save_path, original_name = await _save_upload(file)
    file_id = Path(save_path).name

    thread_id = f"ingest-{uuid.uuid4().hex[:12]}"
    result = await ingest_wf.ainvoke(
        {"file_paths": [save_path]},
        config={"configurable": {"thread_id": thread_id}},
    )

    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)

    # HITL: 流程在 review 节点挂起, 等待人工审核
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        pending = interrupts[0].value if interrupts else {}
        return IngestResponse(
            file_name=original_name,
            file_id=file_id,
            chunks_count=len(chunks),
            entities_count=total_entities,
            relations_count=total_relations,
            status="review_required",
            thread_id=thread_id,
            pending_review=pending.get("entities", []) + pending.get("relations", []),
        )

    return IngestResponse(
        file_name=original_name,
        file_id=file_id,
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
        thread_id=thread_id,
    )


@app.post(
    "/api/ingest/review",
    response_model=ReviewResponse,
    tags=["文档入库"],
    dependencies=[Depends(require_write_access)],
)
async def review_extractions(req: ReviewRequest):
    """HITL 人工审核: 通过或驳回挂起的知识入库流程"""
    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")

    try:
        resume_value = {"approved": req.approved} if req.approved else {"approved": False}
        if not req.approved:
            resume_value["note"] = req.note
        final = await ingest_wf.ainvoke(
            Command(resume=resume_value),
            config={"configurable": {"thread_id": req.thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resume failed: {e}")

    return ReviewResponse(
        thread_id=req.thread_id,
        status=final.get("review_note", "unknown"),
        vectors_stored=final.get("vectors_stored", 0),
        entities_stored=final.get("entities_stored", 0),
    )


@app.post(
    "/api/ingest/batch",
    response_model=list[IngestResponse],
    tags=["文档入库"],
    dependencies=[Depends(require_write_access)],
)
async def upload_batch(files: list[UploadFile] = File(...)):
    """批量上传文档"""
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status_code=413,
            detail=f"Batch exceeds {settings.max_batch_files} file limit",
        )
    results = []
    for file in files:
        resp = await upload_document(file)
        results.append(resp)
    return results


# ── QA Endpoints ─────────────────────────────────────────────

@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    """智能问答 — 混合检索 + Reranker + Self-RAG"""
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    result = await qa_wf.ainvoke(
        {"question": req.question},
        config={"configurable": {"thread_id": f"qa-{uuid.uuid4().hex[:12]}"}},
    )
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")

    return QuestionResponse(
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
        retrieval_rounds=qa_result.retrieval_rounds,
    )


@app.post("/api/qa/ask_stream", tags=["智能问答"])
async def ask_stream(req: QuestionRequest):
    """
    SSE 流式问答 — 实时输出最终答案 token
    事件格式: {"type": "token", "content": "..."} / {"type": "final", ...} / [DONE]
    """
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    thread_id = f"stream-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    async def event_generator():
        try:
            async for event in qa_wf.astream_events(
                {"question": req.question},
                version="v2",
                config=config,
            ):
                kind = event.get("event")
                # 只流式输出最终答案的 token (tags 过滤, 中间检索/评估调用不输出)
                if kind == "on_chat_model_stream":
                    tags = event.get("tags") or []
                    chunk = event.get("data", {}).get("chunk")
                    if "final_answer" in tags and chunk and chunk.content:
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content}, ensure_ascii=False)}\n\n"

            # 流式结束后读取最终状态, 输出元信息
            try:
                state = qa_wf.get_state(config)
                qa_result = state.values.get("result")
                if qa_result:
                    final_payload = {
                        "type": "final",
                        "confidence": qa_result.confidence,
                        "intent": qa_result.intent.value,
                        "retrieval_rounds": qa_result.retrieval_rounds,
                        "sources": [
                            {"source": c.source, "score": round(c.score, 4), "type": c.retrieval_type}
                            for c in qa_result.contexts
                        ],
                        "reasoning_steps": qa_result.reasoning_steps,
                    }
                    yield f"data: {json.dumps(final_payload, ensure_ascii=False)}\n\n"
            except Exception as e:
                logger.warning("get_state after stream failed: %s", e)
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Admin Endpoints ──────────────────────────────────────────

@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取系统统计信息"""
    if vector_store is None or knowledge_graph is None:
        raise HTTPException(status_code=503, detail="Storage services not initialized")
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats()
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


@app.post(
    "/api/admin/update",
    response_model=UpdateResponse,
    tags=["系统管理"],
    dependencies=[Depends(require_write_access)],
)
async def trigger_update(req: UpdateRequest):
    """手动触发知识更新"""
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    managed_path = _resolve_managed_path(req.file_path)
    change = DocumentChange(
        file_path=managed_path,
        change_type=ChangeType(req.change_type),
    )
    result = await update_wf.ainvoke(
        {"changes": [change]},
        config={"configurable": {"thread_id": f"update-{uuid.uuid4().hex[:12]}"}},
    )
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    if req.change_type == "deleted" and r.success:
        Path(managed_path).unlink(missing_ok=True)
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        chunks_reprocessed=r.chunks_reprocessed,
        chunks_unchanged=r.chunks_unchanged,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


@app.get("/api/health", tags=["系统管理"])
async def health():
    dependencies = getattr(app.state, "dependency_status", {})
    ready = bool(dependencies) and all(dependencies.values())
    return {
        "status": "ok" if ready else "degraded",
        "service": "AgentKnowledgeHub",
        "version": "2.0.0",
        "dependencies": dependencies,
    }


@app.get("/api/health/live", tags=["系统管理"])
async def liveness():
    return {"status": "ok"}


@app.get("/api/health/ready", tags=["系统管理"])
async def readiness():
    dependencies = getattr(app.state, "dependency_status", {})
    if not dependencies or not all(dependencies.values()):
        raise HTTPException(status_code=503, detail={"dependencies": dependencies})
    return {"status": "ok", "dependencies": dependencies}


@app.get("/", response_class=HTMLResponse, tags=["前端"])
async def chat_ui():
    return CHAT_HTML


CHAT_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentKnowledgeHub — Agentic RAG 知识库</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0f172a;color:#e2e8f0;height:100vh;display:flex}
.sidebar{width:320px;background:#1e293b;display:flex;flex-direction:column;border-right:1px solid #334155}
.sidebar-header{padding:20px;border-bottom:1px solid #334155}
.sidebar-header h1{font-size:18px;color:#f1f5f9;margin-bottom:4px}
.sidebar-header p{font-size:12px;color:#94a3b8}
.api-key{margin:12px 16px 0;padding:9px 11px;width:calc(100% - 32px);background:#0f172a;border:1px solid #475569;border-radius:8px;color:#e2e8f0;outline:none}
.api-key:focus{border-color:#818cf8}
.upload-zone{display:block;margin:16px;padding:24px;border:2px dashed #475569;border-radius:12px;text-align:center;cursor:pointer;transition:all .2s;flex-shrink:0}
.upload-zone:hover{border-color:#818cf8;background:#1e1b4b}
.upload-zone input{display:none}
.upload-zone .icon{font-size:32px;margin-bottom:8px}
.upload-zone .text{font-size:13px;color:#94a3b8}
.file-list{flex:1;overflow-y:auto;padding:0 16px;font-size:12px}
.file-item{display:flex;align-items:center;justify-content:space-between;padding:8px 12px;margin:4px 0;background:#0f172a;border-radius:8px}
.file-item .name{color:#e2e8f0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1}
.file-item .status{font-size:11px;padding:2px 8px;border-radius:10px}
.file-item .status.ok{background:#065f46;color:#6ee7b7}
.file-item .status.err{background:#7f1d1d;color:#fca5a5}
.file-item .status.pending{background:#78350f;color:#fbbf24}
.upload-zone.busy{pointer-events:none;opacity:.65}
.stats-bar{padding:16px;border-top:1px solid #334155;font-size:12px;color:#94a3b8}
.stats-bar span{color:#818cf8;font-weight:600}
.main{flex:1;display:flex;flex-direction:column}
.chat-area{flex:1;overflow-y:auto;padding:24px}
.msg{margin-bottom:20px;max-width:80%}
.msg.user{margin-left:auto}
.msg.user .bubble{background:#4f46e5;border-radius:16px 16px 4px 16px}
.msg.agent .bubble{background:#1e293b;border-radius:16px 16px 16px 4px}
.msg .bubble{padding:12px 16px;line-height:1.6;font-size:14px}
.msg .bubble .label{font-size:11px;color:#818cf8;margin-bottom:4px}
.msg .meta{font-size:11px;color:#64748b;margin-top:6px;padding:0 4px}
.empty-state{text-align:center;padding:80px 20px;color:#64748b}
.empty-state .icon{font-size:48px;margin-bottom:16px}
.empty-state h2{font-size:20px;color:#94a3b8;margin-bottom:8px}
.empty-state p{font-size:14px}
.input-area{padding:20px;border-top:1px solid #334155;display:flex;gap:12px}
.input-area input{flex:1;padding:12px 16px;background:#1e293b;border:1px solid #475569;border-radius:12px;color:#e2e8f0;font-size:14px;outline:none}
.input-area input:focus{border-color:#818cf8}
.input-area button{padding:12px 24px;background:#4f46e5;color:white;border:none;border-radius:12px;font-size:14px;cursor:pointer;transition:all .2s}
.input-area button:hover{background:#6366f1}
.input-area button:disabled{background:#334155;cursor:not-allowed}
.loading{display:inline-block;width:20px;height:20px;border:2px solid #64748b;border-top-color:#818cf8;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:8px}
@keyframes spin{to{transform:rotate(360deg)}}
.cursor{display:inline-block;width:8px;height:16px;background:#818cf8;vertical-align:middle;animation:blink 1s infinite}
@keyframes blink{50%{opacity:0}}
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header">
    <h1>🤖 AgentKnowledgeHub</h1>
    <p>Agentic RAG 知识库 v2.0</p>
  </div>
  <input class="api-key" type="password" id="apiKeyInput" placeholder="写接口 API Key（可选）" autocomplete="off">
  <label class="upload-zone" id="uploadZone" for="fileInput">
    <div class="icon">📁</div>
    <div class="text">点击或拖拽上传文档<br>支持 PDF / Word / PPT / Excel / 图片 / TXT</div>
    <input type="file" id="fileInput" multiple accept=".pdf,.doc,.docx,.ppt,.pptx,.xls,.xlsx,.csv,.txt,.md,.png,.jpg,.jpeg">
  </label>
  <div class="file-list" id="fileList">
    <div style="text-align:center;color:#475569;padding:20px">暂无文档</div>
  </div>
  <div class="stats-bar" id="statsBar">📊 加载中...</div>
</div>
<div class="main">
  <div class="chat-area" id="chatArea">
    <div class="empty-state">
      <div class="icon">💬</div>
      <h2>知识问答助手</h2>
      <p>上传文档后，用自然语言提问</p>
    </div>
  </div>
  <div class="input-area">
    <input type="text" id="questionInput" placeholder="输入你的问题，如：张三在哪个公司？负责什么？" onkeydown="if(event.key==='Enter')ask()">
    <button id="askBtn" onclick="ask()">发送</button>
  </div>
</div>
<script>
const API = '';
let uploadedFiles = [];

const apiKeyInput = document.getElementById('apiKeyInput');
apiKeyInput.value = sessionStorage.getItem('agenthub_api_key') || '';
apiKeyInput.oninput = () => sessionStorage.setItem('agenthub_api_key', apiKeyInput.value);

function writeHeaders() {
  const key = apiKeyInput.value.trim();
  return key ? { 'X-API-Key': key } : {};
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, ch => ({
    '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;'
  })[ch]);
}

document.getElementById('uploadZone').ondragover = e => { e.preventDefault(); e.currentTarget.style.borderColor='#818cf8'; };
document.getElementById('uploadZone').ondragleave = e => e.currentTarget.style.borderColor='#475569';
document.getElementById('uploadZone').ondrop = e => {
  e.preventDefault();
  e.currentTarget.style.borderColor='#475569';
  void uploadFiles(Array.from(e.dataTransfer.files));
};
document.getElementById('fileInput').onchange = e => {
  const files = Array.from(e.target.files);
  e.target.value = '';
  void uploadFiles(files);
};

async function uploadFiles(files) {
  if (!files.length) return;
  const uploadZone = document.getElementById('uploadZone');
  const fileInput = document.getElementById('fileInput');
  uploadZone.classList.add('busy');
  fileInput.disabled = true;
  for (const f of files) {
    const item = { name: f.name, uploading: true, ok: false, pending: false };
    uploadedFiles.push(item);
    renderFiles();
    const startedAt = Date.now();
    addMsg('agent', '⏳ 正在解析并入库: ' + f.name + '\n大文档需要逐段抽取知识，请保持页面打开。');
    const form = new FormData();
    form.append('file', f);
    try {
      const res = await fetch(API + '/api/ingest/upload', {
        method: 'POST',
        headers: writeHeaders(),
        body: form
      });
      let data = {};
      try { data = await res.json(); } catch(e) {}
      const elapsed = ((Date.now() - startedAt) / 1000).toFixed(1);
      if (res.ok) {
        const pending = data.status === 'review_required';
        Object.assign(item, { name: data.file_name, fileId: data.file_id, chunks: data.chunks_count, uploading: false, ok: !pending, pending });
        if (pending) {
          addMsg('agent', '📄 文档 ' + data.file_name + ' 已解析，等待人工审核（thread: ' + data.thread_id.slice(0,8) + '）\n分块: ' + data.chunks_count + ' | 实体: ' + data.entities_count + ' | 关系: ' + data.relations_count + ' | 用时: ' + elapsed + ' 秒');
        } else {
          addMsg('agent', '📄 文档 ' + data.file_name + ' 已入库\n分块: ' + data.chunks_count + ' | 实体: ' + data.entities_count + ' | 关系: ' + data.relations_count + ' | 用时: ' + elapsed + ' 秒');
        }
      } else {
        Object.assign(item, { uploading: false, ok: false });
        const detail = typeof data.detail === 'string' ? data.detail : ('HTTP ' + res.status);
        addMsg('agent', '❌ 上传失败: ' + f.name + '\n原因: ' + detail);
      }
    } catch(e) {
      Object.assign(item, { uploading: false, ok: false });
      addMsg('agent', '❌ 网络错误: ' + f.name + '\n原因: ' + (e.message || '无法连接 API 服务'));
    }
    renderFiles();
    await loadStats();
  }
  uploadZone.classList.remove('busy');
  fileInput.disabled = false;
}

function renderFiles() {
  const el = document.getElementById('fileList');
  if (!uploadedFiles.length) { el.innerHTML = '<div style="text-align:center;color:#475569;padding:20px">暂无文档</div>'; return; }
  el.innerHTML = uploadedFiles.map(f => '<div class="file-item"><span class="name" title="'+escapeHtml(f.name)+'">'+escapeHtml(f.name)+'</span><span class="status '+((f.uploading||f.pending)?'pending':(f.ok?'ok':'err'))+'">'+(f.uploading?'⏳ 解析中':(f.pending?'⏳ 待审核':(f.ok?'✓ 已入库':'✗ 失败')))+'</span></div>').join('');
}

async function ask() {
  const inp = document.getElementById('questionInput');
  const q = inp.value.trim();
  if (!q) return;
  addMsg('user', q);
  inp.value = '';
  const btn = document.getElementById('askBtn');
  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span>思考中';

  try {
    const res = await fetch(API + '/api/qa/ask_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    if (!res.ok || !res.body) { await fallbackAsk(q, btn); return; }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let answerText = '';
    const bubbleDiv = document.createElement('div');
    bubbleDiv.className = 'msg agent';
    bubbleDiv.innerHTML = '<div class="bubble"><div class="label">🤖 AI 助手</div><span class="content"></span><span class="cursor"></span></div>';
    const area = document.getElementById('chatArea');
    const empty = area.querySelector('.empty-state');
    if (empty) empty.remove();
    area.appendChild(bubbleDiv);
    const contentEl = bubbleDiv.querySelector('.content');

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 2);
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6);
        if (payload === '[DONE]') continue;
        try {
          const evt = JSON.parse(payload);
          if (evt.type === 'token') {
            answerText += evt.content;
            contentEl.textContent = answerText;
            area.scrollTop = area.scrollHeight;
          } else if (evt.type === 'final') {
            const sources = (evt.sources||[]).map(s => s.source).join(', ') || '';
            const rounds = evt.retrieval_rounds > 1 ? ' | 🔁 重检 ' + evt.retrieval_rounds + ' 轮' : '';
            bubbleDiv.querySelector('.cursor').remove();
            const meta = document.createElement('div');
            meta.className = 'meta';
            meta.textContent = '🎯 置信度: ' + ((evt.confidence||0)*100).toFixed(0) + '% | 意图: ' + (evt.intent||'') + rounds + (sources?' | 来源: '+sources:'');
            bubbleDiv.appendChild(meta);
          }
        } catch(e) {}
      }
    }
    bubbleDiv.querySelector('.cursor')?.remove();
    if (!answerText) { bubbleDiv.remove(); await fallbackAsk(q, btn); return; }
  } catch(e) {
    await fallbackAsk(q, btn);
  }
  btn.disabled = false;
  btn.textContent = '发送';
  loadStats();
}

async function fallbackAsk(q, btn) {
  try {
    const res = await fetch(API + '/api/qa/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    if (res.ok) {
      const sources = data.sources?.map(s => s.source).join(', ') || '';
      addMsg('agent', data.answer, '🎯 置信度: ' + (data.confidence*100).toFixed(0) + '% | 意图: ' + data.intent + (sources?' | 来源: '+sources:''));
    } else {
      addMsg('agent', '❌ 问答失败，请确认已上传文档');
    }
  } catch(e) {
    addMsg('agent', '❌ 请求失败，请检查服务是否正常');
  }
  btn.disabled = false;
  btn.textContent = '发送';
}

function addMsg(role, text, metaText='') {
  const area = document.getElementById('chatArea');
  const empty = area.querySelector('.empty-state');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = role === 'agent' ? '🤖 AI 助手' : '👤 你';
  const content = document.createElement('div');
  content.style.whiteSpace = 'pre-wrap';
  content.textContent = text;
  bubble.append(label, content);
  if (metaText) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = metaText;
    bubble.appendChild(meta);
  }
  div.appendChild(bubble);
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

async function loadStats() {
  try {
    const res = await fetch(API + '/api/admin/stats');
    const data = await res.json();
    document.getElementById('statsBar').textContent = '📊 向量: ' + (data.vector_store?.total_vectors||0) + ' 条 | 实体: ' + (data.knowledge_graph?.total_entities||0) + ' | 关系: ' + (data.knowledge_graph?.total_relations||0);
  } catch(e) {}
}

loadStats();
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
