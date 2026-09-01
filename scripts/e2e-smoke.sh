#!/usr/bin/env bash
set -euo pipefail

base_url="${AGENTHUB_URL:-http://localhost:8080}"
api_key="${AGENTHUB_API_KEY:-}"
run_id="$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
temp_base="$(mktemp)"
temp_path="${temp_base}.md"
file_id=""
auth_args=()

if [[ -n "$api_key" ]]; then
  auth_args=(-H "X-API-Key: $api_key")
fi

cleanup() {
  if [[ -n "$file_id" ]]; then
    curl -sS -X DELETE "${auth_args[@]}" "$base_url/api/documents/$file_id" >/dev/null || true
  fi
  rm -f "$temp_base" "$temp_path"
}
trap cleanup EXIT

mv "$temp_base" "$temp_path"
printf '# AgentHub smoke test\n\nThe service identifier is %s. It uses Qdrant as its vector database.\n' "$run_id" >"$temp_path"

curl -fsS "$base_url/api/health/ready" >/dev/null
baseline_json="$(curl -fsS "$base_url/api/admin/stats")"
upload_json="$(curl -fsS "${auth_args[@]}" -F "file=@$temp_path;type=text/markdown" "$base_url/api/ingest/upload")"
file_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["file_id"])' <<<"$upload_json")"
status="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"$upload_json")"
thread_id="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("thread_id", ""))' <<<"$upload_json")"

duplicate_json="$(curl -fsS "${auth_args[@]}" -F "file=@$temp_path;type=text/markdown" "$base_url/api/ingest/upload")"
python3 -c 'import json,sys; data=json.load(sys.stdin); assert data["duplicate"] is True' <<<"$duplicate_json"

if [[ "$status" == "review_required" ]]; then
  curl -fsS "${auth_args[@]}" -H 'Content-Type: application/json' \
    -d "{\"thread_id\":\"$thread_id\",\"approved\":true}" "$base_url/api/ingest/review" >/dev/null
fi

curl -fsS "$base_url/api/documents" | python3 -c \
  'import json,sys; file_id=sys.argv[1]; assert any(d["file_id"] == file_id for d in json.load(sys.stdin))' "$file_id"

question_json="$(printf '{\"question\":\"Which vector database does service %s use?\"}' "$run_id")"
curl -fsS -H 'Content-Type: application/json' -d "$question_json" "$base_url/api/qa/ask" | python3 -c \
  'import json,sys; assert json.load(sys.stdin)["answer"]'

curl -fsS -X DELETE "${auth_args[@]}" "$base_url/api/documents/$file_id" >/dev/null
curl -fsS "$base_url/api/documents" | python3 -c \
  'import json,sys; file_id=sys.argv[1]; assert all(d["file_id"] != file_id for d in json.load(sys.stdin))' "$file_id"
final_json="$(curl -fsS "$base_url/api/admin/stats")"
python3 -c '
import json, sys
before, after = json.loads(sys.argv[1]), json.loads(sys.argv[2])
assert before["vector_store"]["total_vectors"] == after["vector_store"]["total_vectors"]
assert before["knowledge_graph"]["total_entities"] == after["knowledge_graph"]["total_entities"]
assert before["knowledge_graph"]["total_relations"] == after["knowledge_graph"]["total_relations"]
' "$baseline_json" "$final_json"
file_id=""
echo "E2E smoke passed: readiness, upload, dedupe, registry, QA, and delete."
