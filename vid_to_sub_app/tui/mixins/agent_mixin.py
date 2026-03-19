from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from textual import work
from textual.css.query import NoMatches
from textual.widgets import Button, RichLog, Static, TextArea

from ..helpers import (
    DEFAULT_MODEL,
    ENV_AGENT_KEY,
    ENV_AGENT_MOD,
    ENV_AGENT_URL,
    ENV_TRANS_KEY,
    ENV_TRANS_MOD,
    ENV_TRANS_URL,
    KNOWN_MODELS,
    PIP_REQUIREMENT_FILES,
    extract_json_payload,
    normalize_chat_endpoint,
)
from ..state import db as _db


class AgentMixin:
    def _effective_agent_config(self) -> tuple[str, str, str]:
        model = (
            _db.get(ENV_AGENT_MOD)
            or _db.get(ENV_TRANS_MOD)
            or os.environ.get(ENV_AGENT_MOD, "")
            or os.environ.get(ENV_TRANS_MOD, "")
        )
        base_url = (
            _db.get(ENV_AGENT_URL)
            or _db.get(ENV_TRANS_URL)
            or os.environ.get(ENV_AGENT_URL, "")
            or os.environ.get(ENV_TRANS_URL, "")
        )
        api_key = (
            _db.get(ENV_AGENT_KEY)
            or _db.get(ENV_TRANS_KEY)
            or os.environ.get(ENV_AGENT_KEY, "")
            or os.environ.get(ENV_TRANS_KEY, "")
        )
        return model, base_url, api_key

    def _update_agent_config_status(self) -> None:
        model, base_url, api_key = self._effective_agent_config()
        status = (
            f"[dim]Effective agent model:[/] {model or '(missing)'}   "
            f"[dim]base URL:[/] {base_url or '(missing)'}   "
            f"[dim]API key:[/] {'set' if api_key else 'missing'}"
        )
        try:
            self.query_one("#agent-config", Static).update(status)
        except NoMatches:
            pass

    def _set_agent_status(self, message: str) -> None:
        try:
            self.query_one("#agent-status", Static).update(message)
        except NoMatches:
            pass

    def _agent_log_write(self, message: str) -> None:
        try:
            self.query_one("#agent-log", RichLog).write(message)
        except NoMatches:
            pass

    def _update_agent_apply_state(self) -> None:
        try:
            button = self.query_one("#btn-agent-apply", Button)
            button.disabled = not bool(
                self._agent_plan and self._agent_plan.get("actions")
            )
        except NoMatches:
            pass

    def _clear_agent_ui(self) -> None:
        self._agent_plan = None
        try:
            self.query_one("#agent-prompt", TextArea).text = ""
            self.query_one("#agent-log", RichLog).clear()
        except NoMatches:
            pass
        self._set_agent_status("[dim]No agent plan yet.[/]")
        self._update_agent_apply_state()
        self._update_agent_config_status()

    def _normalize_agent_plan(self, payload: dict[str, Any]) -> dict[str, Any]:
        summary = str(payload.get("summary") or "No summary provided.").strip()
        analysis = str(payload.get("analysis") or "").strip()
        normalized_actions: list[dict[str, Any]] = []
        raw_actions = payload.get("actions")
        known_job_ids = {int(job["id"]) for job in _db.get_jobs()}

        if isinstance(raw_actions, list):
            for raw_action in raw_actions:
                if not isinstance(raw_action, dict):
                    continue
                action_type = str(raw_action.get("type") or "").strip()
                if action_type == "run_detection":
                    normalized_actions.append({"type": "run_detection"})
                elif action_type == "auto_setup":
                    mode = str(raw_action.get("mode") or "essential").strip().lower()
                    model = str(raw_action.get("model") or DEFAULT_MODEL).strip()
                    normalized_actions.append(
                        {
                            "type": "auto_setup",
                            "mode": "full" if mode == "full" else "essential",
                            "model": model if model in KNOWN_MODELS else DEFAULT_MODEL,
                        }
                    )
                elif action_type == "build_whisper_cpp":
                    normalized_actions.append({"type": "build_whisper_cpp"})
                elif action_type == "download_model":
                    model = str(raw_action.get("model") or DEFAULT_MODEL).strip()
                    normalized_actions.append(
                        {
                            "type": "download_model",
                            "model": model if model in KNOWN_MODELS else DEFAULT_MODEL,
                        }
                    )
                elif action_type == "pip_install":
                    target = str(raw_action.get("target") or "").strip()
                    if target in PIP_REQUIREMENT_FILES:
                        normalized_actions.append(
                            {"type": "pip_install", "target": target}
                        )
                elif action_type in {
                    "refresh_history",
                    "kill_active_run",
                }:
                    normalized_actions.append({"type": action_type})
                elif action_type in {
                    "load_history_job",
                    "rerun_history_job",
                    "delete_history_job",
                }:
                    try:
                        job_id = int(raw_action.get("job_id"))
                    except (TypeError, ValueError):
                        continue
                    if job_id in known_job_ids:
                        normalized_actions.append(
                            {"type": action_type, "job_id": job_id}
                        )

        return {
            "summary": summary,
            "analysis": analysis,
            "actions": normalized_actions,
        }

    def _render_agent_plan(self, plan: dict[str, Any]) -> None:
        try:
            log = self.query_one("#agent-log", RichLog)
            log.clear()
        except NoMatches:
            log = None

        if log is not None:
            log.write(f"[bold]Summary[/] {plan['summary']}")
            if plan.get("analysis"):
                log.write(f"\n[bold]Analysis[/] {plan['analysis']}")
            actions = plan.get("actions") or []
            if actions:
                log.write("\n[bold]Actions[/]")
                for idx, action in enumerate(actions, start=1):
                    line = f"{idx}. {action['type']}"
                    if action["type"] == "auto_setup":
                        line += f" mode={action['mode']} model={action['model']}"
                    elif action["type"] == "download_model":
                        line += f" model={action['model']}"
                    elif action["type"] == "pip_install":
                        line += f" target={action['target']}"
                    elif action["type"].endswith("_history_job"):
                        line += f" job_id={action['job_id']}"
                    log.write(line)
            else:
                log.write("\n[yellow]No executable actions returned.[/]")

        if plan.get("actions"):
            self._set_agent_status(
                "[green]Agent plan ready. Review and apply when ready.[/]"
            )
        else:
            self._set_agent_status("[yellow]Agent returned guidance only.[/]")
        self._update_agent_apply_state()

    def _agent_context(self) -> dict[str, Any]:
        detections = {
            name: {"found": found, "detail": detail}
            for name, (found, detail) in self._detect_results.items()
        }
        recent_jobs = [
            {
                "id": job["id"],
                "created_at": job["created_at"],
                "video_path": job["video_path"],
                "backend": job["backend"],
                "model": job["model"],
                "status": job["status"],
                "language": job.get("language"),
                "target_lang": job.get("target_lang"),
                "error": job.get("error"),
            }
            for job in _db.get_jobs(limit=20)
        ]
        active_run = {
            "is_running": bool(self._active_worker and self._active_worker.is_running),
            "started_at": self._run_started_at,
            "elapsed_sec": (
                round(time.monotonic() - self._run_started_at, 3)
                if self._run_started_at is not None
                else None
            ),
            "total_found": self._run_total_found,
            "total_queued": self._run_total_queued,
            "completed": self._run_completed,
            "failed": self._run_failed,
            "active_jobs": [
                {
                    "executor": job.executor,
                    "video_path": job.video_path,
                    "started_at": job.started_at,
                    "run_for_sec": round(time.monotonic() - job.started_at, 3),
                }
                for job in self._active_jobs.values()
            ],
        }
        selected_job = self._selected_history_job()
        return {
            "detections": detections,
            "active_run": active_run,
            "selected_history_job": (
                {
                    "id": selected_job["id"],
                    "video_path": selected_job["video_path"],
                    "status": selected_job["status"],
                }
                if selected_job
                else None
            ),
            "recent_jobs": recent_jobs,
            "execution_mode": self._sel(
                "sel-execution-mode", _db.get("tui.execution_mode") or "local"
            ),
            "remote_resources": [profile.name for profile in self._remote_resources],
            "selected_download_model": self._sel("sel-dl-model", DEFAULT_MODEL),
            "selected_backend": self._sel("sel-backend", DEFAULT_BACKEND),
            "selected_model": self._sel("sel-model", DEFAULT_MODEL),
            "known_models": KNOWN_MODELS,
            "allowed_actions": [
                {"type": "run_detection"},
                {
                    "type": "auto_setup",
                    "mode": "essential|full",
                    "model": "KNOWN_MODELS",
                },
                {"type": "build_whisper_cpp"},
                {"type": "download_model", "model": "KNOWN_MODELS"},
                {
                    "type": "pip_install",
                    "target": list(PIP_REQUIREMENT_FILES),
                },
                {"type": "refresh_history"},
                {"type": "load_history_job", "job_id": "recent_jobs[].id"},
                {"type": "rerun_history_job", "job_id": "recent_jobs[].id"},
                {"type": "delete_history_job", "job_id": "recent_jobs[].id"},
                {"type": "kill_active_run"},
            ],
        }

    def _request_agent_plan(self) -> None:
        try:
            prompt = self.query_one("#agent-prompt", TextArea).text.strip()
            self.query_one("#agent-log", RichLog).clear()
        except NoMatches:
            return
        if not prompt:
            self._set_agent_status("[red]Enter a request for the agent first.[/]")
            return
        self._agent_plan = None
        self._update_agent_apply_state()
        self._set_agent_status("[cyan]Requesting agent plan…[/]")
        self._update_agent_config_status()
        self._fetch_agent_plan(prompt)

    @work(thread=True, exclusive=True, exit_on_error=False, name="agent-plan")
    def _fetch_agent_plan(self, prompt: str) -> None:
        log = lambda m: self.call_from_thread(self._agent_log_write, m)
        model, base_url, api_key = self._effective_agent_config()
        if not model or not base_url or not api_key:
            self.call_from_thread(
                self._set_agent_status,
                "[red]Agent settings are incomplete. Configure agent or translation API settings first.[/]",
            )
            return

        endpoint = normalize_chat_endpoint(base_url)
        context = self._agent_context()
        payload = {
            "model": model,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a bounded operations agent for the vid_to_sub TUI. "
                        "Return JSON only with keys summary, analysis, actions. "
                        "Actions must be an array using only these types: "
                        "run_detection, auto_setup, build_whisper_cpp, download_model, pip_install, "
                        "refresh_history, load_history_job, rerun_history_job, delete_history_job, kill_active_run. "
                        "Never invent shell commands or unsupported action types. "
                        "Prefer empty actions when the request is informational. "
                        "History-destructive or run-control actions must be conservative and should only be proposed "
                        "when clearly asked for or strongly justified by the current context."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"request": prompt, "context": context},
                        ensure_ascii=False,
                    ),
                },
            ],
        }

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "vid_to_sub/agent",
            },
        )

        log(f"[dim]POST {endpoint}[/]")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            self.call_from_thread(
                self._set_agent_status,
                f"[red]Agent API HTTP {exc.code}. Check agent log.[/]",
            )
            log(f"[red]HTTP {exc.code}[/] {body}")
            return
        except urllib.error.URLError as exc:
            self.call_from_thread(
                self._set_agent_status,
                f"[red]Agent API request failed: {exc.reason}[/]",
            )
            return
        except json.JSONDecodeError as exc:
            self.call_from_thread(
                self._set_agent_status,
                "[red]Agent API returned invalid JSON.[/]",
            )
            log(f"[red]JSON decode failed[/] {exc}")
            return

        try:
            message = response_payload["choices"][0]["message"]["content"]
            if isinstance(message, list):
                message = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in message
                )
            plan = self._normalize_agent_plan(extract_json_payload(str(message)))
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            self.call_from_thread(
                self._set_agent_status,
                "[red]Could not parse the agent response as a valid plan. Check agent log.[/]",
            )
            log(f"[red]Raw response[/] {response_payload}")
            log(f"[red]Parse error[/] {exc}")
            return

        self._agent_plan = plan
        self.call_from_thread(self._render_agent_plan, plan)

    def _apply_agent_plan(self) -> None:
        if not self._agent_plan or not self._agent_plan.get("actions"):
            self._set_agent_status(
                "[yellow]There is no executable agent plan to apply.[/]"
            )
            return
        self._set_agent_status("[cyan]Applying agent plan…[/]")
        self._execute_agent_plan(list(self._agent_plan["actions"]))

    @work(thread=True, exclusive=True, exit_on_error=False, name="agent-apply")
    def _execute_agent_plan(self, actions: list[dict[str, Any]]) -> None:
        log = lambda m: self.call_from_thread(self._agent_log_write, m)
        for idx, action in enumerate(actions, start=1):
            action_type = action["type"]
            log(f"[bold cyan]Step {idx}/{len(actions)}[/] {action_type}")
            if action_type == "run_detection":
                self._refresh_detection_from_worker()
            elif action_type == "auto_setup":
                self._auto_setup_sync(
                    action.get("mode") == "full",
                    str(action.get("model") or DEFAULT_MODEL),
                )
            elif action_type == "build_whisper_cpp":
                self._build_whisper_cpp_sync()
            elif action_type == "download_model":
                self._download_model_sync(str(action.get("model") or DEFAULT_MODEL))
            elif action_type == "pip_install":
                target = str(action.get("target") or "")
                if target in PIP_REQUIREMENT_FILES:
                    self._install_requirement_target(target)
            elif action_type == "load_history_job":
                job_id = int(action["job_id"])
                self.call_from_thread(self._load_history_job, job_id)
            elif action_type == "rerun_history_job":
                job_id = int(action["job_id"])
                self.call_from_thread(self._rerun_history_job, job_id)
            elif action_type == "delete_history_job":
                job_id = int(action["job_id"])
                _db.delete_job(job_id)
                self.call_from_thread(self._refresh_history)
            elif action_type == "kill_active_run":
                self.call_from_thread(self.action_kill)

        self._refresh_detection_from_worker()
        self.call_from_thread(
            self._set_agent_status,
            "[green]Agent plan applied. Review setup and logs.[/]",
        )
