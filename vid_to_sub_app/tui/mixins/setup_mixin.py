from __future__ import annotations

import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Sequence

from textual import work
from textual.css.query import NoMatches
from textual.widgets import Input, RichLog, Static

from vid_to_sub_app.shared.constants import ROOT_DIR

from ..helpers import (
    DEFAULT_BACKEND,
    DEFAULT_MODEL,
    DetectResult,
    ENV_WCPP_BIN,
    ENV_WCPP_MODEL,
    HF_MODEL_BASE,
    KNOWN_MODELS,
    PIP_REQUIREMENT_FILES,
    build_system_install_commands,
    detect_all,
    detect_package_manager,
    discover_ggml_models,
    packages_for_manager,
    summarize_ggml_models,
)
from ..state import db as _db

SCRIPT_DIR = ROOT_DIR


class SetupMixin:
    # ── Setup tab ─────────────────────────────────────────────────────────

    def _load_setup_inputs(self) -> None:
        pairs = [
            ("inp-build-dir", "tui.build_dir"),
            ("inp-install-dir", "tui.install_dir"),
            ("inp-model-dir", "tui.model_dir"),
        ]
        for wid, key in pairs:
            try:
                self.query_one(f"#{wid}", Input).value = _db.get(key)
            except NoMatches:
                pass

    def _save_setup_build_fields(self) -> None:
        for wid, key in [
            ("inp-build-dir", "tui.build_dir"),
            ("inp-install-dir", "tui.install_dir"),
        ]:
            v = self._val(wid)
            if v:
                _db.set(key, v)

    def _save_setup_model_fields(self) -> None:
        v = self._val("inp-model-dir")
        if v:
            _db.set("tui.model_dir", v)

    def _setup_log(self, text: str) -> None:
        try:
            self.query_one("#setup-log", RichLog).write(text)
        except NoMatches:
            pass

    def _run_cmd(self, cmd: list[str]) -> bool:
        """Run a subprocess, stream to setup-log. Returns True on success. Call from worker."""
        self.call_from_thread(self._setup_log, f"[dim]$ {' '.join(cmd)}[/]")
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout
            for line in iter(proc.stdout.readline, ""):
                s = line.rstrip()
                if s:
                    self.call_from_thread(self._setup_log, s)
            proc.stdout.close()
            rc = proc.wait()
            if rc != 0:
                self.call_from_thread(self._setup_log, f"[red]✗ exit {rc}[/]")
            return rc == 0
        except FileNotFoundError as e:
            self.call_from_thread(self._setup_log, f"[red]✗ Not found: {e}[/]")
            return False
        finally:
            if proc is not None:
                if proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
                if proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass

    def _capture_detection_state(self) -> DetectResult:
        self._detected_ggml_models = discover_ggml_models()
        results = detect_all()
        self._detect_results = results
        return results

    def _refresh_detection_from_worker(self) -> DetectResult:
        results = self._capture_detection_state()
        self.call_from_thread(self._update_detect_ui, results)
        return results

    @work(thread=True, exclusive=True, exit_on_error=False)
    def _run_detection(self) -> None:
        self._refresh_detection_from_worker()

    def _update_detect_ui(self, results: DetectResult) -> None:
        for name, (found, detail) in results.items():
            icon = "[green]✅[/]" if found else "[red]❌[/]"
            if name == "ggml-model" and self._detected_ggml_models:
                detail = summarize_ggml_models(self._detected_ggml_models)
            dshort = detail[:55] + "…" if len(detail) > 56 else detail
            try:
                self.query_one(f"#det-{name}", Static).update(
                    f"{icon} [bold]{name:<20}[/] {dshort}"
                )
            except NoMatches:
                pass

        # Auto-fill settings if not already set
        wbin = results.get("whisper-cli", (False, ""))[1]
        if results.get("whisper-cli", (False,))[0] and not _db.get(ENV_WCPP_BIN):
            _db.set(ENV_WCPP_BIN, wbin)
            os.environ[ENV_WCPP_BIN] = wbin
            for wid in ("stg-wcpp-bin", "inp-wcpp-bin"):
                try:
                    self.query_one(f"#{wid}", Input).value = wbin
                except NoMatches:
                    pass

        wmod = results.get("ggml-model", (False, ""))[1]
        if results.get("ggml-model", (False,))[0] and not _db.get(ENV_WCPP_MODEL):
            _db.set(ENV_WCPP_MODEL, wmod)
            os.environ[ENV_WCPP_MODEL] = wmod
            for wid in ("stg-wcpp-model",):
                try:
                    self.query_one(f"#{wid}", Input).value = wmod
                except NoMatches:
                    pass
        self._update_wcpp_model_status()

    def _resolved_wcpp_model_path(self) -> str:
        if manual := self._val("inp-wcpp-model"):
            return manual
        selected_model = self._sel("sel-model", DEFAULT_MODEL)
        return self._detected_ggml_models.get(selected_model, "")

    def _update_wcpp_model_status(self) -> None:
        backend = self._sel("sel-backend", DEFAULT_BACKEND)
        manual = self._val("inp-wcpp-model")
        selected_model = self._sel("sel-model", DEFAULT_MODEL)
        auto_path = self._detected_ggml_models.get(selected_model, "")

        if backend != "whisper-cpp":
            status = (
                "[dim]GGML auto-detect is used only with the whisper-cpp backend.[/]"
            )
        elif manual:
            status = f"[yellow]Manual override[/] {manual}"
        elif auto_path:
            filename = Path(auto_path).name
            status = f"[green]Auto[/] {filename} -> {auto_path}"
        elif self._detected_ggml_models:
            available = ", ".join(sorted(self._detected_ggml_models)[:4])
            extra = len(self._detected_ggml_models) - 4
            more = f" (+{extra} more)" if extra > 0 else ""
            status = (
                f"[yellow]No ggml-{selected_model}.bin detected.[/] "
                f"Available: {available}{more}"
            )
        else:
            status = (
                f"[yellow]No GGML model detected for {selected_model}.[/] "
                "Use Setup -> Download Model or set a manual override."
            )

        try:
            self.query_one("#wcpp-model-status", Static).update(status)
        except NoMatches:
            pass

    def _set_setup_status(self, message: str) -> None:
        try:
            self.query_one("#auto-setup-status", Static).update(message)
        except NoMatches:
            pass

    def _system_install_tools(self, tools: Sequence[str], reason: str) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)
        manager = detect_package_manager()
        wanted = list(dict.fromkeys(tools))

        if not manager:
            log("[yellow]⚠ No supported package manager detected for auto-install.[/]")
            self.call_from_thread(
                self._set_setup_status,
                "[yellow]Auto-install skipped: no supported package manager detected.[/]",
            )
            return False

        use_sudo = manager != "brew" and hasattr(os, "geteuid") and os.geteuid() != 0
        if use_sudo and not shutil.which("sudo"):
            log(
                "[red]✗ sudo is not available, so system package auto-install cannot run.[/]"
            )
            self.call_from_thread(
                self._set_setup_status,
                "[red]Auto-install failed: sudo is required for system packages.[/]",
            )
            return False

        commands = build_system_install_commands(manager, wanted, use_sudo=use_sudo)
        if not commands:
            log(
                f"[yellow]⚠ No package mapping for {', '.join(wanted)} on {manager}. Skipping auto-install.[/]"
            )
            return False

        log(
            f"[cyan]Auto-installing system packages via {manager} for {reason}: {', '.join(wanted)}[/]"
        )
        if use_sudo:
            log(
                "[dim]Using sudo -n; password prompts are not supported in this TUI worker.[/]"
            )
        self.call_from_thread(
            self._set_setup_status,
            f"[cyan]Auto-installing system packages via {manager}…[/]",
        )

        for cmd in commands:
            if not self._run_cmd(cmd):
                log("[red]✗ System package auto-install failed.[/]")
                self.call_from_thread(
                    self._set_setup_status,
                    "[red]System package auto-install failed. Check setup log.[/]",
                )
                self._refresh_detection_from_worker()
                return False

        log("[green]✅ System package auto-install completed.[/]")
        self._refresh_detection_from_worker()
        return True

    def _install_requirement_target(self, target: str) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)
        spec = PIP_REQUIREMENT_FILES.get(target)
        if not spec:
            log(f"[red]✗ Unknown pip install target: {target}[/]")
            return False
        filename, label = spec
        req_path = SCRIPT_DIR / filename
        if not req_path.exists():
            log(f"[red]✗ Requirement file not found: {req_path}[/]")
            return False
        self.call_from_thread(self._set_setup_status, f"[cyan]Installing {label}…[/]")
        log(f"[cyan]Installing {label}…[/]")
        ok = self._run_cmd(
            [sys.executable, "-m", "pip", "install", "-r", str(req_path)]
        )
        if ok:
            log(f"[green]✅ Installed {label}[/]")
        self._refresh_detection_from_worker()
        return ok

    def _build_whisper_cpp_sync(self) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)

        build_dir = _db.get("tui.build_dir") or str(
            Path.home() / ".cache/vid_to_sub_build"
        )
        install_dir = _db.get("tui.install_dir") or str(Path.home() / ".local/bin")
        build_root = Path(build_dir).expanduser().resolve()
        install_d = Path(install_dir).expanduser().resolve()
        repo_dir = build_root / "whisper.cpp"

        missing_tools: list[str] = []
        if not shutil.which("git"):
            missing_tools.append("git")
        if not shutil.which("cmake"):
            missing_tools.append("cmake")
        if missing_tools:
            log(
                f"[yellow]⚠ Missing build tools: {', '.join(missing_tools)} — attempting auto-install…[/]"
            )
            self._system_install_tools(
                [*missing_tools, "whisper-build"],
                "whisper.cpp build prerequisites",
            )

        if not shutil.which("git"):
            log("[red]✗ git not found in PATH — needed for cloning[/]")
            return False
        if not shutil.which("cmake"):
            log("[red]✗ cmake not found in PATH — needed for building[/]")
            return False

        build_root.mkdir(parents=True, exist_ok=True)

        if not (repo_dir / ".git").exists():
            log("[cyan]Cloning whisper.cpp (shallow)…[/]")
            ok = self._run_cmd(
                [
                    "git",
                    "clone",
                    "--depth=1",
                    "https://github.com/ggerganov/whisper.cpp",
                    str(repo_dir),
                ]
            )
            if not ok:
                log("[red]✗ Clone failed[/]")
                return False
        else:
            log("[cyan]Pulling latest whisper.cpp…[/]")
            self._run_cmd(["git", "-C", str(repo_dir), "pull", "--ff-only"])

        log("[cyan]CMake configure…[/]")
        cmake_build = repo_dir / "build"
        ok = self._run_cmd(
            [
                "cmake",
                "-B",
                str(cmake_build),
                "-DCMAKE_BUILD_TYPE=Release",
                str(repo_dir),
            ]
        )
        if not ok:
            log("[red]✗ cmake configure failed[/]")
            return False

        nproc = str(os.cpu_count() or 4)
        log(f"[cyan]Building with {nproc} threads…[/]")
        ok = self._run_cmd(
            [
                "cmake",
                "--build",
                str(cmake_build),
                "-j",
                nproc,
                "--target",
                "whisper-cli",
            ]
        )
        if not ok:
            log("[yellow]whisper-cli target not found, building all…[/]")
            ok = self._run_cmd(["cmake", "--build", str(cmake_build), "-j", nproc])
        if not ok:
            log("[red]✗ Build failed[/]")
            return False

        # Locate binary
        candidates = [
            cmake_build / "bin" / "whisper-cli",
            cmake_build / "whisper-cli",
            cmake_build / "bin" / "main",
            cmake_build / "main",
        ]
        binary = next((c for c in candidates if c.exists()), None)
        if not binary:
            log("[red]✗ Built binary not found — check cmake output above[/]")
            return False

        install_d.mkdir(parents=True, exist_ok=True)
        dest = install_d / "whisper-cli"
        try:
            shutil.copy2(str(binary), str(dest))
            os.chmod(str(dest), 0o755)
        except PermissionError:
            log(f"[red]✗ Permission denied writing to {dest}[/]")
            log(f"[yellow]  Tip: set Install dir to ~/.local/bin[/]")
            return False

        _db.set(ENV_WCPP_BIN, str(dest))
        os.environ[ENV_WCPP_BIN] = str(dest)

        def _update_bin_inputs() -> None:
            for wid in ("stg-wcpp-bin", "inp-wcpp-bin"):
                try:
                    self.query_one(f"#{wid}", Input).value = str(dest)
                except NoMatches:
                    pass

        self.call_from_thread(_update_bin_inputs)

        log(f"[green]✅ whisper-cli installed → {dest}[/]")
        self.call_from_thread(
            self._set_setup_status,
            f"[green]whisper-cli installed → {dest}[/]",
        )
        self._refresh_detection_from_worker()
        return True

    @work(thread=True, exclusive=True, exit_on_error=False, name="build-whisper")
    def _build_whisper_cpp(self) -> None:
        self._build_whisper_cpp_sync()

    def _download_model_sync(self, model_name: str) -> bool:
        log = lambda m: self.call_from_thread(self._setup_log, m)

        model_dir_s = _db.get("tui.model_dir") or str(SCRIPT_DIR / "models")
        model_dir = Path(model_dir_s).expanduser()
        if not model_dir.is_absolute():
            model_dir = SCRIPT_DIR / model_dir
        model_dir.mkdir(parents=True, exist_ok=True)

        filename = f"ggml-{model_name}.bin"
        dest = model_dir / filename

        if dest.exists():
            size = dest.stat().st_size / 1024 / 1024
            log(f"[yellow]⚠ {filename} already at {dest} ({size:.0f} MB) — skipping[/]")
            _db.set(ENV_WCPP_MODEL, str(dest))
            os.environ[ENV_WCPP_MODEL] = str(dest)
            self.call_from_thread(
                self._set_setup_status,
                f"[green]{filename} already available.[/]",
            )
            self._refresh_detection_from_worker()
            return True

        url = f"{HF_MODEL_BASE}/{filename}"
        log(f"[cyan]Downloading {filename}…[/]")
        log(f"[dim]  {url}[/]")
        self.call_from_thread(
            self._set_setup_status,
            f"[cyan]Downloading {filename}…[/]",
        )

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "vid_to_sub/2"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                total_mb = total / 1024 / 1024
                downloaded = 0
                last_pct = -1
                with open(dest, "wb") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = downloaded * 100 // total
                            if pct >= last_pct + 5:
                                log(
                                    f"  [{pct:3d}%] {downloaded / 1024 / 1024:.0f}/{total_mb:.0f} MB"
                                )
                                last_pct = pct
        except urllib.error.URLError as e:
            log(f"[red]✗ Download failed: {e}[/]")
            if dest.exists():
                dest.unlink()
            self.call_from_thread(
                self._set_setup_status,
                f"[red]Download failed for {filename}. Check setup log.[/]",
            )
            return False
        except OSError as e:
            log(f"[red]✗ Write error: {e}[/]")
            if dest.exists():
                dest.unlink()
            self.call_from_thread(
                self._set_setup_status,
                f"[red]Could not write {filename}. Check setup log.[/]",
            )
            return False

        size = dest.stat().st_size / 1024 / 1024
        _db.set(ENV_WCPP_MODEL, str(dest))
        os.environ[ENV_WCPP_MODEL] = str(dest)
        log(f"[green]✅ {filename} saved → {dest} ({size:.0f} MB)[/]")
        self.call_from_thread(
            self._set_setup_status,
            f"[green]{filename} saved → {dest}[/]",
        )
        self._refresh_detection_from_worker()
        return True

    @work(thread=True, exclusive=True, exit_on_error=False, name="download-model")
    def _download_model(self, model_name: str) -> None:
        self._download_model_sync(model_name)

    @work(thread=True, exclusive=True, exit_on_error=False, name="pip-install")
    def _pip_install(self, package: str) -> None:
        if package in PIP_REQUIREMENT_FILES:
            self._install_requirement_target(package)
            return
        self.call_from_thread(self._setup_log, f"[cyan]Installing {package}…[/]")
        ok = self._run_cmd([sys.executable, "-m", "pip", "install", package])
        if ok:
            self.call_from_thread(self._setup_log, f"[green]✅ {package} installed[/]")
        self._refresh_detection_from_worker()

    def _auto_setup_sync(self, full: bool, model_name: str) -> None:
        log = lambda m: self.call_from_thread(self._setup_log, m)
        mode_label = "full install" if full else "auto setup"
        self.call_from_thread(
            self._set_setup_status,
            f"[cyan]Running {mode_label}…[/]",
        )
        log(f"[bold cyan]Starting {mode_label}[/]")
        self._refresh_detection_from_worker()

        essential_tools = ["ffmpeg", "git", "cmake", "whisper-build"]
        self._system_install_tools(essential_tools, "default whisper.cpp pipeline")
        self._install_requirement_target("base")

        if full:
            for target in ("faster-whisper", "whisper", "whisperx"):
                self._install_requirement_target(target)

        self._build_whisper_cpp_sync()
        self._download_model_sync(model_name)
        results = self._refresh_detection_from_worker()

        missing = [
            name
            for name in ("ffmpeg", "whisper-cli", "ggml-model")
            if not results.get(name, (False, ""))[0]
        ]
        if missing:
            summary = f"[yellow]{mode_label.title()} incomplete: missing {', '.join(missing)}[/]"
            log(summary)
            self.call_from_thread(self._set_setup_status, summary)
        else:
            summary = f"[green]✅ {mode_label.title()} complete[/]"
            log(summary)
            self.call_from_thread(self._set_setup_status, summary)

    @work(thread=True, exclusive=True, exit_on_error=False, name="auto-setup")
    def _auto_setup(self, full: bool, model_name: str) -> None:
        self._auto_setup_sync(full, model_name)
