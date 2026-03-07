from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils.io import ensure_dir


def _load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_float(value: str | None, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_csv_ints(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return list(default)
    out: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            out.append(int(chunk))
        except ValueError:
            continue
    return out or list(default)


def _parse_csv_strs(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return list(default)
    out: list[str] = []
    for chunk in value.split(","):
        chunk = chunk.strip().upper()
        if not chunk:
            continue
        out.append(chunk)
    return out or list(default)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """
    Simple YAML parser for flat `key: value` settings.
    Keeps dependencies minimal for this project environment.
    """
    out: dict[str, Any] = {}
    if not path.exists():
        return out

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [x.strip().strip('"').strip("'") for x in value[1:-1].split(",") if x.strip()]
            parsed: list[Any] = []
            for item in items:
                try:
                    parsed.append(int(item))
                    continue
                except ValueError:
                    pass
                try:
                    parsed.append(float(item))
                    continue
                except ValueError:
                    pass
                parsed.append(item)
            out[key] = parsed
            continue
        if value.lower() in {"true", "false"}:
            out[key] = value.lower() == "true"
            continue
        try:
            out[key] = int(value)
            continue
        except ValueError:
            pass
        try:
            out[key] = float(value)
            continue
        except ValueError:
            pass
        out[key] = value.strip('"').strip("'")
    return out


@dataclass
class Settings:
    project_root: Path
    paper_root: Path

    config_dir: Path
    state_dir: Path
    logs_dir: Path
    reports_dir: Path

    env_file: Path
    settings_yaml: Path

    paper_mode: bool = True
    binance_mode: str = "marketdata_only"
    start_equity_eur: float = 320.0

    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    binance_base_url: str = "https://api.binance.com"

    telegram_token: str | None = None
    telegram_chat_id: str | None = None

    daily_summary_hour_utc: int = 21
    poll_seconds: int = 20
    log_level: str = "INFO"

    fee_bps: float = 7.0
    slippage_bps_choices: list[int] = field(default_factory=lambda: [2, 5, 7, 10, 12])
    reconciliation_lookback_bars: int = 72

    symbol_error_quarantine_threshold: int = 4
    symbol_quarantine_minutes: int = 120

    api_circuit_fail_threshold: int = 5
    api_circuit_cooldown_sec: int = 600

    max_bars_fetch: int = 1200
    require_repaired_posture_pack: bool = True
    repaired_posture_freeze_dir: str = (
        "/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126"
    )
    repaired_active_subset_csv: str = (
        "/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_subset.csv"
    )
    repaired_active_params_dir: str = (
        "/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_params"
    )
    paper_symbol_allowlist: list[str] = field(default_factory=lambda: ["SOLUSDT"])
    required_active_strategy_id: str = "M1_ENTRY_ONLY_PASSIVE_BASELINE"
    repaired_contract_defer_exit_to_next_bar: bool = True

    @property
    def unsafe_live_endpoint(self) -> bool:
        base = (self.binance_base_url or "").lower()
        return "binance.com" in base and "testnet" not in base

    @property
    def allow_testnet_reset(self) -> bool:
        return (
            self.paper_mode
            and self.binance_mode.lower() == "testnet"
            and not self.unsafe_live_endpoint
            and bool(self.binance_api_key)
            and bool(self.binance_api_secret)
        )


def load_settings(project_root: Path | None = None) -> Settings:
    root = (project_root or Path("/root/analysis/0.87")).resolve()
    paper_root = root / "paper_trading"
    config_dir = ensure_dir(paper_root / "config")
    state_dir = ensure_dir(paper_root / "state")
    logs_dir = ensure_dir(paper_root / "logs")
    reports_dir = ensure_dir(paper_root / "reports")

    env_file = config_dir / ".env"
    settings_yaml = config_dir / "settings.yaml"
    live_env_file = Path("/root/live/long_universe/.env")

    _load_dotenv(root / ".env")
    _load_dotenv(env_file)
    _load_dotenv(live_env_file)

    yaml_cfg = _read_simple_yaml(settings_yaml)

    def env_or_yaml(name: str, yaml_key: str | None = None, default: Any = None) -> Any:
        if name in os.environ:
            return os.environ.get(name)
        if yaml_key and yaml_key in yaml_cfg:
            return yaml_cfg[yaml_key]
        return default

    slippage_vals = env_or_yaml("SLIPPAGE_BPS_CHOICES", "slippage_bps_choices", [2, 5, 7, 10, 12])
    if isinstance(slippage_vals, str):
        slippage_bps_choices = _parse_csv_ints(slippage_vals, [2, 5, 7, 10, 12])
    elif isinstance(slippage_vals, list):
        parsed: list[int] = []
        for item in slippage_vals:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        slippage_bps_choices = parsed or [2, 5, 7, 10, 12]
    else:
        slippage_bps_choices = [2, 5, 7, 10, 12]

    allowlist_val = env_or_yaml("PAPER_SYMBOL_ALLOWLIST", "paper_symbol_allowlist", "SOLUSDT")
    if isinstance(allowlist_val, list):
        allowlist = [str(x).strip().upper() for x in allowlist_val if str(x).strip()]
    else:
        allowlist = _parse_csv_strs(str(allowlist_val), ["SOLUSDT"])

    settings = Settings(
        project_root=root,
        paper_root=paper_root,
        config_dir=config_dir,
        state_dir=state_dir,
        logs_dir=logs_dir,
        reports_dir=reports_dir,
        env_file=env_file,
        settings_yaml=settings_yaml,
        paper_mode=_parse_bool(str(env_or_yaml("PAPER_MODE", "paper_mode", "true")), True),
        binance_mode=str(env_or_yaml("BINANCE_MODE", "binance_mode", "marketdata_only")),
        start_equity_eur=_parse_float(str(env_or_yaml("START_EQUITY_EUR", "start_equity_eur", 320.0)), 320.0),
        binance_api_key=env_or_yaml("BINANCE_API_KEY", "binance_api_key", None),
        binance_api_secret=env_or_yaml("BINANCE_API_SECRET", "binance_api_secret", None),
        binance_base_url=str(env_or_yaml("BINANCE_BASE_URL", "binance_base_url", "https://api.binance.com")),
        telegram_token=env_or_yaml("TELEGRAM_TOKEN", "telegram_token", None)
        or env_or_yaml("TELEGRAM_BOT_TOKEN", default=None),
        telegram_chat_id=env_or_yaml("TELEGRAM_CHAT_ID", "telegram_chat_id", None),
        daily_summary_hour_utc=_parse_int(str(env_or_yaml("DAILY_SUMMARY_HOUR_UTC", "daily_summary_hour_utc", 21)), 21),
        poll_seconds=_parse_int(str(env_or_yaml("POLL_SECONDS", "poll_seconds", 20)), 20),
        log_level=str(env_or_yaml("LOG_LEVEL", "log_level", "INFO")),
        fee_bps=_parse_float(str(env_or_yaml("FEE_BPS", "fee_bps", 7.0)), 7.0),
        slippage_bps_choices=slippage_bps_choices,
        reconciliation_lookback_bars=_parse_int(
            str(env_or_yaml("RECONCILIATION_LOOKBACK_BARS", "reconciliation_lookback_bars", 72)),
            72,
        ),
        symbol_error_quarantine_threshold=_parse_int(
            str(env_or_yaml("SYMBOL_ERROR_QUARANTINE_THRESHOLD", "symbol_error_quarantine_threshold", 4)),
            4,
        ),
        symbol_quarantine_minutes=_parse_int(
            str(env_or_yaml("SYMBOL_QUARANTINE_MINUTES", "symbol_quarantine_minutes", 120)),
            120,
        ),
        api_circuit_fail_threshold=_parse_int(
            str(env_or_yaml("API_CIRCUIT_FAIL_THRESHOLD", "api_circuit_fail_threshold", 5)),
            5,
        ),
        api_circuit_cooldown_sec=_parse_int(
            str(env_or_yaml("API_CIRCUIT_COOLDOWN_SEC", "api_circuit_cooldown_sec", 600)),
            600,
        ),
        max_bars_fetch=_parse_int(str(env_or_yaml("MAX_BARS_FETCH", "max_bars_fetch", 1200)), 1200),
        require_repaired_posture_pack=_parse_bool(
            str(env_or_yaml("REQUIRE_REPAIRED_POSTURE_PACK", "require_repaired_posture_pack", "true")),
            True,
        ),
        repaired_posture_freeze_dir=str(
            env_or_yaml(
                "REPAIRED_POSTURE_FREEZE_DIR",
                "repaired_posture_freeze_dir",
                "/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126",
            )
        ),
        repaired_active_subset_csv=str(
            env_or_yaml(
                "REPAIRED_ACTIVE_SUBSET_CSV",
                "repaired_active_subset_csv",
                "/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_subset.csv",
            )
        ),
        repaired_active_params_dir=str(
            env_or_yaml(
                "REPAIRED_ACTIVE_PARAMS_DIR",
                "repaired_active_params_dir",
                "/root/analysis/0.87/reports/execution_layer/REPAIRED_BRANCH_3M_POSTURE_FREEZE_20260306_194126/repaired_active_3m_params",
            )
        ),
        paper_symbol_allowlist=allowlist,
        required_active_strategy_id=str(
            env_or_yaml(
                "REQUIRED_ACTIVE_STRATEGY_ID",
                "required_active_strategy_id",
                "M1_ENTRY_ONLY_PASSIVE_BASELINE",
            )
        ),
        repaired_contract_defer_exit_to_next_bar=_parse_bool(
            str(
                env_or_yaml(
                    "REPAIRED_CONTRACT_DEFER_EXIT_TO_NEXT_BAR",
                    "repaired_contract_defer_exit_to_next_bar",
                    "true",
                )
            ),
            True,
        ),
    )

    return settings
