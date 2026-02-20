"""
scenarios.py — 시나리오 저장/불러오기 관리
"""
import json
import os
from pathlib import Path

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def _ensure_dir():
    SCENARIOS_DIR.mkdir(exist_ok=True)


def save_scenario(name: str, variables: list[dict], settings: dict) -> None:
    """시나리오를 JSON 파일로 저장합니다."""
    _ensure_dir()
    payload = {"name": name, "variables": variables, "settings": settings}
    path = SCENARIOS_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_scenario(name: str) -> dict:
    """저장된 시나리오를 불러옵니다."""
    path = SCENARIOS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"시나리오 '{name}'을 찾을 수 없습니다.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_scenarios() -> list[str]:
    """저장된 시나리오 이름 목록을 반환합니다."""
    _ensure_dir()
    return [p.stem for p in sorted(SCENARIOS_DIR.glob("*.json"))]


def delete_scenario(name: str) -> None:
    """시나리오 파일을 삭제합니다."""
    path = SCENARIOS_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
