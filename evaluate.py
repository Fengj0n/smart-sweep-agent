"""运行轻量回归评估：python evaluate.py。"""

import json
from pathlib import Path

from agent.react_agent import ReactAgent


def main() -> None:
    cases_path = Path(__file__).resolve().parent / "evaluation_cases.json"
    with open(str(cases_path), "r", encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    for case in cases:
        intent = ReactAgent._intent(case["query"])
        intent_ok = intent == case["expected_intent"]
        marker = case.get("must_contain", "")
        parse_ok = True
        if marker:
            if intent.startswith("weather"):
                parse_ok = marker == ReactAgent._extract_city(case["query"])
            elif intent.startswith("report"):
                user_id, _, _ = ReactAgent._parse_report_query(case["query"])
                parse_ok = marker in case["query"] and (not marker.isdigit() or user_id == marker)
        success = intent_ok and parse_ok
        passed += int(success)
        print(f"[{'PASS' if success else 'FAIL'}] {case['query']} -> {intent}")

    print(f"\n结果：{passed}/{len(cases)} 通过")
    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
