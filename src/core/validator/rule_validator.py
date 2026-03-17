from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session


class RuleValidator:
    """规则校验模块 - 错接/漏接/多接校验、短接线/短接片校验、层级顺序校验"""

    def __init__(
        self,
        product_id: int,
        station_id: int,
        layer: int = 1,
        db: Optional[Session] = None,
    ):
        self.product_id = product_id
        self.station_id = station_id
        self.layer = layer
        self.db = db
        self.rules = self._load_rules()
        self.station_config = self._load_station_config()

    def _load_rules(self) -> List[Dict[str, Any]]:
        """从数据库加载接线规则"""
        if self.db is None:
            return []

        try:
            from src.database.models import WiringRule

            rules = (
                self.db.query(WiringRule)
                .filter(WiringRule.product_id == self.product_id)
                .filter(WiringRule.station_id == self.station_id)
            )
            if self.layer > 0:
                rules = rules.filter(WiringRule.layer == self.layer)

            return [
                {
                    "hole_number": r.hole_number,
                    "wire_number": r.wire_number,
                    "wire_color": r.wire_color,
                    "has_connector": r.has_connector,
                    "has_short_wire": r.has_short_wire,
                    "has_jumper": r.has_jumper,
                    "layer": r.layer,
                }
                for r in rules.all()
            ]
        except Exception as e:
            print(f"[RuleValidator] 加载规则失败: {e}")
            return []

    def _load_station_config(self) -> Dict[str, Any]:
        """加载工位配置"""
        try:
            from src.utils.config import get_station_config

            config = get_station_config()
            return config.get("stations", {}).get(self.station_id, {})
        except Exception:
            return {}

    def validate(
        self,
        ocr_result: Dict[str, Any],
        color_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """执行完整规则校验"""
        if not self.rules:
            return {
                "passed": True,
                "error": "无接线规则，跳过校验",
                "errors": [],
                "details": {},
            }

        errors = []
        check_items = self.station_config.get("check_items", [])

        structure = ocr_result.get("structure", {})
        terminal_numbers = structure.get("terminal_numbers", [])
        wire_numbers = structure.get("wire_numbers", [])

        detected_terminals = {t["text"]: t for t in terminal_numbers}
        detected_wires = {w["text"]: w for w in wire_numbers}

        detected_colors = {
            c.get("color", "未知"): c for c in color_result.get("colors", [])
        }

        if "wire_number_match" in check_items:
            match_errors = self._validate_wire_match(detected_terminals, detected_wires)
            errors.extend(match_errors)

        if "wire_color" in check_items:
            color_errors = self._validate_color(detected_terminals, detected_colors)
            errors.extend(color_errors)

        if "missing_wire" in check_items:
            missing_errors = self._validate_missing_wire(
                detected_terminals, detected_wires
            )
            errors.extend(missing_errors)

        if "extra_wire" in check_items:
            extra_errors = self._validate_extra_wire(detected_terminals, detected_wires)
            errors.extend(extra_errors)

        if "short_wire_correct" in check_items:
            short_errors = self._validate_short_wire(detected_terminals)
            errors.extend(short_errors)

        if "jumper_correct" in check_items:
            jumper_errors = self._validate_jumper(detected_terminals)
            errors.extend(jumper_errors)

        if "connector_installed" in check_items:
            connector_errors = self._validate_connector(detected_terminals)
            errors.extend(connector_errors)

        passed = len(errors) == 0

        return {
            "passed": passed,
            "errors": errors,
            "details": {
                "total_rules": len(self.rules),
                "detected_terminals": len(detected_terminals),
                "detected_wires": len(detected_wires),
                "detected_colors": len(detected_colors),
                "error_count": len(errors),
            },
        }

    def _validate_wire_match(
        self,
        detected_terminals: Dict[str, Dict],
        detected_wires: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """校验线号-端子号匹配"""
        errors = []

        for rule in self.rules:
            hole = rule["hole_number"]
            expected_wire = rule.get("wire_number")

            if not expected_wire:
                continue

            detected_terminal = detected_terminals.get(hole)
            if not detected_terminal:
                continue

            matched_wire = self._find_matching_wire(hole, detected_wires)

            if matched_wire and matched_wire["text"] != expected_wire:
                errors.append(
                    {
                        "type": "wire_mismatch",
                        "hole_number": hole,
                        "expected_wire": expected_wire,
                        "detected_wire": matched_wire["text"],
                        "position": detected_terminal.get("position"),
                        "message": f"端子{hole}线号不匹配: 期望[{expected_wire}], 检测到[{matched_wire['text']}]",
                    }
                )

        return errors

    def _validate_color(
        self,
        detected_terminals: Dict[str, Dict],
        detected_colors: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """校验线材颜色"""
        errors = []

        for rule in self.rules:
            hole = rule["hole_number"]
            expected_color = rule.get("wire_color")

            if not expected_color:
                continue

            detected_terminal = detected_terminals.get(hole)
            if not detected_terminal:
                continue

            color_match = detected_colors.get(expected_color)
            if not color_match:
                errors.append(
                    {
                        "type": "color_mismatch",
                        "hole_number": hole,
                        "expected_color": expected_color,
                        "detected_color": "未检测到",
                        "position": detected_terminal.get("position"),
                        "message": f"端子{hole}颜色不匹配: 期望[{expected_color}], 检测到[未检测到]",
                    }
                )

        return errors

    def _validate_missing_wire(
        self,
        detected_terminals: Dict[str, Dict],
        detected_wires: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """校验漏接"""
        errors = []

        for rule in self.rules:
            hole = rule["hole_number"]
            expected_wire = rule.get("wire_number")

            if not expected_wire:
                continue

            detected_terminal = detected_terminals.get(hole)
            if not detected_terminal:
                continue

            matched_wire = self._find_matching_wire(hole, detected_wires)

            if not matched_wire:
                errors.append(
                    {
                        "type": "missing_wire",
                        "hole_number": hole,
                        "expected_wire": expected_wire,
                        "position": detected_terminal.get("position"),
                        "message": f"端子{hole}漏接: 应接[{expected_wire}]",
                    }
                )

        return errors

    def _validate_extra_wire(
        self,
        detected_terminals: Dict[str, Dict],
        detected_wires: Dict[str, Dict],
    ) -> List[Dict[str, Any]]:
        """校验多接"""
        errors = []

        rule_holes = {r["hole_number"]: r for r in self.rules}

        for wire_text, wire_info in detected_wires.items():
            matched_hole = self._find_matching_hole(wire_text, detected_terminals)

            if matched_hole and matched_hole not in rule_holes:
                errors.append(
                    {
                        "type": "extra_wire",
                        "wire_number": wire_text,
                        "position": wire_info.get("position"),
                        "message": f"检测到多余线号: [{wire_text}]",
                    }
                )

        return errors

    def _validate_short_wire(
        self, detected_terminals: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """校验短接线"""
        errors = []

        if not self.station_config.get("has_short_wire", False):
            return errors

        for rule in self.rules:
            if not rule.get("has_short_wire", False):
                continue

            hole = rule["hole_number"]
            detected_terminal = detected_terminals.get(hole)

            if not detected_terminal:
                errors.append(
                    {
                        "type": "short_wire_missing",
                        "hole_number": hole,
                        "message": f"端子{hole}缺少短接线",
                    }
                )

        return errors

    def _validate_jumper(
        self, detected_terminals: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """校验短接片"""
        errors = []

        if not self.station_config.get("has_jumper", False):
            return errors

        for rule in self.rules:
            if not rule.get("has_jumper", False):
                continue

            hole = rule["hole_number"]
            detected_terminal = detected_terminals.get(hole)

            if not detected_terminal:
                errors.append(
                    {
                        "type": "jumper_missing",
                        "hole_number": hole,
                        "message": f"端子{hole}缺少短接片",
                    }
                )

        return errors

    def _validate_connector(
        self, detected_terminals: Dict[str, Dict]
    ) -> List[Dict[str, Any]]:
        """校验插头安装"""
        errors = []

        for rule in self.rules:
            if not rule.get("has_connector", False):
                continue

            hole = rule["hole_number"]
            detected_terminal = detected_terminals.get(hole)

            if not detected_terminal:
                errors.append(
                    {
                        "type": "connector_missing",
                        "hole_number": hole,
                        "message": f"端子{hole}插头未安装",
                    }
                )

        return errors

    def _find_matching_wire(self, hole: str, wires: Dict[str, Dict]) -> Optional[Dict]:
        """根据端子号位置找到对应的线号"""
        if not wires:
            return None

        hole_num = self._extract_number(hole)
        if hole_num is None:
            return None

        sorted_wires = sorted(
            wires.values(), key=lambda w: w.get("position", [0, 0])[0]
        )

        if 0 <= hole_num - 1 < len(sorted_wires):
            return sorted_wires[hole_num - 1]

        return None

    def _find_matching_hole(
        self, wire_text: str, terminals: Dict[str, Dict]
    ) -> Optional[str]:
        """根据线号找到对应的端子号"""
        for term_text, term_info in terminals.items():
            if (
                abs(
                    self._extract_number(term_text)
                    or 0 - self._extract_number(wire_text)
                    or 0
                )
                <= 1
            ):
                return term_text
        return None

    def _extract_number(self, text: str) -> Optional[int]:
        """从文本中提取数字"""
        import re

        numbers = re.findall(r"\d+", text)
        if numbers:
            return int(numbers[0])
        return None
