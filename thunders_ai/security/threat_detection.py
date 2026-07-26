"""Threat Detection Module for Thunders AI.

Provides pattern-based and ML-based threat analysis for AI inputs,
including prompt injection detection, malware pattern recognition,
API request scanning, and threat reporting.
"""

from __future__ import annotations

import hashlib
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class ThreatLevel(Enum):
    """Severity levels for detected threats."""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


# Prompt injection patterns
PROMPT_INJECTION_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", "ignore_previous"),
    (r"(?i)forget\s+(all\s+)?previous\s+(instructions?|rules?)", "forget_previous"),
    (r"(?i)you\s+are\s+now\s+(?:a|an)\s+", "role_override"),
    (r"(?i)pretend\s+(you\s+are|to\s+be)", "pretend"),
    (r"(?i)system\s*:\s*", "system_prefix"),
    (r"(?i)<\|im_start\|>", "chatml_injection"),
    (r"(?i)\[INST\]", "inst_injection"),
    (r"(?i)jailbreak", "jailbreak_keyword"),
    (r"(?i)bypass\s+(the\s+)?(filter|safety|security|guard)", "bypass_filter"),
    (r"(?i)disregard\s+(your|the|all)\s+(rules?|guidelines?|policies?)", "disregard_rules"),
    (r"(?i)do\s+not\s+(follow|obey|comply)", "disobey"),
    (r"(?i)reveal\s+(your|the)\s+(system|initial|original)\s+prompt", "prompt_extraction"),
    (r"(?i)output\s+your\s+(system|hidden)\s+prompt", "prompt_extraction_output"),
    (r"(?i)sudo\s+mode", "sudo_mode"),
    (r"(?i)developer\s+mode", "developer_mode"),
]

# Malware signature patterns (simplified)
MALWARE_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)eval\s*\(\s*(base64_decode|gzinflate|gzuncompress|str_rot13)", "php_obfuscation"),
    (r"(?i)(?:document\.)?write\s*\(\s*(?:unescape|fromCharCode)", "js_payload"),
    (r"(?i)powershell\s+-enc(odedcommand)?\s+", "powershell_encoded"),
    (r"(?i)cmd\s+/c\s+", "cmd_execution"),
    (r"(?i)(?:wget|curl)\s+.*\|\s*(?:bash|sh|python)", "remote_script_exec"),
    (r"(?i)chmod\s+\+x", "make_executable"),
    (r"(?i)/etc/(?:passwd|shadow|hosts)", "sensitive_file_access"),
    (r"(?i)rm\s+-rf\s+/", "destructive_delete"),
    (r"(?i)(?:CREATE|DROP|ALTER|INSERT|UPDATE|DELETE)\s+.*(?:UNION|OR\s+1=1)", "sql_injection"),
    (r"(?i)<script[^>]*>.*</script>", "xss_script"),
    (r"(?i)(?:%3C|%3E|%22|%27).*(?:%3C|%3E|%22|%27)", "encoded_injection"),
]

# API request threat patterns
API_THREAT_PATTERNS: List[Tuple[str, str]] = [
    (r"(?i)(\.\.\/){2,}", "path_traversal"),
    (r"(?i)<\s*!\s*\[CDATA\[", "cdata_injection"),
    (r"(?i)%(?:2[2-7]|[36][a-f0-9])", "double_encoding"),
    (r"(?i)\$\{.*\}", "template_injection"),
    (r"(?i)\{\{.*\}\}", "ssti_pattern"),
    (r"(?i);?\s*(?:DROP|DELETE|TRUNCATE)\s+TABLE", "sql_destructive"),
    (r"(?i)(?:admin|root|debug|test)\s*[:=]", "credential_probe"),
    (r"(?i)content-type\s*:\s*(?:multipart|application/x-www-form-urlencoded).*\n.*(?:script|eval)", "content_type_attack"),
]


class _ThreatDatabase:
    """In-memory threat signature database."""

    def __init__(self) -> None:
        self._known_hashes: Dict[str, Dict[str, Any]] = {}
        self._custom_patterns: List[Tuple[str, str]] = []
        self._last_updated = time.time()

    def add_signature(self, content_hash: str, threat_info: Dict[str, Any]) -> None:
        """Add a known threat signature."""
        self._known_hashes[content_hash] = threat_info
        self._last_updated = time.time()

    def add_pattern(self, pattern: str, name: str) -> None:
        """Add a custom threat pattern."""
        try:
            re.compile(pattern)
            self._custom_patterns.append((pattern, name))
        except re.error as exc:
            logger.warning("Invalid pattern '%s': %s", name, exc)

    def lookup_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Look up a content hash in the threat database."""
        return self._known_hashes.get(content_hash)

    def get_custom_patterns(self) -> List[Tuple[str, str]]:
        """Return all custom patterns."""
        return list(self._custom_patterns)

    @property
    def signature_count(self) -> int:
        return len(self._known_hashes) + len(self._custom_patterns)


class ThreatDetector:
    """AI-powered threat detection system.

    Analyzes inputs, API requests, and content for security threats using
    pattern matching and machine-learning-inspired scoring.

    Args:
        detection_level: Sensitivity level ('low', 'medium', 'high').
        custom_patterns: Additional (regex_pattern, name) tuples.

    Example::

        detector = ThreatDetector()
        report = detector.analyze("ignore all previous instructions")
        is_injection = detector.detect_injection(user_prompt)
    """

    LEVEL_THRESHOLDS: Dict[str, Dict[str, int]] = {
        "low": {"low": 3, "medium": 2, "high": 1, "critical": 1},
        "medium": {"low": 5, "medium": 3, "high": 2, "critical": 1},
        "high": {"low": 2, "medium": 1, "high": 1, "critical": 1},
    }

    def __init__(
        self,
        detection_level: Optional[str] = None,
        custom_patterns: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        cfg = get_config().security
        self._detection_level = detection_level or cfg.threat_detection_level
        self._database = _ThreatDatabase()
        self._analysis_history: List[Dict[str, Any]] = []

        if custom_patterns:
            for pattern, name in custom_patterns:
                self._database.add_pattern(pattern, name)

        logger.info("ThreatDetector initialized (level: %s)", self._detection_level)

    def analyze(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Comprehensively analyze text for all types of threats.

        Args:
            text: Input text to analyze.
            context: Optional context metadata (source, user, etc.).

        Returns:
            Analysis report with threat level, detected threats, and scores.
        """
        findings: List[Dict[str, Any]] = []

        injection_threats = self._match_patterns(text, PROMPT_INJECTION_PATTERNS, "prompt_injection")
        findings.extend(injection_threats)

        malware_threats = self._match_patterns(text, MALWARE_PATTERNS, "malware")
        findings.extend(malware_threats)

        custom_threats = self._match_patterns(text, self._database.get_custom_patterns(), "custom")
        findings.extend(custom_threats)

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        db_match = self._database.lookup_hash(content_hash)
        if db_match:
            findings.append({
                "type": "known_threat",
                "category": db_match.get("category", "unknown"),
                "name": db_match.get("name", "database_match"),
                "match": content_hash[:16],
                "severity": db_match.get("severity", "high"),
            })

        overall_level = self._calculate_threat_level(findings)
        threat_score = self._calculate_score(findings)

        report: Dict[str, Any] = {
            "threat_level": overall_level.name,
            "threat_score": threat_score,
            "findings_count": len(findings),
            "findings": findings,
            "content_hash": content_hash[:16],
            "analyzed_at": time.time(),
            "context": context or {},
        }

        self._analysis_history.append(report)
        if len(self._analysis_history) > 1000:
            self._analysis_history = self._analysis_history[-500:]

        if overall_level.value >= ThreatLevel.HIGH.value:
            logger.warning("High-severity threat detected: level=%s, score=%.2f, findings=%d",
                           overall_level.name, threat_score, len(findings))

        return report

    def detect_injection(self, text: str) -> Dict[str, Any]:
        """Detect prompt injection attempts in text.

        Args:
            text: Input text to check for injection patterns.

        Returns:
            Dictionary with 'is_injection', 'confidence', and 'matched_patterns'.
        """
        matches = self._match_patterns(text, PROMPT_INJECTION_PATTERNS, "prompt_injection")
        confidence = min(1.0, len(matches) * 0.3 + (0.1 if any(m["severity"] == "high" for m in matches) else 0.0))
        is_injection = len(matches) > 0 and confidence >= 0.4

        return {
            "is_injection": is_injection,
            "confidence": round(confidence, 3),
            "matched_patterns": [m["name"] for m in matches],
            "threat_level": ThreatLevel.HIGH.name if is_injection else ThreatLevel.NONE.name,
        }

    def detect_malware(self, text: str) -> Dict[str, Any]:
        """Detect malware patterns in text content.

        Args:
            text: Text to scan for malware signatures.

        Returns:
            Dictionary with 'is_malicious', 'threat_level', and 'detected_patterns'.
        """
        matches = self._match_patterns(text, MALWARE_PATTERNS, "malware")
        is_malicious = len(matches) > 0

        return {
            "is_malicious": is_malicious,
            "threat_level": ThreatLevel.HIGH.name if is_malicious else ThreatLevel.NONE.name,
            "detected_patterns": matches,
        }

    def scan_api_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Scan an API request for security threats.

        Args:
            method: HTTP method (GET, POST, etc.).
            path: Request path.
            headers: Request headers dictionary.
            body: Request body text.

        Returns:
            Scan report with threat assessment.
        """
        all_text = f"{method} {path}"
        if headers:
            all_text += " " + " ".join(f"{k}:{v}" for k, v in headers.items())
        if body:
            all_text += " " + body

        findings: List[Dict[str, Any]] = []
        api_threats = self._match_patterns(all_text, API_THREAT_PATTERNS, "api_attack")
        findings.extend(api_threats)

        injection_threats = self._match_patterns(all_text, PROMPT_INJECTION_PATTERNS, "prompt_injection")
        findings.extend(injection_threats)

        malware_threats = self._match_patterns(all_text, MALWARE_PATTERNS, "malware")
        findings.extend(malware_threats)

        overall_level = self._calculate_threat_level(findings)
        is_safe = overall_level.value <= ThreatLevel.LOW.value

        return {
            "is_safe": is_safe,
            "threat_level": overall_level.name,
            "findings_count": len(findings),
            "findings": findings,
            "method": method,
            "path": path,
        }

    def get_threat_report(self, limit: int = 10) -> Dict[str, Any]:
        """Generate a summary report of recent threat analyses.

        Args:
            limit: Maximum number of recent analyses to include.

        Returns:
            Summary report with statistics and recent findings.
        """
        recent = self._analysis_history[-limit:]
        level_counts: Dict[str, int] = {"NONE": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        category_counts: Dict[str, int] = {}

        for report in self._analysis_history:
            level = report.get("threat_level", "NONE")
            level_counts[level] = level_counts.get(level, 0) + 1
            for finding in report.get("findings", []):
                cat = finding.get("category", "unknown")
                category_counts[cat] = category_counts.get(cat, 0) + 1

        return {
            "total_analyses": len(self._analysis_history),
            "detection_level": self._detection_level,
            "database_signatures": self._database.signature_count,
            "level_distribution": level_counts,
            "category_distribution": category_counts,
            "recent_analyses": recent,
        }

    def _match_patterns(self, text: str, patterns: List[Tuple[str, str]], category: str) -> List[Dict[str, Any]]:
        """Match text against a list of regex patterns.

        Args:
            text: Text to scan.
            patterns: List of (regex, name) tuples.
            category: Threat category label.

        Returns:
            List of finding dictionaries for each match.
        """
        findings: List[Dict[str, Any]] = []
        for pattern, name in patterns:
            try:
                matches = re.findall(pattern, text)
                if matches:
                    severity = self._assess_severity(category, name)
                    findings.append({
                        "type": category,
                        "category": category,
                        "name": name,
                        "match_count": len(matches),
                        "severity": severity,
                    })
            except re.error as exc:
                logger.warning("Regex error in pattern '%s': %s", name, exc)
        return findings

    def _assess_severity(self, category: str, name: str) -> str:
        """Assess the severity of a detected pattern.

        Args:
            category: Threat category.
            name: Specific pattern name.

        Returns:
            Severity string: 'low', 'medium', 'high', or 'critical'.
        """
        critical_keywords = {"jailbreak", "bypass_filter", "prompt_extraction", "destructive_delete", "remote_script_exec"}
        high_keywords = {"role_override", "pretend", "system_prefix", "sql_injection", "xss_script", "path_traversal", "sql_destructive"}
        medium_keywords = {"ignore_previous", "forget_previous", "disobey", "sudo_mode", "developer_mode"}

        name_lower = name.lower()
        if any(kw in name_lower for kw in critical_keywords):
            return "critical"
        if any(kw in name_lower for kw in high_keywords):
            return "high"
        if any(kw in name_lower for kw in medium_keywords):
            return "medium"
        return "low"

    def _calculate_threat_level(self, findings: List[Dict[str, Any]]) -> ThreatLevel:
        """Calculate overall threat level from findings.

        Args:
            findings: List of finding dictionaries.

        Returns:
            Overall ThreatLevel enum value.
        """
        if not findings:
            return ThreatLevel.NONE

        severity_counts: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for f in findings:
            sev = f.get("severity", "low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        thresholds = self.LEVEL_THRESHOLDS.get(self._detection_level, self.LEVEL_THRESHOLDS["medium"])

        if severity_counts.get("critical", 0) >= thresholds["critical"]:
            return ThreatLevel.CRITICAL
        if severity_counts.get("high", 0) >= thresholds["high"]:
            return ThreatLevel.HIGH
        if severity_counts.get("medium", 0) >= thresholds["medium"]:
            return ThreatLevel.MEDIUM
        if severity_counts.get("low", 0) >= thresholds["low"]:
            return ThreatLevel.LOW
        return ThreatLevel.NONE

    def _calculate_score(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate a numeric threat score from 0.0 to 1.0.

        Args:
            findings: List of finding dictionaries.

        Returns:
            Threat score between 0.0 and 1.0.
        """
        if not findings:
            return 0.0

        weights = {"low": 0.1, "medium": 0.3, "high": 0.6, "critical": 0.9}
        total_score = sum(weights.get(f.get("severity", "low"), 0.1) for f in findings)
        normalized = min(1.0, total_score / max(len(findings), 1))
        return round(normalized, 3)

    def add_threat_signature(self, content: str, threat_info: Dict[str, Any]) -> None:
        """Add a known threat signature to the database.

        Args:
            content: The malicious content to register.
            threat_info: Metadata about the threat (category, name, severity).
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        self._database.add_signature(content_hash, threat_info)
        logger.info("Added threat signature: %s", threat_info.get("name", "unnamed"))

    def get_info(self) -> Dict[str, Any]:
        """Return detector configuration and status."""
        return {
            "detection_level": self._detection_level,
            "database_signatures": self._database.signature_count,
            "total_analyses": len(self._analysis_history),
            "injection_patterns": len(PROMPT_INJECTION_PATTERNS),
            "malware_patterns": len(MALWARE_PATTERNS),
            "api_patterns": len(API_THREAT_PATTERNS),
        }
