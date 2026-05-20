"""JUnit XML emission for ``audiobench gate`` and ``audiobench run-matrix``.

The schema is intentionally minimal but matches what GitHub Actions, GitLab,
Jenkins, and most other CI systems parse out of the box::

    <testsuites name="audiobench" tests="N" failures="F">
      <testsuite name="ab/asr-robust" tests="..." failures="...">
        <testcase classname="ab/asr-robust" name="weighted_mean_wer" time="0">
          <failure message="actual=30.0 > threshold=20.0"/>
        </testcase>
      </testsuite>
    </testsuites>
"""

from __future__ import annotations

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

from audiobench.gating import GateReport


@dataclass
class JUnitCase:
    classname: str
    name: str
    failure_message: str | None = None  # None => passing case
    skipped: bool = False
    skip_reason: str | None = None
    system_out: str | None = None


@dataclass
class JUnitSuite:
    name: str
    cases: list[JUnitCase] = field(default_factory=list)

    @property
    def tests(self) -> int:
        return len(self.cases)

    @property
    def failures(self) -> int:
        return sum(1 for case in self.cases if case.failure_message is not None)

    @property
    def skipped(self) -> int:
        return sum(1 for case in self.cases if case.skipped)


def render_junit_xml(suites: list[JUnitSuite], *, root_name: str = "audiobench") -> str:
    """Serialize ``suites`` to a JUnit-compatible XML string."""
    total_tests = sum(s.tests for s in suites)
    total_failures = sum(s.failures for s in suites)
    total_skipped = sum(s.skipped for s in suites)

    root = ET.Element(
        "testsuites",
        {
            "name": root_name,
            "tests": str(total_tests),
            "failures": str(total_failures),
            "skipped": str(total_skipped),
        },
    )
    for suite in suites:
        suite_el = ET.SubElement(
            root,
            "testsuite",
            {
                "name": suite.name,
                "tests": str(suite.tests),
                "failures": str(suite.failures),
                "skipped": str(suite.skipped),
            },
        )
        for case in suite.cases:
            case_el = ET.SubElement(
                suite_el,
                "testcase",
                {
                    "classname": case.classname,
                    "name": case.name,
                    "time": "0",
                },
            )
            if case.skipped:
                ET.SubElement(
                    case_el,
                    "skipped",
                    {"message": case.skip_reason or "skipped"},
                )
            elif case.failure_message is not None:
                ET.SubElement(
                    case_el,
                    "failure",
                    {"message": case.failure_message},
                )
            if case.system_out:
                out_el = ET.SubElement(case_el, "system-out")
                out_el.text = case.system_out

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def gate_report_to_junit(report: GateReport) -> JUnitSuite:
    """Convert one :class:`GateReport` to a :class:`JUnitSuite`."""
    classname = report.suite or "audiobench"
    suite_name = classname
    if report.model:
        suite_name = f"{classname}::{report.model}"
    suite = JUnitSuite(name=suite_name)
    for check in report.checks:
        if check.passed:
            suite.cases.append(JUnitCase(classname=classname, name=check.name))
        else:
            actual = "missing" if check.actual is None else f"{check.actual:g}"
            msg = (
                f"actual={actual} {check.comparator} threshold={check.threshold:g} failed"
            )
            if check.detail:
                msg += f" ({check.detail})"
            suite.cases.append(
                JUnitCase(
                    classname=classname,
                    name=check.name,
                    failure_message=msg,
                )
            )
    return suite
