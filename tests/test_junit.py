from __future__ import annotations

import unittest
from xml.etree import ElementTree as ET

from audiobench.gating import GateCheck, GateReport
from audiobench.junit import (
    JUnitCase,
    JUnitSuite,
    gate_report_to_junit,
    render_junit_xml,
)


class JUnitRenderTest(unittest.TestCase):
    def test_render_basic(self) -> None:
        suite = JUnitSuite(
            name="ab/sound-id",
            cases=[
                JUnitCase(classname="ab/sound-id", name="weighted_recall"),
                JUnitCase(
                    classname="ab/sound-id",
                    name="weighted_fpr",
                    failure_message="actual=0.5 <= threshold=0.1 failed",
                ),
            ],
        )
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        self.assertEqual(root.tag, "testsuites")
        self.assertEqual(root.attrib["tests"], "2")
        self.assertEqual(root.attrib["failures"], "1")
        cases = root.findall(".//testcase")
        self.assertEqual(len(cases), 2)
        failures = root.findall(".//failure")
        self.assertEqual(len(failures), 1)
        self.assertIn("threshold=0.1", failures[0].attrib["message"])

    def test_gate_report_to_junit_failure_message(self) -> None:
        report = GateReport(
            suite="ab/asr-robust",
            model="fake-asr",
            run_hash="abcdef",
            checks=[
                GateCheck(
                    name="weighted_mean_wer",
                    actual=40.0,
                    threshold=20.0,
                    comparator="<=",
                    passed=False,
                ),
                GateCheck(
                    name="wer[clean]",
                    actual=5.0,
                    threshold=10.0,
                    comparator="<=",
                    passed=True,
                ),
            ],
        )
        junit_suite = gate_report_to_junit(report)
        self.assertEqual(junit_suite.tests, 2)
        self.assertEqual(junit_suite.failures, 1)
        self.assertTrue(junit_suite.name.startswith("ab/asr-robust"))

    def test_render_handles_missing_actual(self) -> None:
        report = GateReport(
            suite="ab/asr-robust",
            model=None,
            run_hash=None,
            checks=[
                GateCheck(
                    name="wer[missing]",
                    actual=None,
                    threshold=10.0,
                    comparator="<=",
                    passed=False,
                    detail="condition not in run",
                ),
            ],
        )
        suite = gate_report_to_junit(report)
        xml = render_junit_xml([suite])
        root = ET.fromstring(xml)
        failure = root.find(".//failure")
        self.assertIsNotNone(failure)
        self.assertIn("missing", failure.attrib["message"])
        self.assertIn("condition not in run", failure.attrib["message"])
