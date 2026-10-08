"""Tests for the single-page demo UI that ``GET /app`` serves.

The page is generated from the feature schema, so most of these tests exist to
fail loudly on the day the schema and the UI drift apart. The HTTP-level
behaviour of the route lives in ``test_server.py``, next to the socket fixture.
"""

from __future__ import annotations

import re
import unittest

from riskscore.features import NUMERIC_FIELDS, PURPOSE_CODES, normalise_payload
from riskscore.webapp import FIELD_LABELS, FIELD_STEPS, SAMPLE_APPLICATION, render_page


class PageContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = render_page()

    def test_every_numeric_field_has_an_input(self) -> None:
        for field in NUMERIC_FIELDS:
            self.assertIn(f'name="{field}"', self.page, f"{field} has no form input")

    def test_labels_and_steps_cover_the_numeric_schema(self) -> None:
        # Adding a feature to features.py without a label and a step here would
        # leave the generated form incomplete, so the mismatch fails the suite.
        self.assertEqual(set(FIELD_LABELS), set(NUMERIC_FIELDS))
        self.assertEqual(set(FIELD_STEPS), set(NUMERIC_FIELDS))

    def test_every_purpose_code_is_offered(self) -> None:
        for code in PURPOSE_CODES:
            self.assertIn(f'value="{code}"', self.page)

    def test_sample_application_is_scorable(self) -> None:
        # The prefill has to pass validation: a form that errors on first submit
        # is worse than no prefill at all.
        normalised = normalise_payload(dict(SAMPLE_APPLICATION))
        self.assertEqual(normalised["purpose"], SAMPLE_APPLICATION["purpose"])
        self.assertEqual(
            set(SAMPLE_APPLICATION) - {"purpose"},
            set(NUMERIC_FIELDS),
        )

    def test_page_references_no_external_resources(self) -> None:
        # "Works with no network" is a maintained property, not an accident:
        # no CDN, no font host, no analytics.
        references = re.findall(r'(?:src|href)="([^"]+)"', self.page)
        external = [
            url for url in references if url.startswith(("http://", "https://", "//"))
        ]
        self.assertEqual(external, [])

    def test_page_is_a_single_document(self) -> None:
        # One request, no sub-resources -- so there is no second path to serve,
        # and nothing to 404 on a fresh clone.
        self.assertNotIn('<link rel="stylesheet"', self.page)
        self.assertNotIn("<script src=", self.page)

    def test_page_calls_only_public_endpoints(self) -> None:
        for endpoint in ("/v1/score", "/v1/bands"):
            self.assertIn(endpoint, self.page)

    def test_curl_example_matches_the_sample_application(self) -> None:
        # The page shows a copy-pasteable command; it must carry the same values
        # the form is prefilled with, or the first thing a reviewer tries fails.
        for field in NUMERIC_FIELDS:
            self.assertIn(str(SAMPLE_APPLICATION[field]), self.page)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
