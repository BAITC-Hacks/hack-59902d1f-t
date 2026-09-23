import unittest

from agent import Agent
from dashboard import run_dashboard
from local_eval import evaluate_agent


class DashboardTests(unittest.TestCase):
    def test_dashboard_matches_local_evaluation(self):
        expected = evaluate_agent(Agent(), seed=42, verbose=False)
        actual = run_dashboard({"seed": 42, "runs": 1})["primary"]
        self.assertAlmostEqual(actual["net_gain"], expected["net_arpu_gain"])
        self.assertEqual(actual["contacts"], expected["total_contacts"])
        self.assertEqual(actual["final_count"], 3)

    def test_rejects_invalid_run_count(self):
        with self.assertRaisesRegex(ValueError, "число прогонов"):
            run_dashboard({"seed": 42, "runs": 31})

    def test_rejects_missing_csv_columns(self):
        with self.assertRaisesRegex(ValueError, "отсутствуют колонки"):
            run_dashboard({"profile_csv": "ID_NUMBER\n1\n"})


if __name__ == "__main__":
    unittest.main()
