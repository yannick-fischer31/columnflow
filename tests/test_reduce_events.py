import unittest

from columnflow.tasks.reduction import ReduceEvents


class TestReduceEvents(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.options = {
            "dataset": "tt_powheg",
            "version": "test",
            "config": "test_campaign_limited",
        }
        cls.task = ReduceEvents(**cls.options)

    def test_reduce_events(self):
        self.task.law_run()

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":
    unittest.main()
