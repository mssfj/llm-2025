import unittest


class TestSmoke(unittest.TestCase):
    """Basic sanity test to verify the unittest runner is wired up."""

    def test_truth(self):
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
