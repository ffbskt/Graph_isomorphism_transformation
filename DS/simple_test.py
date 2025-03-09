import unittest

class SimpleTest(unittest.TestCase):
    def test_simple(self):
        print("Simple test running")
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main() 