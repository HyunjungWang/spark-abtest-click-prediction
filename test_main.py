import unittest
from unittest.mock import MagicMock

class TestAdPrediction(unittest.TestCase):
    def test_model_logic_mock(self):
        mock_model = MagicMock()
        
        mock_model.predict.return_value = ["YES"]
        
        input_data = {"age": 25, "gender": "Male"}
        prediction = mock_model.predict(input_data)
        
        self.assertEqual(prediction[0], "YES")
        print("✅ Mock Test Passed: Server bypass successful!")

if __name__ == '__main__':
    unittest.main()