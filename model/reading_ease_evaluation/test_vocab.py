import json
import unittest
import statistics
from reading import ReadingDifficultyCalculator

class TestReadingEase(unittest.TestCase):
    def test_calculate_reading_difficulty(self):
        with open('umls_vocab_sample_onedef.json', 'r') as f:
            data = json.load(f)
        
        results = []
        for key, value in data.items():
            print(f"Testing '{key}'")
            print(f"{value}")
            calculator = ReadingDifficultyCalculator(value)
            result = calculator.calculate_reading_difficulty()
            results.append(result)
            print(f"Reading ease score: {result}\n")

        mean_score = round(statistics.mean(results), 2)
        median_score = round(statistics.median(results), 2)
        print(f"Mean reading ease score: {mean_score}")
        print(f"Median reading ease score: {median_score}")
        high_score = max(results)
        low_score = min(results)
        print(f"Highest reading ease score: {high_score}")
        print(f"Lowest reading ease score: {low_score}")

if __name__ == '__main__':
    unittest.main()