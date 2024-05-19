import unittest
import math

from reading import ReadingDifficultyCalculator

# this can test both reading level (0-18) or reading ease (0-100)
# note the test you want to run and uncomment the correct line, comment out the other
# assertions test if result is within 0.5 of the target
# this can be adjusted per project needs

class TestReadingDifficultyCalculator(unittest.TestCase):
    def test_calculate_reading_difficulty(self):
        text = "This is a sample text. It has multiple sentences!"
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertAlmostEqual(difficulty, 4.51, places=1)
        # reading ease: self.assertTrue(math.isclose(difficulty, 70.6, abs_tol=0.5))

        text = "The quick brown fox jumps over the lazy dog."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertAlmostEqual(difficulty, 2.34, places=1)
        # reading ease: self.assertTrue(math.isclose(difficulty, 94.30, abs_tol=0.5))

        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertAlmostEqual(difficulty, 15.55, places=1)
        # reading ease: self.assertLessEqual(difficulty, 0)

        text = "Gestational hypertension is a medical term for high blood pressure in pregnancy."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertAlmostEqual(difficulty, 11.71, delta=1)
        # reading ease: self.assertTrue(math.isclose(difficulty, 25.46, abs_tol=0.5))

        text = "Blue skies is a medical term for high blood pressure in pregnancy."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertAlmostEqual(difficulty, 5.81, delta=1)
        # reading ease: self.assertTrue(math.isclose(difficulty, 67.76, abs_tol=0.5))

if __name__ == '__main__':
    unittest.main()