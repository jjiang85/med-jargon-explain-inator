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
        # reading level: self.assertAlmostEqual(difficulty, 6.67, places=2)
        self.assertTrue(math.isclose(difficulty, 70.6, abs_tol=0.5))

        text = "The quick brown fox jumps over the lazy dog."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        # reading level: self.assertAlmostEqual(difficulty, -1.67, places=2)
        self.assertTrue(math.isclose(difficulty, 94.30, abs_tol=0.5))

        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        # reading level: self.assertAlmostEqual(difficulty, 5.89, places=2)
        self.assertLessEqual(difficulty, 0)

        # TODO check answer for two calculations below
        text = "Gestational hypertension is a medical term for high blood pressure in pregnancy."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertTrue(math.isclose(difficulty, 25.46, abs_tol=0.5))

        text = "Blue skys is a medical term for high blood pressure in pregnancy."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        self.assertTrue(math.isclose(difficulty, 67.76, abs_tol=0.5))

    # this method was simplified and put into the main method
    # left here in case we want to separate it back out
    # def test_count_syllables(self):
    #     word = "hello"
    #     syllables = ReadingDifficultyCalculator.count_syllables(word)
    #     self.assertEqual(syllables, 2)

    #     word = "world"
    #     syllables = ReadingDifficultyCalculator.count_syllables(word)
    #     self.assertEqual(syllables, 1)

    #     word = "banana"
    #     syllables = ReadingDifficultyCalculator.count_syllables(word)
    #     self.assertEqual(syllables, 3)

    #     word = "hypertension"
    #     syllables = ReadingDifficultyCalculator.count_syllables(word)
    #     self.assertEqual(syllables, 4)

    #     word = "intracerebral"
    #     syllables = ReadingDifficultyCalculator.count_syllables(word)
    #     self.assertEqual(syllables, 5)

        # how many syllables is hemmorage????
        # word = "hemmorage"
        # syllables = ReadingDifficultyCalculator.count_syllables(word)
        # self.assertEqual(syllables, 3)

if __name__ == '__main__':
    unittest.main()