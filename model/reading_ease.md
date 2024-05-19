# Grade Level Reading Evaluation

This directory contains the `reading.py` script, which is used to evaluate the reading ease of medical jargon definitions.

The output is a grade level number, with the difficulty increasing as the numbers increase. The intent of this is to output a number than can be used to evaluate the reading difficulty of the report.

The target reading level is 6.0 as recommended by the American Medical Association for medical information.

## Calculation

This uses the Flesch-Kincaid Grade Level formula which counts the number of syllables, words, and sentences. This formula heavily weights multi-syllable words. Medical jargon terms can have many, many, many syllables so they will lower the reading ease.

`reading_ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)`

## Usage

To use the `reading.py` script, you need to import the `ReadingDifficultyCalculator` class from it.

Create an instance of ReadingDifficultyCalculator and use the
calculate_reading_difficulty method to calculate the reading difficulty of a text. This will print the Flesch-Kincaid Grade Level of the text.

```python
from reading import ReadingDifficultyCalculator

calculator = ReadingDifficultyCalculator("This is a simple sentence for testing.")
print(calculator.calculate_reading_difficulty())

```

## Testing

To run the tests for the reading.py script, you can use the test_reading.py script in the same directory. This script uses the unittest module to test the calculate_reading_difficulty method with various inputs.

The nltk tokenizer has some variation between its results and those of human speakers. For instance, it tokenizes "pressure" as three syllables while many speakers pronounce it with two. To account for this variation, tests use delta=1.

## Dependencies

The reading.py script depends on the nltk library for counting syllables. You can install this library using pip:
`pip install nltk`

## References

Rooney MK, Santiago G, Perni S, et al. Readability of Patient Education Materials From High-Impact Medical Journals: A 20-Year Analysis. Journal of Patient Experience. 2021;8. doi:10.1177/2374373521998847

Solnyshkina, M.I., Zamaletdinov, R.R., Gorodetskaya, L.A., & Gabitov, A.I.
(2017). Evaluating Text Complexity and Flesch-Kincaid Grade Level. *Journal of Social Studies Education Research, 8* , 238-248.
