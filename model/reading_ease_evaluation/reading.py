import nltk
class ReadingDifficultyCalculator:

    """
    A class to calculate the reading difficulty of a given text using the Flesch-Kincaid formula.

    Args:   text (str): the text for which reading difficulty should be calculated

    Returns:
        reading_ease (float): The reading difficulty score rounded to two decimal points

    Usage:
        text = "airway protection: Inserting a tube into the windpipe to keep it wide open and prevent \
            vomit or other material from getting into the lungs."
        calculator = ReadingDifficultyCalculator(text)
        difficulty = calculator.calculate_reading_difficulty()
        print(f"Reading difficulty: {difficulty}")

    """
    def __init__(self, text: str):
        self.text = text

    def calculate_reading_difficulty(self) -> float:
        """
        Calculate the reading difficulty of the given text using the Flesch-Kincaid formula.
        Uses reading ease instead of grade_level
        Returns:
            reading_ease (float): reading difficulty score represented as the Flesch-Kincaid reading ease
            rounded to two decimal points
        """

        # count the number of words
        words = len(self.text.split())

        # count the number of sentences
        sentences = len(nltk.sent_tokenize(self.text))

        # count the number of syllables
        # TODO: can this be faster?
        syllables = 0
        for word in self.text.split():
            syllables += len(nltk.SyllableTokenizer().tokenize(word))

        reading_ease = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return round(reading_ease, 2)
    
# Example usage
