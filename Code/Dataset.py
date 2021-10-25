"""Dataset file."""
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
import re


class Data:
    """Class to create an object representing the dataset."""

    def __init__(self) -> None:
        # nltk.download('stopwords')
        # nltk.download('wordnet')
        self.one_hot = MultiLabelBinarizer()

    def filter_synopsis(self, text: str) -> str:
        """
        Use to clean the synopsis by removing the stop words, replacing some abbreviations and stem each words.

        :param text: Each tweet one by one.
        :return: the cleaned tweet.
        """
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub(r"#", "", text)
        text = text.strip(' ')

        stop = set(stopwords.words('english'))
        stemmer = SnowballStemmer("english")
        cleaned_text = [stemmer.stem(word.lower()) for word in text.split() if word.lower() not in stop]

        return " ".join(cleaned_text)
