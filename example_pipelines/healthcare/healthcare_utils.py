"""
Some useful utils for the project
"""

from typing import Any, Callable

import numpy as np
from gensim import models
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from typing_extensions import Self


# copied as gensim migration guide described here: https://github.com/RaRe-Technologies/gensim/wiki/Migrating-from-Gensim-3.x-to-4
class W2VTransformer(TransformerMixin, BaseEstimator):
    """Base Word2Vec module, wraps :class:`~gensim.models.word2vec.Word2Vec`.

    For more information please have a look to `Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean: "Efficient
    Estimation of Word Representations in Vector Space" <https://arxiv.org/abs/1301.3781>`_.

    """

    def __init__(
        self,
        vector_size: int = 100,
        alpha: float = 0.025,
        window: int = 5,
        min_count: int = 5,
        max_vocab_size: int | None = None,
        sample: float = 1e-3,
        seed: int = 1,
        workers: int = 3,
        min_alpha: float = 0.0001,
        sg: int = 0,
        hs: int = 0,
        negative: int = 5,
        cbow_mean: int = 1,
        hashfxn: Callable = hash,
        epochs: int = 5,
        null_word: int = 0,
        trim_rule: Callable | None = None,
        sorted_vocab: int = 1,
        batch_words: int = 10000,
    ) -> None:
        """

        Parameters
        ----------
        vector_size : int
            Dimensionality of the feature vectors.
        alpha : float
            The initial learning rate.
        window : int
            The maximum distance between the current and predicted word within a sentence.
        min_count : int
            Ignores all words with total frequency lower than this.
        max_vocab_size : int
            Limits the RAM during vocabulary building; if there are more unique
            words than this, then prune the infrequent ones. Every 10 million word types need about 1GB of RAM.
            Set to `None` for no limit.
        sample : float
            The threshold for configuring which higher-frequency words are randomly downsampled,
            useful range is (0, 1e-5).
        seed : int
            Seed for the random number generator. Initial vectors for each word are seeded with a hash of
            the concatenation of word + `str(seed)`. Note that for a fully deterministically-reproducible run,
            you must also limit the model to a single worker thread (`workers=1`), to eliminate ordering jitter
            from OS thread scheduling. (In Python 3, reproducibility between interpreter launches also requires
            use of the `PYTHONHASHSEED` environment variable to control hash randomization).
        workers : int
            Use these many worker threads to train the model (=faster training with multicore machines).
        min_alpha : float
            Learning rate will linearly drop to `min_alpha` as training progresses.
        sg : int {1, 0}
            Defines the training algorithm. If 1, CBOW is used, otherwise, skip-gram is employed.
        hs : int {1,0}
            If 1, hierarchical softmax will be used for model training.
            If set to 0, and `negative` is non-zero, negative sampling will be used.
        negative : int
            If > 0, negative sampling will be used, the int for negative specifies how many "noise words"
            should be drawn (usually between 5-20).
            If set to 0, no negative sampling is used.
        cbow_mean : int {1,0}
            If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
        hashfxn : callable (object -> int), optional
            A hashing function. Used to create an initial random reproducible vector by hashing the random seed.
        epochs : int
            Number of iterations (epochs) over the corpus.
        null_word : int {1, 0}
            If 1, a null pseudo-word will be created for padding when using concatenative L1 (run-of-words)
        trim_rule : function
            Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary,
            be trimmed away, or handled using the default (discard if word count < min_count).
            Can be None (min_count will be used, look to :func:`~gensim.utils.keep_vocab_item`),
            or a callable that accepts parameters (word, count, min_count) and returns either
            :attr:`gensim.utils.RULE_DISCARD`, :attr:`gensim.utils.RULE_KEEP` or :attr:`gensim.utils.RULE_DEFAULT`.
            Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
            of the model.
        sorted_vocab : int {1,0}
            If 1, sort the vocabulary by descending frequency before assigning word indexes.
        batch_words : int
            Target size (in words) for batches of examples passed to worker threads (and
            thus cython routines).(Larger batches will be passed if individual
            texts are longer than 10000 words, but the standard cython code truncates to that maximum.)

        """
        self.gensim_model = None
        self.vector_size = vector_size
        self.alpha = alpha
        self.window = window
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.sg = sg
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.null_word = null_word
        self.trim_rule = trim_rule
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X: Any, y: Any | None = None) -> Self:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : iterable of iterables of str
            The input corpus. X can be simply a list of lists of tokens, but for larger corpora,
            consider an iterable that streams the sentences directly from disk/network.
            See :class:`~gensim.models.word2vec.BrownCorpus`, :class:`~gensim.models.word2vec.Text8Corpus`
            or :class:`~gensim.models.word2vec.LineSentence` in :mod:`~gensim.models.word2vec` module for such examples.

        Returns
        -------
        :class:`~gensim.sklearn_api.w2vmodel.W2VTransformer`
            The trained model.

        """
        self.gensim_model = models.Word2Vec(
            sentences=X,
            vector_size=self.vector_size,
            alpha=self.alpha,
            window=self.window,
            min_count=self.min_count,
            max_vocab_size=self.max_vocab_size,
            sample=self.sample,
            seed=self.seed,
            workers=self.workers,
            min_alpha=self.min_alpha,
            sg=self.sg,
            hs=self.hs,
            negative=self.negative,
            cbow_mean=self.cbow_mean,
            hashfxn=self.hashfxn,
            epochs=self.epochs,
            null_word=self.null_word,
            trim_rule=self.trim_rule,
            sorted_vocab=self.sorted_vocab,
            batch_words=self.batch_words,
        )
        return self

    def transform(self, words: Any) -> np.ndarray:
        """Get the word vectors the input words.

        Parameters
        ----------
        words : {iterable of str, str}
            Word or a collection of words to be transformed.

        Returns
        -------
        np.ndarray of shape [`len(words)`, `size`]
            A 2D array where each row is the vector of one word.

        """
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(words, str):
            words = [words]
        vectors = [self.gensim_model.wv[word] for word in words]
        return np.reshape(np.array(vectors), (len(words), self.vector_size))

    def partial_fit(self, X: Any) -> Any:
        raise NotImplementedError(
            "'partial_fit' has not been implemented for W2VTransformer. "
            "However, the model can be updated with a fixed vocabulary using Gensim API call."
        )


class MyW2VTransformer(W2VTransformer):
    """Some custom w2v transformer."""

    def partial_fit(self, X: Any) -> Any:
        # pylint: disable=useless-super-delegation
        super().partial_fit([X])

    def fit(self, X: Any, y: Any | None = None) -> Self:
        X = X.iloc[:, 0].tolist()
        return super().fit([X], y)

    def transform(self, words: Any) -> np.ndarray:
        words = words.iloc[:, 0].tolist()
        if self.gensim_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        vectors = []
        for word in words:
            if word in self.gensim_model.wv:
                vectors.append(self.gensim_model.wv[word])
            else:
                vectors.append(np.zeros(self.vector_size))
        return np.reshape(np.array(vectors), (len(words), self.vector_size))


def create_model() -> Sequential:
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation="relu", input_dim=109))
    clf.add(Dense(9, activation="relu"))
    clf.add(Dense(1, activation="softmax"))
    clf.compile(
        loss="categorical_crossentropy",
        optimizer=gradient_descent.SGD(),
        metrics=["accuracy"],
    )
    return clf


def create_model_with_input(input_dim: int = 10) -> Sequential:
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation="relu", input_dim=input_dim))
    clf.add(Dense(9, activation="relu"))
    clf.add(Dense(2, activation="softmax"))
    clf.compile(
        loss="categorical_crossentropy",
        optimizer=gradient_descent.SGD(),
        metrics=["accuracy"],
    )
    return clf


def create_model_predict() -> Sequential:
    """Create a simple neural network"""
    clf = Sequential()
    clf.add(Dense(9, activation="relu", input_dim=9))
    clf.add(Dense(9, activation="relu"))
    clf.add(Dense(1, activation="sigmoid"))
    clf.compile(
        loss="binary_crossentropy",
        optimizer=gradient_descent.SGD(),
        metrics=["accuracy"],
    )
    return clf
