from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file, send_from_directory
from grr.model import ShowAndTellModel
from vocabulary import Vocabulary
#from model.caption_generator import CaptionGenerator
import heapq
import math


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class Caption(object):
    """Represents a complete or partial caption."""

    def __init__(self, sentence, state, logprob, score, metadata=None):
        """Initializes the Caption.

        Args:
          sentence: List of word ids in the caption.
          state: Model state after generating the previous word.
          logprob: Log-probability of the caption.
          score: Score of the caption.
          metadata: Optional metadata associated with the partial sentence. If not
            None, a list of strings with the same length as 'sentence'.
        """
        self.sentence = sentence
        self.state = state
        self.logprob = logprob
        self.score = score
        self.metadata = metadata

    def __cmp__(self, other):
        """Compares Captions by score."""
        assert isinstance(other, Caption)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1

    # For Python 3 compatibility (__cmp__ is deprecated).
    def __lt__(self, other):
        assert isinstance(other, Caption)
        return self.score < other.score

    # Also for Python 3 compatibility.
    def __eq__(self, other):
        assert isinstance(other, Caption)
        return self.score == other.score


class CaptionGenerator(object):
    """Class to generate captions from an image-to-text model.
    This code is a modification of https://github.com/tensorflow/models/blob/master/research/im2txt/im2txt/inference_utils/caption_generator.py
    """

    def __init__(self, model, vocab, beam_size=3, max_caption_length=20, length_normalization_factor=0.0):

        self.vocab = vocab
        self.model = model

        self.beam_size = beam_size
        self.max_caption_length = max_caption_length
        self.length_normalization_factor = length_normalization_factor

    def beam_search(self, encoded_image):
        # Feed in the image to get the initial state.
        partial_caption_beam = TopN(self.beam_size)
        complete_captions = TopN(self.beam_size)
        initial_state = self.model.feed_image(encoded_image)

        initial_beam = Caption(
            sentence=[self.vocab.start_id],
            state=initial_state[0],
            logprob=0.0,
            score=0.0,
            metadata=[""])

        partial_caption_beam.push(initial_beam)

        # Run beam search.
        for _ in range(self.max_caption_length - 1):
            partial_captions_list = partial_caption_beam.extract()
            partial_caption_beam.reset()
            input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
            state_feed = np.array([c.state for c in partial_captions_list])

            softmax, new_states, metadata = self.model.inference_step(input_feed,
                                                                      state_feed)

            for i, partial_caption in enumerate(partial_captions_list):
                word_probabilities = softmax[i]
                state = new_states[i]
                # For this partial caption, get the beam_size most probable next words.
                words_and_probs = list(enumerate(word_probabilities))
                words_and_probs.sort(key=lambda x: -x[1])
                words_and_probs = words_and_probs[0:self.beam_size]
                # Each next word gives a new partial caption.
                for w, p in words_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).
                    sentence = partial_caption.sentence + [w]
                    logprob = partial_caption.logprob + math.log(p)
                    score = logprob
                    if metadata:
                        metadata_list = partial_caption.metadata + [metadata[i]]
                    else:
                        metadata_list = None
                    if w == self.vocab.end_id:
                        if self.length_normalization_factor > 0:
                            score /= len(sentence) ** self.length_normalization_factor
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        complete_captions.push(beam)
                    else:
                        beam = Caption(sentence, state, logprob, score, metadata_list)
                        partial_caption_beam.push(beam)
            if partial_caption_beam.size() == 0:
                # We have run out of partial candidates; happens when beam_size = 1.
                break

        # If we have no complete captions then fall back to the partial captions.
        # But never output a mixture of complete and partial captions because a
        # partial caption could have a higher score than all the complete captions.
        if complete_captions.size() == 0:
            complete_captions = partial_caption_beam

        return complete_captions.extract(sort=True)


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model", "./show-and-tell.pb", "Model graph def path")
tf.flags.DEFINE_string("vocab", "./word_counts.txt", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("port", "5000", "Port of the server.")
tf.flags.DEFINE_string("host", "localhost", "Host of the server.")
tf.flags.DEFINE_integer("beam_size", 3, "Size of the beam.")
tf.flags.DEFINE_integer("max_caption_length", 20, "Maximum length of the generate caption.")

vocab = Vocabulary(vocab_file_path=FLAGS.vocab)
model = ShowAndTellModel(model_path=FLAGS.model)
generator = CaptionGenerator(model=model, vocab=vocab, beam_size=FLAGS.beam_size,
                             max_caption_length=FLAGS.max_caption_length)

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/api/image-caption/predict', methods=['GET','POST'])
def caption():
    if request.method == 'POST':
        file = request.files['image']
        image = file.read()
        captions = generator.beam_search(image)
        sentences = []
        for caption in captions:
            sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
            sentences.append((" ".join(sentence), np.exp(caption.logprob)))

        logger.info(sentences)
        #print(sentences[0][0])
        #return 'ok'
        #return jsonify({"captions": sentences})
        return render_template('index.html',filename = file.filename, var1=sentences[0][0],var2=sentences[0][1],var3=sentences[1][0],var4=sentences[1][1],var5=sentences[2][0],var6=sentences[2][1],)
    return render_template('index.html', filename = "doggo.jpg")
@app.route('/get_image')
def get_image():
    #route = "./imgs/" + str(request.args.get('filename'))
#    route = "./imgs/"+str(filename)
    return send_file("./imgs/"+request.args.get('filename'), mimetype='image/gif')

@app.route('/api/image-caption/dark.css')
def send_css():
    return send_file("./templates/dark.css")

# @app.route('/api/image-caption/home.html',methods=['GET','POST'])
# def send_home():
#     if request.method == 'POST':
#         file = request.files['image']
#         image = file.read()
#         captions = generator.beam_search(image)
#         sentences = []
#         for caption in captions:
#             sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
#             sentences.append((" ".join(sentence), np.exp(caption.logprob)))
#
#         logger.info(sentences)
#         #print(sentences[0][0])
#         #return 'ok'
#         #return jsonify({"captions": sentences})
#         return render_template('index.html',filename = file.filename, var1=sentences[0][0],var2=sentences[0][1],var3=sentences[1][0],var4=sentences[1][1],var5=sentences[2][0],var6=sentences[2][1],)
#
#     return render_template("./templates/home.html")

if __name__ == '__main__':
    app.run(host=FLAGS.host, port=FLAGS.port)
