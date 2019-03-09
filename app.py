from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from model import Model
from vocabulary import Vocabulary
from caption_genarator import CaptionGenerator


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model", "", "Model graph def path")
tf.flags.DEFINE_string("vocab", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("port", "5000", "Port of the server.")
tf.flags.DEFINE_string("host", "localhost", "Host of the server.")
tf.flags.DEFINE_integer("beam_size", 3, "Size of the beam.")
tf.flags.DEFINE_integer("max_caption_length", 20, "Maximum length of the generate caption.")

vocab = Vocabulary(FLAGS.vocab)
model = Model(model_path=FLAGS.model)
generator = CaptionGenerator(model=model, vocab=vocab, beam_size=FLAGS.beam_size, max_caption_length=FLAGS.max_caption_length)

logger = logging.getLogger(__name__)
app = Flask(__name__)


@app.route('/api/image-caption/predict', methods=['POST'])
def caption():
    file = request.files['image']
    image = file.read()    
    captions = generator.beam_search(image)
    sentences = []
    for caption in captions:
        sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]      
        sentences.append((" ".join(sentence), np.exp(caption.logprob)))
    
    logger.info(sentences)
    return jsonify({"captions": sentences}) 

if __name__ == '__main__':
    app.run(host=FLAGS.host, port=FLAGS.port)