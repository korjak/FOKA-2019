from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from model import ShowAndTellModel
from vocabulary import Vocabulary
from caption_generator import CaptionGenerator

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
        return render_template('template.html', var1=sentences[0][0],var2=sentences[0][1],var3=sentences[1][0],var4=sentences[1][1],var5=sentences[2][0],var6=sentences[2][1],)
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
          <p><input type=file name=image>
             <input type=submit value=Upload>
        </form>
        '''


if __name__ == '__main__':
    app.run(host=FLAGS.host, port=FLAGS.port)