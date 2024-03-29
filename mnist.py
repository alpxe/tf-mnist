import cv2
import base64
import numpy as np
import tensorflow as tf
import json
import web

urls = (
    '/tensorflow/mnist', 'mnist',
    '/tensorflow/test', 'test'
)


class test:
    def GET(self):
        return "hello world"

    pass


class mnist:
    def POST(self):
        web.header('Content-Type', 'application/json')
        web.header('Access-Control-Allow-Origin', '*')
        web.header('Access-Control-Allow-Credentials', 'true')

        params = web.input()
        base_str = params.get("base")

        base_str = base_str.replace('data:image/jpeg;base64,', '')
        base_str = base_str.replace(' ', '+')

        img_data = base64.b64decode(base_str)
        img_array = np.fromstring(img_data, np.uint8)

        img = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (28, 28))

        _bytes = bytearray(img)
        np_arr = np.array(_bytes)
        print(np_arr.shape)

        resimg = np.reshape(np_arr, [1, 28, 28, 1])

        with tf.Session(graph=tf.Graph()) as sess:
            # 读取完成的模型
            # 读取完成的模型
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.TRAINING], "model_data/")
            # # 输入
            is_training = sess.graph.get_tensor_by_name("holder/is_training:0")

            img = sess.graph.get_tensor_by_name("holder/input_image:0")

            softmax = sess.graph.get_tensor_by_name("logit/softmax_1:0")
            _res = sess.run(softmax, feed_dict={img: resimg, is_training: False})

            data = {"num": _res[0].tolist()}
            print(np.argmax(_res[0].tolist()))
            pass

        return json.dumps(data)


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
