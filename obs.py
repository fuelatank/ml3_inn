
import tensorflow as tf
import numpy as np
from baseObs import Observation, BaseTFObservation

INDICES = {'main': 0, 'oc': 120, 'ot': 226, 'ac': 237, 'at': 343, 'yn': 348, 'age': 350, 'r': 360}

class TFObservation(BaseTFObservation):
    def convert(self, obs):
        r = []
        for d in obs.data:
            if isinstance(d, np.ndarray):
                r.append(tf.constant([d], dtype=tf.float32))
            else:
                a = np.zeros((105,))
                #print(a, d, self.type, self.valids)
                a[d] = 1
                r.append(tf.constant([a], dtype=tf.float32))
        self.data = r
        index = INDICES[obs.type]
        valids = np.zeros((361,))
        for i in obs.valids:
            valids[i+index] = 1
        self.valids = tf.constant(valids, dtype=tf.float32)
        self.typeIndex = INDICES[obs.type]

def stackObs(obses):
    datas = []
    for i in range(len(obses[0][0].data)):
        datas.append(tf.concat([obs[0].data[i] for obs in obses], 0))
    valids = tf.stack([obs[0].valids for obs in obses])
    return datas, valids

def stackObsTorch(obses):
    datas = []
    for i in range(len(obses[0][0].data)):
        datas.append(torch.cat([obs[0].data[i] for obs in obses], 0))
    valids = torch.stack([obs[0].valids for obs in obses])
    return datas, valids