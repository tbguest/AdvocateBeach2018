

import numpy as np
import json

data = {}
data ['key1'] = np.array([1,2,3])
data ['key2'] = np.array([4,5,6])

data2 = {}
data2 ['key1'] = np.array([1,2,3]).tolist()
data2 ['key2'] = np.array([4,5,6]).tolist()



with open('data.json', 'w') as fp:
    json.dump(data2, fp)


with open('data.json', 'r') as fpp:
    ldata = json.load(fpp)

np.array(ldata['key1'])
