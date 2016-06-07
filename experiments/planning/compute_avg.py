import numpy as np
import pickle

filename = 'test-horizon-50-r-50-lamb-02-alp-0005-blend-01-disc-099-decay-02.data'
with open(filename, 'rb') as f:
	traj = pickle.load(f)
	f.close()

score = []
for tra in traj:                                                                                                                                     
    score.append([np.sum(t[2] for t in tr) for tr in tra])

avg = np.mean(score, axis = 0)
print avg