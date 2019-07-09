import numpy as np
import glob

for path in glob.glob("./results/*.txt"):
    print(path)
    data = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                sparsity = line.split(',')
                sparsity[0] = sparsity[0][sparsity[0].index('[')+1:]
                sparsity[-1] = sparsity[-1][:-2]
            else:
                runs = line.split(' ')
                m = np.mean(np.float32(runs))
                std = np.std(np.float32(runs),ddof=1)
                se = std/np.sqrt(len(runs))
                data.append((m, se))

    for s, (m, se) in zip(sparsity, data):
        print(s, m, se)
