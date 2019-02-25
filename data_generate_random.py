import os
import random
import sys


fname = sys.argv[1]
num_customers = int(sys.argv[2])
num_items = int(sys.argv[3])
max_basket = int(sys.argv[4])

with open(fname, 'w') as f:
    for c in xrange(num_customers):
        items = [random.randint(1, num_items) \
                 for i in xrange(random.randint(1, max_basket))]
        f.write(' '.join([str(j) for j in [c] + items]) + '\n')
