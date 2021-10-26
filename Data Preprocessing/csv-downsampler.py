#!/usr/bin/env python
# coding: utf-8

# In[3]:


"""
    This code block downsamples each csv with 4.5M data into n data
"""
import pandas as pd
import os
import glob
import numpy as np
import multiprocessing
import string
import random
import time

def multiprocessing_func(csv_files):
    #print('Chunked dataset shape in this process: {}'.format(x.shape))
    ''' Use datatuples to write into text file'''
    for csv in csv_files:
        print("Now working on: {}".format(csv))        
        dataset = pd.read_csv(csv)
        dataset = dataset.sample(n=100000) # RANDOM Down sample to 250025
        
        length = 5 # length of filename
        randomstr = ''.join(random.choices(string.ascii_letters+string.digits,k=length))
        filename = randomstr +'.csv'
        filepath = r"/home/muddi004/muddi/GIANT/downsized-100000" + filename
        dataset.to_csv(filepath, index = False, header = True)
    
        print('Done Writing {}'.format(csv))


# use glob to get all the csv files 
# in the folder
csvDirectory = r"/home/muddi004/muddi/GIANT/Source-CSV" # Original CSV Source
csv_files = glob.glob(os.path.join(csvDirectory, "*.csv"))
csv_files = csv_files[:120]
start = time.time()
print("Starting now")
n_proc = int(os.environ.get('SLURM_CPUS_PER_TASK', '1')) # TODO: Change number of available processors
chunks = np.array_split(csv_files, n_proc) # Chunk CSV file paths
print(len(csv_files))
print(len(chunks))

processes=[] #Initialize the parallel processes list
for i in np.arange(0,n_proc):    
    """Execute the target function on the n_proc target processors using the splitted input""" 
    p = multiprocessing.Process(target=multiprocessing_func,args=(chunks[i],))
    processes.append(p)
    p.start()
for process in processes:
    process.join()

end = time.time()
print("Total time required: {}".format(end - start))


# In[4]:


'''
import pandas as pd
import os
import glob
import numpy as np
import multiprocessing
import string
import random
import time

csvDirectory = r"/home/muddi004/muddi/GIANT/"
csv_files = glob.glob(os.path.join(csvDirectory, "*.csv"))

for csv in csv_files[:1]:
    print("Now working on: {}".format(csv))
    dataset = pd.read_csv(csv)
    dataset = dataset.sample(n=250000) # RANDOM Down sample to 250025
    dataset.to_csv(r"/home/muddi004/muddi/GIANT/downsized-csv/test.csv", index = False, header = True)

n_proc = 40 # TODO: Change number of available processors
chunks = np.array_split(csv_files, n_proc)

print(chunks[38].shape)
'''


# In[ ]:




