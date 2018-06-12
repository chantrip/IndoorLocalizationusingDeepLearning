# coding: utf-8

# In[19]:


import atexit
import time
import docker
import os
import sys
import argparse
from datetime import datetime
#python run_docker_client.py tewichtete31/indoorlocalization http://192.168.1.102/indoor


# In[20]:



# In[21]:


image = sys.argv[1]
url = sys.argv[2]
#image = "tewichtete31/indoorlocalization"
#url = "http://192.168.1.102/indoor"


# In[22]:


client = docker.from_env()


# In[23]:


#pull image


# In[24]:


import multiprocessing
threadpoolSize = multiprocessing.cpu_count()
threadpoolSize = 4
print(str(datetime.now())+"\t"+"Threadpool size: "+str(threadpoolSize))


# In[25]:


containers=[None]*threadpoolSize
for i in range(0,threadpoolSize):
    #containers[i]=client.containers.run("tewichtete31/indoorlocalization",environment=["url=http://192.168.1.102/indoor"],detach=True)
    containers[i]=client.containers.run(image,environment=["url=" + str(url)],detach=True)
    print(str(datetime.now())+"\t"+str(containers[i]) + " running")


# In[26]:


while True:
    for i in range(0,threadpoolSize):
        containers[i].reload()
        if containers[i].status == "exited":
                containers[i].remove()
                print(str(datetime.now())+"\t"+str(containers[i]) + " exited")              
                
                containers[i]=client.containers.run(image,environment=["url=" + str(url)],detach=True)          
                print(str(datetime.now())+"\t"+str(containers[i]) + " running")
    time.sleep(2)

