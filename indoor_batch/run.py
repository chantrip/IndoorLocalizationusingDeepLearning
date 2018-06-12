import subprocess
import time
import multiprocessing
from datetime import datetime

threadpoolSize = multiprocessing.cpu_count()

processes=[None]*threadpoolSize
comm = ['env','url=http://192.168.1.102/indoor','python', 'client_docker.py']

for i in range(0,threadpoolSize):
    processes[i] = subprocess.Popen(comm, shell=True)
    print(str(datetime.now())+"\t"+str(processes[i].pid) + " running")

while True:
    for i in range(0,threadpoolSize):
        #Check if process is terminated
        if processes[i].poll() is not None:
            print(str(datetime.now())+"\t"+str(processes[i].pid) + " exited")              
            
            processes[i] = subprocess.Popen(comm, shell=True)
            print(str(datetime.now())+"\t"+str(processes[i].pid) + " running")
    time.sleep(2)
