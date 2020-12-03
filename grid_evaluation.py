import numpy as np
import _pickle as pickle
from contextlib import contextmanager
from flow_scheduling import throughput_scheduler,get_best_throughput
from greedy_scheduling import greedy_scheduler
from lower_bound import get_lower_bound


import threading
import _thread

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
        
        
all_results = []
time = 1
lb1 = 1
lb2 = 1
lb3 = 1
for m in [10,50,100,500,1000,2000,5000]:
    iteration = 0
    while iteration < 10:
        if iteration % 1 == 0:
            print("On {} iteration {}. Last competitive ratio: {}".format(m,iteration,time/max(lb1,lb2,lb3)))
            
             
        #m = np.random.randint(10,4000)
        n = 10 #np.random.randint(2,20)
        p = 10 #np.random.randint(2,20)
        c = 10 #np.random.randint(1,20)
        worker_disparity = 0.5 #np.random.rand()
        beta_max = 10
        delta_max = 5
        # generate jobs    
        rands = np.random.rand(c)
        rands = rands/ sum(rands)
        class_frequencies = np.round(rands*m).astype(int)
        all_jobs = []
        job_classes = []
        for i in range(len(class_frequencies)):
            jobs = np.random.rand(class_frequencies[i],n,p)*worker_disparity  + (1-worker_disparity)
            all_jobs.append(jobs)
            for count in range(class_frequencies[i]):
                job_classes.append(i)    
        jobs = np.concatenate(all_jobs, axis = 0)
        
        # parallelization limits
        beta = np.random.randint(1,beta_max,size = n)
        # batching limits
        delta = np.random.randint(1,delta_max,size = n)
        
        # constrain that all batchable tasks have same time
        for t in range(len(beta)):
            if beta[t] > 1: # task is batchable
                for w in range(p):
                    jobs[:,t,w] = np.random.rand()*worker_disparity  + (1-worker_disparity)
        
        # get lower bounds
        throughput,_ = get_best_throughput(jobs,delta,beta)
        lb1 = len(jobs) / throughput
        lb2,lb3 = get_lower_bound(jobs,delta,beta)
        
        
        # compute schedule
        try:
            with time_limit(m/10):
                schedule, completed, time = throughput_scheduler(jobs,delta,beta,job_classes)
                schedule2,completed2,time2  = greedy_scheduler(jobs,delta,beta,job_classes)
        except:# TimeoutException as e:
            print("Timed out")
            continue
        
        best_to_worst = np.max(np.max(jobs,axis = 2) / np.min(jobs,axis = 2))
        
    #                        print("Throughput-based lower bound: {}".format(lb1))
    #                        print("Work-based lower bound: {}".format(lb3))
    #                        print("Bottleneck-based lower bound: {}".format(lb2))
    #                        print("Flow-Scheduler time: {}".format(time))
    #                        print("Greedy Scheduler time: {}".format(time2))
    #                        print("Best worker is {} times better than worst worker for a task".format(best_to_worst))
        
        result = {"lb": max(lb1,lb2,lb3),
                  "flow":time,
                  "greedy":time2,
                  "ratio":best_to_worst,
                  "m":m,
                  "n":n,
                  "p":p,
                  "c":c,
                  "disparity":worker_disparity,
                  "delta_max":delta_max,
                  "beta_max":beta_max
                }
        #result = [lb1,lb2]
        
        
        all_results.append(result)
        iteration += 1
        
    with open("fixed_results.cpkl","wb") as f:
        pickle.dump(all_results,f)
        

                        