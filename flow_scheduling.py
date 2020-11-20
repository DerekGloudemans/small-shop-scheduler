"""
Computes small shop optimal schedule by brute force
"""

import numpy as np
import queue
import copy
from itertools import combinations 
from mip import *

def get_best_throughput(jobs,delta,beta):
    """
    Finds the maximal throughput attainable simultaneously across all tasks
    which can be used as a lower bound, since the whole pipeline cannot output
    on average more than any single component's throughputs, and time varying methods
    can be summarized as an average, time constant worker allotment.
    
    jobs - [m,n,p] array with m = job, n = task, p = worker and jobs[i,j,k] representint
    work efficiency for that job, task and worker
    delta - [n] array - parallelization limits for each task
    beta -[n] array - batching limits for each task
    """
    n = jobs.shape[1]
    p = jobs.shape[2] 
    # get average workload across all jobs
    avg_jobs = np.mean(jobs,axis = 0)
    
    # multiply by batching limit per task
    batches = np.broadcast_to(beta,(p,)+beta.shape).transpose()
    avg_batched_jobs = np.multiply(avg_jobs,batches)
    
    all_results = {}
    for j_star in range(n): # consider as a separate case each task being the bottleneck
        # now maximize minimum sum of avg_batched_jobs * x for each task
        m = Model(sense = MAXIMIZE)
        x = [[ m.add_var(name = "x_{},{}".format(j,k),lb = 0, ub = 1) for k in range(p) ] for j in range(n)]
        y = [m.add_var(name = "y_{}".format(j)) for j in range(n)] # will equal sum of work per task
        
        # subject to: sum of x's per worker <= 1
        for k in range(p):
            m += xsum(x[j][k] for j in range(n)) <= 1
            
        # sum of x's per task <= delta
        for j in range(n):
            m += xsum(x[j][k] for k in range(p)) <= delta[j]
            
        # set y = sum of xs
        for  j in range(n):
            m += (y[j] - xsum(x[j][k]*avg_batched_jobs[j,k] for k in range(p))) <= 0
            
        # lastly, set a specific y[j_star] to be the minimum of all y's
        for j in range(n):
            if j != j_star:
                m += (y[j_star] -y[j]) <= 0
        
        m.objective = y[j_star]
        
        # solve
        m.max_gap = 0.0005
        status = m.optimize(max_seconds=300)
#        if status == OptimizationStatus.OPTIMAL:
#            print('optimal solution cost {} found'.format(m.objective_value))
        if status == OptimizationStatus.OPTIMAL or status == OptimizationStatus.FEASIBLE:
            result = np.zeros([n,p])
            throughput = m.objective_value
            for v in m.vars:
               name = v.name
               if "x" in name:
                   j = int(name.split("_")[1].split(",")[0])
                   k = int(name.split("_")[1].split(",")[1])
                   value = v.x
                   result[j,k] = value
            all_results[j_star] = [result,throughput]
                   
            
        min_throughput = np.inf
        min_key = None
        for key in all_results:
            item = all_results[key]
            if item[1] < min_throughput:
                min_throughput = item[1]
                min_key =  key
        return min_throughput, all_results[key][0]
    
    
def throughput_scheduler(jobs,delta,beta,job_classes,stage_minimum = 50):
    """
    Schedules tasks for the small-shop scheduling problem
     
    jobs - [m,n,p] array with m = job, n = task, p = worker and jobs[i,j,k] representent
        work efficiency for that job, task and worker
    delta - [n] array - parallelization limits for each task
    beta -[n] array - batching limits for each task
    job_classes - [m] array - specifies job type, all jobs with same integer job id 
        have identical processing times
    """
    
    (m,n,p) = jobs.shape
    ### first, get worker allocation to tasks that optimizes throughput
    throughput, allocation = get_best_throughput(jobs,delta,beta)
    
    ### pick a stage time that isn't restrictively small
    # max time for an assigned worker to finish a batch
    time_per_batch = 1.0/jobs
    active = np.ceil(allocation) 
    active = np.broadcast_to(active,(m,)+active.shape)#.transpose()
    time_per_active_batch = np.multiply(time_per_batch, active)
    max_batch_time  = np.max(time_per_active_batch)
    
    # max time for a set number of items to be produced
    stage_minimum_time = stage_minimum/throughput
    
    stage_time = max(stage_minimum_time,max_batch_time)
    # how many stages should there be? Total jobs / throughput per stage_time
    stage_count = np.ceil(m/(throughput*stage_time))
    
    ### order jobs - cluster into groups with average processing time = stage time, 
    # count number of each job class
    classes,counts = np.unique(job_classes,return_counts = True)
    
    # assign jobs as equally as possible per stage
    per_stage_min   = counts // stage_count
    per_stage_extra = counts %  stage_count
    
    
    # main loop
        # keep track of stage time 
        
        # if a worker is free - assign it to a job based on optimal allocation
        
        # as a secondary goal, assign the worker to the specific task 
        #for which it is most optimal relative to other workers on that task
    
        # update completed and in_progress
        
        # if no jobs assignable, advance to next compeletion task time
        
# dummy tasks and such
        
    
# Specify jobs [jobs,tasks,workers]

m = 1000 # num jobs
n = 5 # num tasks per job
p = 10 # num workers

c = 10 # number of job classes
rands = np.random.rand(c)
rands = rands/ sum(rands)
class_frequencies = np.round(rands*m).astype(int)
all_jobs = []
job_classes = []
for i in range(len(class_frequencies)):
    jobs = np.random.rand(class_frequencies[i],n,p)
    all_jobs.append(jobs)
    for count in range(class_frequencies[i]):
        job_classes.append(i)    
jobs = np.concatenate(all_jobs, axis = 0)

# parallelization limits
beta = np.random.randint(1,10,size = n)
# batching limits
delta = np.random.randint(1,10,size = n)

# constrain that all batchable tasks have same time
for t in range(len(beta)):
    if beta[t] > 1: # task is batchable
        for w in range(p):
            jobs[:,t,w] = np.random.rand()

lb = get_best_throughput(jobs,delta,beta)

throughput_scheduler(jobs,delta,beta,job_classes)     