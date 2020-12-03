"""
Computes small shop optimal schedule by brute force
"""

import numpy as np
import queue
import copy
from itertools import combinations 


def get_lower_bound(jobs,delta,beta):
    """
    jobs - [m,n,p] array with m = job, n = task, p = worker and jobs[i,j,k] representint
    work efficiency for that job, task and worker
    delta - [n] array - parallelization limits for each task
    beta -[n] array - batching limits for each task
    """
    # calculate lower bound by summing work (divided by max efficiency)
    
    max_eff = np.mean( np.max(jobs,axis = 2), axis = 0) # average max across all jobs
    
    
    n_jobs = jobs.shape[0]
    n_workers = jobs.shape[2]
    
    total_weighted_work_time = sum(n_jobs/max_eff) /n_workers
    temp1 = 1.0/ np.max(jobs,axis = 2)
    temp2 = np.sum(temp1,axis = 0)
    temp3 = temp2/beta
    temp4 = np.sum(temp3)
    temp5 = temp4/n_workers
    total_required_work_time = temp5

#    # calculate lower bound by identifying the job that takes the longest to go through
    fastest_job_tasks = np.max(jobs,axis = 2)
    times = 1/fastest_job_tasks
    sum_times = np.sum(times,axis = 1)
    worst_job_time = np.max(sum_times)     
#    
#    # find best throughput for each task
#    throughput = np.zeros(len(delta))
#    avg_jobs = np.mean(jobs,axis = 0)
#    for j in range (len(delta)):
#        # find best delta workers at task
#        sorted_workers = np.sort(avg_jobs[j,:])
#        throughput[j] = sum(sorted_workers[-delta[j]:]) * beta[j]
#     
#    # for each other task, add in the maximum amount of time one job can take
#    worst_job_per_task = np.max((np.min(jobs,axis= 2)),axis = 0)
#    
#    idx = np.argmin(throughput)
#    
#    total_time = worst_job_per_task
#    total_time[idx] = jobs.shape[0] / throughput[idx]
#    total_bottleneck_time = sum(total_time)
    
    return total_required_work_time,worst_job_time
        
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# Specify jobs [jobs,tasks,workers]
m = 50 # num jobs
n = 5 # num tasks per job
p = 10 # num workers
jobs = np.random.rand(m,n,p)
# parallelization limits
delta = np.array([2,1,3,1,2])
# batching limits
beta = np.array([5,10,1,1,2])

# constrain that all batchable tasks have same time
for t in range(len(beta)):
    if beta[t] > 1: # task is batchable
        for w in range(p):
            jobs[:,t,w] = np.random.rand()

lb1 = get_lower_bound(jobs,delta,beta)