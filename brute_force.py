"""
Computes small shop optimal schedule by brute force
"""

import numpy as np
import queue
import copy
from itertools import combinations 
from lower_bound import get_lower_bound
from flow_scheduling import get_best_throughput

def get_possible_steps(time,in_progress,complete,t_avail,schedule):
    """
    time - float > 0
    in_progress - list of (completion_time,[job idx,task_idx])
    complete - m x n array with 1 if complete, 0.5 if in progress, 0 if unstarted
    t_avail - p array with next available time of each worker
    schedule - list of all previously assigned tasks
    """
    possible_next_steps = []
    # perform some checks to speed up computation - if we're already over best schedule time, prune
    if time > best_time:
        return []
    
    if np.sum(complete)/time < len(complete)/best_time * 0.5:
        return []
    
    # perhaps if we're less than half as efficient as best, prune
    
    
    # first, consider waiting until next job finishes to do anything (change in_progress, complete,time)
    if len(in_progress) > 0:
        next_schedule    = copy.deepcopy(schedule)
        next_in_progress = copy.deepcopy(in_progress)
        next_t_avail     = copy.deepcopy(t_avail)
        next_complete    = copy.deepcopy(complete) 
        
        # find first item that completes
        next_time = np.inf

        for idx,item in enumerate(next_in_progress):
            if item[0] < next_time:
                next_time = item[0]
               
        done = []
        for idx,job in enumerate(next_in_progress):
            if job[0] == next_time: # job finishes earliest
                task = job[1]
                job_batch = job[2]
                
                # change complete
                for j in job_batch:
                    next_complete[j,task] = 1
                
                done.append(idx)
        
        # remove from in progress
        done.reverse()
        for idx in done:
            del next_in_progress[idx]     
        
        possible_next_steps.append((next_time,next_in_progress,next_complete,next_t_avail,next_schedule))
    
    # which tasks are available (below parallelization limit)?
    task_ip = np.zeros(n)
    for item in in_progress:
        task = item[1]
        task_ip[task] += 1
    
    available_tasks = []
    for j in range(n):
        if task_ip[j] < delta[j]:
            available_tasks.append(j)

    available_job_tasks = []     # all pairs (i,j) of job tasks available
    # of available tasks, which have jobs available?
    for task in available_tasks:
        for i in range(m):
            if (task == 0 or complete[i,task-1] == 1) and complete[i,task] == 0:
                available_job_tasks.append((i,task))
    
    # which workers are available?
    available_workers = []
    for k in range(p):
        if t_avail[k] <= time:
            available_workers.append(k)
            
    # convert list of job-task pairs into list of batch-task pairs (i.e. get all combinations at batching limit or less)   
    all_combos = []
    for j in range(n):
        job_list = []
        for pair in available_job_tasks:
            if pair[1] == j:
                job_list.append(pair[0])
        
        combos = list(combinations(job_list,min(beta[j],len(job_list)))) # each item in combo is a list,each item in list is a job
        if len(combos[0]) == 0:
            combos = []
        all_combos.append(combos)
        
    for j,combos in enumerate(all_combos): # j is task number
        for combo in combos:
            for worker in available_workers:
                next_schedule    = copy.deepcopy(schedule)
                next_in_progress = copy.deepcopy(in_progress)
                next_t_avail     = copy.deepcopy(t_avail)
                next_complete    = copy.deepcopy(complete)

                # append to schedule  worker on task j for the given combo of jobs
                step = {"worker":worker,
                        "start_time": time,
                        "task": j,
                        "jobs": combo}
                next_schedule.append(step)
                
                # calculate completion time - we use time for combo[0] because either that's the only job or time is the same across all jobs for that task
                completion_time = time + (1.0 / jobs[combo[0],j,worker])
                # add this job to in-progress
                next_in_progress.append((completion_time,j,combo))
                
                # set t_avail to finish time for task
                next_t_avail[worker] = completion_time
                
                # set completed to 0.5
                for job in combo:
                    next_complete[job,j] = 0.5
            
                possible_next_steps.append((time,next_in_progress,next_complete,next_t_avail,next_schedule))
      
    return possible_next_steps

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# Specify jobs [jobs,tasks,workers]
m = 2 # num jobs
n = 5 # num tasks per job
p = 2 # num workers
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

lb1,lb2 = get_lower_bound(jobs,delta,beta)
lb3 = get_best_throughput(jobs,delta,beta)

print("Lower bound: {} {} {}".format(lb1,lb2,lb3))

# define arrays for keeping track of schedule
time = 0
complete = np.zeros([m,n]) # 1 if complete, 0 if not started and 0.5 if in progress
t_avail = np.zeros(p) # item k - time at which worker k will be available again
in_progress = [] # each item is (completion_time,job_task)
schedule  = [] # each item in schedule is dict with keys: worker, start time, task, job(s) 

# each item appended is (time,in_progress,complete,t_avail,schedule) which uniquely defines schedule thus far
branch_stack = [] 
branch_stack.append((time,in_progress,complete,t_avail,schedule))

# keep track of best schedule
best_schedule = None
best_time = np.inf
count = 0
while len(branch_stack) > 0: # there is at least one unexplored schedule
    
    # pop top off branch_stack to get complete, t_avail, etc
    (time,in_progress, complete,t_avail,schedule) = branch_stack.pop()


    if np.min(complete) == 1:
        #finish time is highest t_avail
        finish = max(t_avail)
        if finish < best_time:
            best_time = finish
            best_schedule = copy.deepcopy(schedule)
            
    
    # given complete, t_avail, generate a list of all valid assignments (always include option to wait until next task finishes)
    possible_steps = get_possible_steps(time,in_progress,complete,t_avail,schedule)
    
    # for each assignment, append to branch_stack
    for step in possible_steps:
        branch_stack.append(step)
        
        
    count += 1
    if count % 1000 == 0:
        #print(complete)
        print("On iteration {}. Best time: {}".format(count,best_time))
    
