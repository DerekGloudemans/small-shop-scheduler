"""
Computes small shop optimal schedule by brute force
"""
import time
import numpy as np
import queue
import copy
from itertools import combinations 
from mip import *
from lower_bound import get_lower_bound
from greedy_scheduling import greedy_scheduler
from plot_results import plot_schedule

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
    
    
def throughput_scheduler(jobs,delta,beta,job_classes,stage_minimum = 50,FLEXIBLE = False):
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
    ### 1.) get worker allocation to tasks that optimizes throughput
    throughput, allocation = get_best_throughput(jobs,delta,beta)
    
    ### 2.) Based on this allocation pick stage time 
    
    # max time for an assigned worker to finish a batch
    time_per_batch = 1.0/jobs
    active = np.ceil(allocation) 
    active = np.broadcast_to(active,(m,)+active.shape)#.transpose()
    time_per_active_batch = np.multiply(time_per_batch, active)
    max_batch_time  = np.max(time_per_active_batch)
    
    # max time for a set number of items to be produced
    stage_minimum_time = stage_minimum/throughput
    
    stage_time = max(stage_minimum_time,max_batch_time)
    stage_count = int(np.ceil(m/(throughput*stage_time)))
    
    ### 3.) order jobs
    classes,counts = np.unique(job_classes,return_counts = True)
    
    # assign job classes as equally as possible per stage
    per_stage_min   = counts // stage_count
    per_stage_extra = counts %  stage_count
    per_stage_extra_copy = copy.deepcopy(per_stage_extra)
    
    # jobs are assigned to list of lists - one per stage - each sublist element is a job index
    classes_per_stage = np.zeros([stage_count,len(classes)]) # stage,job class count at that stage
    jobs_per_stage = [[] for i in range(stage_count)]
    extra_per_stage_so_far = np.zeros(stage_count)
    for i in range(len(job_classes)):
        cls = job_classes[i]
        stage_idx = 0
        ASSIGNED = False
        while not ASSIGNED:
            if classes_per_stage[stage_idx,cls] > per_stage_min[cls]: # stage is full
                stage_idx += 1
            elif classes_per_stage[stage_idx,cls] == per_stage_min[cls] and per_stage_extra[cls] > 0: # stage is one from full and needs an extra job
                per_stage_extra[cls] -= 1
                jobs_per_stage[stage_idx].append(i)
                classes_per_stage[stage_idx,cls] += 1
                ASSIGNED = True
            elif classes_per_stage[stage_idx,cls] < per_stage_min[cls]:
                jobs_per_stage[stage_idx].append(i)
                classes_per_stage[stage_idx,cls] += 1
                ASSIGNED = True
            else:
                stage_idx += 1
    
    #now, simply stack all of the stage assignments to get total job ordering for stage 0
    job_order = [item for stage_list in jobs_per_stage for item in stage_list]
    
    ### 4.) Assign jobs to workers using order and allocation
    schedule = []                     # track all tasks assigned
    completed = np.zeros([m,n])       # track job tasks completed
    #stage_completed = np.zeros([m,n]) # track job tasks completed this stage
    in_progress = []                  # track in-progress job-task-worker combos
    w_avail = np.ones([p])           # track available workers
    w_stage_alloc = np.zeros([n,p])   # percentage of stage time worker spends on task
    time = 0
    stage_start = 0
    while np.min(completed) != 1:
        #print("Time: {}  Percent of tasks done: {}".format(time,np.sum(np.floor(completed))/(m*n)))
        if sum(w_avail) > 0: # at least one worker is available
            for k in range(p): # for each worker
                if w_avail[k] == 1:
                    task = None
                    
                    # available is 1 for jobs that are free and pertinent to worker
                    available = np.ones(completed.shape)
                    available[:,1:] = np.floor(completed[:,:-1]) # precedence constraint met if 1
                    available_all = np.multiply(1-np.ceil(completed),available) 
                    available = np.multiply(available_all,active[:,:,k])
                    
                    # only consider tasks that are below parallelization limit
                    task_ip = np.zeros(n)
                    for item in in_progress:
                        j = item[2]
                        task_ip[j] += 1
                    task_station_avail = (task_ip < delta).astype(int)
                    
                    avail_tasks = np.multiply(np.sum(available,axis = 0),task_station_avail)
                    
                    # get set of tasks indexes which have enough available jobs for a batch
                    ready_tasks = np.where(avail_tasks >= beta)[0]
                    
                    last_batch_tasks = np.where(np.min(completed,axis = 0) == 1)[0]
                    some_ready_tasks = np.where(avail_tasks > 0)[0]
                        
                    # select most important task based on stage quotas
                    if len(ready_tasks) > 0:
                        importance = 1- (w_stage_alloc[:,k] / allocation[:,k])
                        for j in range(n):
                            if j not in ready_tasks:
                                importance[j] = -np.inf
                        task = np.argmax(importance)
                      
                    elif len(some_ready_tasks) > 0:  
                        for j in range(n):
                            last_batch = False
                            if j == 0:
                                last_batch = True
                            elif j-1 in last_batch_tasks:
                                last_batch = True
                                
                            if j in some_ready_tasks and last_batch:
                                task = j
                                break                        
                        
                    # find first beta[task] jobs in job_order with available[job,task] = 1
                    if task is not None:
                        task_jobs = []
                        order_idx = 0
                        while len(task_jobs) < beta[task] and order_idx < m:
                            job_idx = job_order[order_idx]
                            if available[job_idx,task] == 1:
                                task_jobs.append(job_idx)
                            order_idx += 1

                        # now we have worker, task, and job(s) 
                        try:
                            completion_time = time + (1.0/jobs[task_jobs[0],task,k])
                        except:
                            print(task_jobs)
                            
                        # update completed
                        for i in task_jobs:
                            completed[i,task] = 0.5
                        
                        # update in_progress
                        in_progress.append([completion_time,task_jobs,task,k])
                        
                        # update schedule
                        step = {"worker":k,
                            "start_time": time,
                            "end_time": completion_time,
                            "task": task,
                            "jobs": task_jobs
                            }
                        schedule.append(step)
                
                        # update w_avail
                        w_avail[k] = 0
                        
                        # update w_stage_alloc
                        w_stage_alloc[task,k] += completion_time / stage_time # proportion of one stage spent on that task
                
        # increment time, etc
        
        # find first item that completes
        next_time = 1e+10
        for idx,item in enumerate(in_progress):
            if item[0] < next_time:
                next_time = item[0]
        
        time = next_time
        # start new stage if necessary
        if time - stage_start >stage_time:
            stage_start = time
            w_stage_alloc *= 0
            
        done = []
        for idx,job in enumerate(in_progress):
            if job[0] == next_time: # job finishes earliest
                task = job[2]
                job_batch = job[1]
                # change complete
                for j in job_batch:
                    completed[j,task] = 1
                    
                worker = job[3]
                w_avail[worker] = 1
                
                done.append(idx)
        
        # remove from in_progress
        done.reverse()
        for idx in done:
            del in_progress[idx]    
    
    
    return schedule, completed, time
        



#################################################################################################
if __name__ == "__main__":
    # Specify jobs [jobs,tasks,workers]
    worker_disparity = 1
    beta_max = 10
    delta_max = 5
    m = 100 # num jobs
    n = 10 # num tasks per job
    p = 10  # num workers
    c = 2 # number of job classes
    start = time.time()
    
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
    
    throughput,_ = get_best_throughput(jobs,delta,beta)
    
    lb1 = len(jobs) / throughput
    lb2= get_lower_bound(jobs,delta,beta)
    
    schedule, completed, end_time = throughput_scheduler(jobs,delta,beta,job_classes)  
    schedule2,completed2,end_time2  = greedy_scheduler(jobs,delta,beta,job_classes)
    
    best_to_worst = np.max(np.max(jobs,axis = 2) / np.min(jobs,axis = 2))
    
    print("Throughput-based lower bound: {}".format(lb1))
    print("Work-based lower bound: {}".format(lb3))
    #print("Bottleneck-based lower bound: {}".format(lb2))
    print("Flow-Scheduler time: {}".format(end_time))
    print("Greedy Scheduler time: {}".format(end_time2))
    print("Best worker is {} times better than worst worker for a task".format(best_to_worst))

    colors = np.random.rand(n,3)
    colors[:,2] = 0.5
#    colors[:,1] = 0.3
#    colors[:,0] = np.arange(0,1,0.2)
    
    elapsed = time.time() - start
    print("Took {} seconds".format(elapsed))
    
    lb = max(lb1,lb2)
    xmax = max(end_time,end_time2)
    plot_schedule(schedule ,lb =lb, xmax = xmax,colors = colors)
    plot_schedule(schedule2,lb =lb, xmax = xmax,colors = colors)