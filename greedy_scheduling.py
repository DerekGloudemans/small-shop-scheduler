import numpy as np


def greedy_scheduler(jobs,delta,beta,job_classes):
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
    
    # Assign jobs to workers using order and allocation
    schedule = []                     # track all tasks assigned
    completed = np.zeros([m,n])       # track job tasks completed
    #stage_completed = np.zeros([m,n]) # track job tasks completed this stage
    in_progress = []                  # track in-progress job-task-worker combos
    w_avail = np.ones([p])           # track available workers
    time = 0
    while np.min(completed) != 1:
        #print("Time: {}  Percent of tasks done: {}".format(time,np.sum(np.floor(completed))/(m*n)))
        if sum(w_avail) > 0: # at least one worker is available
            for k in range(p): # for each worker
                if w_avail[k] == 1:
                    task = None
                    
                    # available is 1 for jobs that are free and pertinent to worker
                    available = np.ones(completed.shape)
                    available[:,1:] = np.floor(completed[:,:-1]) # precedence constraint met if 1
                    available = np.multiply(1-np.ceil(completed),available) 
                    
                    # only consider tasks that are below parallelization limit
                    task_ip = np.zeros(n)
                    for item in in_progress:
                        j = item[2]
                        task_ip[j] += 1
                    task_station_avail = (task_ip < delta).astype(int)
                    
                    avail_tasks = np.multiply(np.sum(available,axis = 0),task_station_avail)
                                        
                    last_batch_tasks = np.where(np.min(completed,axis = 0) == 1)[0]
                       
                    if sum(avail_tasks) == 0:
                        pass
                    
                    # select a task by: 
                    # best worker relative efficiency * available/batch_max_size
                    batch_efficiency = np.minimum(avail_tasks,beta)/beta
                    for idx in range(n):
                        if n in last_batch_tasks:
                            batch_efficiency[n] = 1
                            
                    relative_efficiency = jobs[:,:,k] / np.max(jobs,axis = 2)
                    batch_efficiency = np.broadcast_to(batch_efficiency,(m,)+batch_efficiency.shape)

                    efficiency = relative_efficiency * batch_efficiency 
                    
                    
                    # find most efficient task
                    flat_idx = np.argmax(efficiency)
                    task = flat_idx % n
                    job = flat_idx // n
                    
                    if np.max(efficiency) == 0:
                        task = None
                        
                    # find first beta[task] jobs in job_order with available[job,task] = 1
                    if task is not None:
                        
#                        if beta[task] == 1:
#                            task_jobs = [job]
#                        else:
                        task_jobs = []
                        job_idx = 0
                        while len(task_jobs) < min(avail_tasks[task],beta[task]) and job_idx < m:
                            if available[job_idx,task] == 1:
                                task_jobs.append(job_idx)
                            job_idx += 1

                        # now we have worker, task, and job(s) 
                        try:
                            completion_time = time + (1.0/jobs[task_jobs[0],task,k])
                        except:
                            #print(task_jobs)
                            pass
                        
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
                        
                
        # increment time, etc
        
        # find first item that completes
        next_time = 1e+10
        for idx,item in enumerate(in_progress):
            if item[0] < next_time:
                next_time = item[0]
        
        time = next_time
            
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








if __name__ == "__main__":
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
        jobs = np.random.rand(class_frequencies[i],n,p)/1.25 + 0.2
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
        
    greedy_scheduler(jobs,delta,beta,job_classes)