import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_schedule(schedule,lb = None,xmax = None, colors = None):
    n_tasks = 0
    n_workers = 0 
    


    for item in schedule:
        task = item["task"]
        worker = item["worker"]
        if task > n_tasks:
            n_tasks = task
        if worker > n_workers:
            n_workers = worker
    
    n_workers += 1
    n_tasks += 1
    
    fig,ax = plt.subplots()
    major_ticks = np.arange(0,n_workers)
    ax.set_yticks(major_ticks)
    ax.set_ylim(-0.5,n_workers-0.5)
    ax.set_xlabel("Time",fontsize = 20)
    ax.set_ylabel("Worker",fontsize = 20)
    ax.grid(True)
    if colors is None:
        colors = np.random.rand(n_tasks,3)
        colors[:,2] = 0.4
    for item in schedule:
        start = item["start_time"]
        duration = item["end_time"] - item["start_time"]
        task = item["task"]
        worker = item["worker"]
        jobs = item["jobs"]
        
        ax.broken_barh([(start,duration)], (worker-0.5,1), color =colors[task])

    if lb:
        ax.broken_barh([(lb-1,1)], (-0.5,n_workers+0.5), color = (1,1,1))
        ax.broken_barh([(lb,2)], (-0.5,n_workers+0.5), color = (1,0,0))
        ax.broken_barh([(lb+2,1)], (-0.5,n_workers+0.5), color = (1,1,1))

    if xmax:
        ax.set_xlim(0,xmax)

    # plot all tasks for each job on one bar



with open("random_results.cpkl", "rb") as f:
    results = pickle.load(f)
    
# aggregators
jobs = []
job_classes = []
workers = []
tasks = []
worker_disparities = []    
greedy_ratios = []
flow_ratios = []
flow_wins = []
flow_times = []

for item in results:
    jobs.append(item["m"])
    tasks.append(item["n"])
    job_classes.append(item["c"])
    workers.append(item["p"])
    worker_disparities.append(item["disparity"])
    greedy_ratios.append(item["greedy"]/item["lb"])
    flow_ratios.append(item["flow"]/item["lb"])
    flow_wins.append(int(item["flow"] < item["greedy"]))
    flow_times.append(item["flow"])
    
# get average competitive ratio over all trials for both algs
cr_flow = sum(flow_ratios)/len(flow_ratios)
cr_greedy = sum(greedy_ratios)/len(greedy_ratios)

## plot competitive ratio versus number of jobs
#plt.figure()
#plt.scatter(jobs,flow_ratios)
##plt.scatter(jobs,greedy_ratios)
#m,b = np.polyfit(np.array(jobs),np.array(flow_ratios),1)
#plt.plot([0,max(jobs)],[b,b+m*max(jobs)],color = (1,0,0))
#plt.xlim([0,max(jobs)])
#plt.ylim([0.9,5])
#plt.xlabel("Number of Jobs",fontsize = 16)
#plt.ylabel("Competitive Ratio",fontsize = 16)
#
#plt.figure()
#plt.scatter(job_classes,flow_ratios)
##plt.scatter(job_classes,greedy_ratios)
#m2,b = np.polyfit(np.array(job_classes),np.array(flow_ratios),1)
#plt.plot([0,max(job_classes)],[b,b+m2*max(job_classes)],color = (1,0,0))
#plt.xlim([1,20])
#plt.ylim([0.9,5])
#plt.xlabel("Number of Unique Job Classes",fontsize = 16)
#plt.ylabel("Competitive Ratio",fontsize = 16)
#
#plt.figure()
#plt.scatter(tasks,flow_ratios)
##plt.scatter(tasks,greedy_ratios)
#m,b = np.polyfit(np.array(tasks),np.array(flow_ratios),1)
#plt.plot([0,max(tasks)],[b,b+m*max(tasks)],color = (1,0,0))
#plt.xlim([1,20])
#plt.ylim([0.9,5])
#plt.xlabel("Number of Tasks per Job",fontsize = 16)
#plt.ylabel("Competitive Ratio",fontsize = 16)
#
#plt.figure()
#plt.scatter(workers,flow_ratios)
##plt.scatter(workers,greedy_ratios)
#m,b = np.polyfit(np.array(workers),np.array(flow_ratios),1)
#plt.plot([0,max(workers)],[b,b+m*max(workers)],color = (1,0,0))
#plt.xlim([1,20])
#plt.ylim([0.9,5])
#plt.xlabel("Number of Workers",fontsize = 16)
#plt.ylabel("Competitive Ratio",fontsize = 16)
#
#plt.figure()
#plt.scatter(worker_disparities,flow_ratios)
##plt.scatter(worker_disparities,greedy_ratios)
#m,b = np.polyfit(np.array(worker_disparities),np.array(flow_ratios),1)
#plt.plot([0,max(worker_disparities)],[b,b+m*max(worker_disparities)],color = (1,0,0))
#plt.xlim([0,1])
#plt.ylim([0.9,5])
#plt.xlabel("Worker Disparity",fontsize = 16)
#plt.ylabel("Competitive Ratio",fontsize = 16)
## plot competitive ratio versus number of job classes
## plot competitive ratio verus number of workers
## plot competitive ratio versus number of tasks
## plot competitive ratio verus worker disparity    
#
#with open("competitive_ratios.cpkl","rb") as f:
#    results = pickle.load(f)
#    
#    
#bottleneck_lb = []
#throughput_lb = []
#work_lb = []
#
#bottleneck = 0
#work = 0
#throughput = 0
#for item in results:
#    bottleneck_lb.append(item[1])
#    throughput_lb.append(item[0])
#    work_lb.append(item[2])
#    
#    if item[1] > item[2] and item[1] > item[0]:
#        bottleneck += 1
#    elif item[2] > item[1] and item[2] > item[0]:
#        work += 1
#    else:#        throughput += 1
    
#print("Bottleneck: {}  Throughput: {}  Work:  {}".format(bottleneck/10000,throughput/10000,work/10000))


with open("example_schedule4.cpkl","rb") as f:
    colors = np.random.rand(100,3)
    colors[:,2] = 0.5
    schedule,schedule2,lb = pickle.load(f)
    plot_schedule(schedule,lb =lb, xmax = 700,colors = colors)
    plot_schedule(schedule2,lb =lb, xmax = 700,colors = colors)