# small-shop-scheduler
This repo is still in progress but figured I may as well post what I've got so far.

The small shop task is defined as follows:

-  A number of jobs, each requiring the same set of tasks in the same order to be carried out. (Note that this can be extended to partially completed work by defining trivially small processing times for unneeded tasks).
-  A number of job classes. All jobs of the same job class are identical.
-  A number of workers, each with a separate processing speed for each task, for each job class.
-  A parallelization limit for each task, defining the maximum number of workers that can work on that task at one time.
-  A batching limit for each task, defining the maximum number of jobs that can be processed in the same time as it takes to process a single job. (If a task is not batchable, this limit is 1).
- Goal: minimize the makespan (time to complete all tasks).

Two schedulers are implemented:
- Greedy Scheduler - tries to keep all workers busy on the available task for which they are best suited at all times
![Greedy1](:ims/greedy 15 tasks.png")
- Flow-based scheduler - allocates workers to tasks such that no single task is a bottleneck (i.e. maximize throughput), and assigns tasks according to this allotment.

Two lower bounds are computed:
- Work-based lower bound 
- Throughput-based lower bound

