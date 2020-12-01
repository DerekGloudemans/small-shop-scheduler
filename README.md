# small-shop-scheduler
The small shop task is defined as follows:

-  A number of jobs, each requiring the same set of tasks in the same order to be carried out. (Note that this can be extended to partially completed work by defining trivially small processing times for unneeded tasks).
-  A number of job classes. All jobs of the same job class are identical.
-  A number of workers, each with a separate processing speed for each task, for each job class.
-  A parallelization limit for each task, defining the maximum number of workers that can work on that task at one time.
-  A batching limit for each task, defining the maximum number of jobs that can be processed in the same time as it takes to process a single job. (If a task is not batchable, this limit is 1).

