# Lock-free Concurrent Hash Tables and Parallel Dynamic Programming on the GPU

We study how to best make use of the a GPU to implement lock-free
concurrent hash tables, and on top of that highly parallel dynamic
programming.  We show that the right hash table can be remarkably
effectively parallelized when implemented on a GPU wth a $55\times$
speedup over a single thread on the same GPU.  More importantly the
GPU implementation is around $7.0\times$ better than on a competitive
multi-core CPU, showing that we can take advantage of the massive
parallelism of the GPU even with its much slower thread computation.
We also examine the use of the hash table for a dynamic programming
application. Here the GPU is not as advantageous over the CPU but by
adding more speculative computation the GPU becomes the superior
choice.


