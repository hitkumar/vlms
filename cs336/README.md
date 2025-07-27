
Lecture Notes

Lecture 2
- Pytorch and Resource accounting
- Helpful video about floating point formats: https://www.youtube.com/watch?v=bbkcEiUjehk
- Introduces a lot of good basics about calculating memory and compute requirements for a model. Check notebook for a good review.

Lecture 3
- Talks about some good tricks about LLM hyperparameters
- Weight decay point was interesting, can be tried.
- Stability tricks are good to know, refer to his slides for more details.
- Best resource for practical LLM tricks.

Lecture 4
- MOE
- Talks about the routing function to get top k experts
- Stability issues with MOE
- Load balancing loss is critical to get all experts to be used. Several aspects of this are not yet fully clear including aux z-loss.
- Deepseek series and OLMOe are good papers to review next.
- Implement MOE in vlms.

Lecture 5
- Talks about GPUs and how to use them effectively
- Shared memory and global memory (DRAM)
- Tiling, burst sections, memory coallescing are some key ideas to understand GPU performance for models
- Flash attention walkthrough - tiling, operator fusion and online softmax are key ideas.

Lecture 6
- Talks about benchmarking and profiling to start the lecture.
- Very important to profile, pytorch profiler is one initial tool to use, but use Nvidia profiler for additional insights
- Triton kernels are a good way to fuse operations and improve performance
- Torch compile should be the first thing to try to improve performance
