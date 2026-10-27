Hi Claude!

I have 100$ in cloud credits, so I want to use them for conducting some experiments for my Master's research.
The current problem I want to solve boils down to essentially this:
I want to find an optimization scheme that allows me to decompose a scene into primitive shapes (mostly superquadrics). It should decompose it based on semantic labels.
The long-term goal of this project is to train a model that can predict such decomposition with minimal input and as few test-time optimization as possible.
However, I have not even managed yet to converge a classic optimization overfit on a single scene resulting in a satisfying decomposition.
Thus, your job is now to try all directions you can think of to achieve a satisfying decomposition.
The data you can test on are three downsampled replica scenes in data/.
However, also feel free to download your own data, in the end this should also work on e.g. ScanNet, but for now Replica would already be great.
Some more advice:
1. Don't rely on standard metrics like CD or SDF reconstruction to judge the decomposition quality. From my experience, they could e.g. lead to decomposition containing 10000s of primitives, while I am much more interested in large, sensible decompositions a human would also make. So try to find good evaluation to judge your own progress.
2. When researching it is worth to pursue a certain direction for a bit longer, even if it does not seem promising.
3. Always make cheap experiments first, before trying anything time-intensive.
4. Document whatever you try, so I can understand it later when I look at this branch.
5. This should still be a private GitHub repo. Thus, don't commit any large files etc and keep everything clean and organized.

Good luck!