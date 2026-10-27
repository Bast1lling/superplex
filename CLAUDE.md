This project is my master thesis, where I want to experiment with Superquadrics as a primitive family to efficiently decompose larges point cloud scenes into meaningful parts or hierarchies. As of now, the direction this research project will take is not yet clear. Therefore, our workflow should be as follows:

1. I propose an experiment, i.e. directions I want to explore to find out if they are promising, and also to help me  better define my project.
2. You implement this experiment according to my instructions. Focus is on fast iterations, BFS search exploring etc. A new experiment gets its subfolder in experiments/ and consists of its logic in code/, a fastAPI html (threeJS) viewer/ for me to inspect, and a folder collecting all conclusions or quantitative evaluations for documentation in results/.
3. Successful experiments might inspire me to adopt these ideas for my project. I am not yet sure how to do this exactly, but the actual code will go into src/ and will be written by myself.
4. Additionally, there is the papers/ directory which contains repository from related work, so we can directly compare their implementation and adopt useful ideas

All datasets used can be found in .../data/ and should also be stored there. In general the focus is to keep the repository as clean as possible except for the experiment subfolders, they can be as messy as you need :)
Result files larger than 1MB go into the experiment's results/large_files/ subdir, which is git-ignored. Code reading them must still work when they are missing.

I am very open to suggestions and your opinions! Please push back on any thoughts I have if you disagree! In general, before you take decisions that will deviate from my intial idea/instructions, I prefer to be asked/informed beforehand.
Good luck and have fun!

Don't start ablations, experiments etc without asking me!