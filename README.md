# applied_deep_learning_final

Project Structure and Environment
Standardized Directory Structure: Organize the repository with clear directories to improve navigation:
data/ (with subdirectories like raw/ and processed/)
src/ (for source code and scripts)
notebooks/ (for exploratory analysis)
models/ (for trained models, and checkpoints)
docs/ (for documentation)
results/ (for output files like plots or metrics)
Use Virtual Environments: Ensure all team members use the same package versions by utilizing a virtual environment and including a requirements file (e.g., requirements.txt for Python) in the repository.
Handle Jupyter Notebooks Carefully: Clear notebook outputs before committing to minimize merge conflicts and repository bloat. 
Dataloaders - try to use https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
Links to an external site.
 if you want, this should expedite things down the road for you (I typically have a src/dataloader.py file) that anyone can create a new one with the same style.

