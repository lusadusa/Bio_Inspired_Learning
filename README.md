# Bio_Inspired_Learning

## About this project
Individual project of Matteo Lusvarghi (5470641) for Bio-Inspired Intelligence and Learning for Aerospace Application Course. 
  The hereby project is an implementation of the REINFORCE algorithm and its application to the Lunar Lander environment
from the box2d folder of the OpenAI gym library. 
  The implications and the results obtained from this simulation will be analyzed in the correspondent report. 

## Prerequisites
A list of the operations that I performed before effectively executing the notebook:
- use anaconda to create a python environment 
- install the following packages in the environment: *torch* and *gym*
- use Jupyter to execute the notebook
- use the following line of code to install the box2d folder in order to access the Lunar Lander environment
  ```
  pip install gym[box2d]
  ```
  If a message of error pops up after running this line, ignore it and go on since it doesn't affect the behaviour of the code.  
## Explanation
- *LunarLander_reinforce.ipynb* is the notebook to run to obtain the results.
- *utils.py* and *parallel_env.py* are auxiliary files from which some useful functions are called in the main script.
- *gym_master* is the folder inside of which all the documentation regarding the openAI gym library is stored, including the 
  Lunar Lander environment script.
