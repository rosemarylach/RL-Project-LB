## source 
## Python3 version 
## https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution/12344609#12344609
## Original post
## http://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution/1557906#1557906

'''I can also call timing.log from within my program if there are significant stages within the program I want to show. 
But just including import timing will print the start and end times, and overall elapsed time. 
(Forgive my obscure secondsToStr function, it just formats a floating point number of seconds to hh:mm:ss.sss form.)
'''

import atexit
from time import time, strftime, localtime
from datetime import timedelta

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed = None):
    line = "="*40
    print(line)
    print(secondsToStr(), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    else:
        end = time()
        elapsed = end - start
        print("Elapsed time:", secondsToStr(elapsed))
    print(line)
    print()

def endlog():
    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

# How to run
start = time()
atexit.register(endlog)
log("Start Program")