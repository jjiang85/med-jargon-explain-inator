executable = model/run_def_simplifier.sh
getenv = True
error = condor_logs/simplifier.error
log = condor_logs/simplifier.log
notification = always
transfer_executable = false
request_memory = 8*1024
request_GPUs = 1
+Research = True
Queue