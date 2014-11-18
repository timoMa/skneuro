import skneuro
from skneuro import workflows as wf



optJsonFile = "opt.json"
optFile = wf.loadJson(optJsonFile)


wf.neuroproofWorkflow(optFile)