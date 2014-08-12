import skneuro
from skneuro import workflows as wf

optJsonFile = "opt.json"
opt = wf.loadJson(optJsonFile)


wf.knottWorkflow(opt)