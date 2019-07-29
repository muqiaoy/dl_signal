from bayes_opt import BayesianOptimization
from train_low_mem_bayes_opt import run
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
import datetime
name = str(datetime.datetime.now().date()) + "-" + str(datetime.datetime.now().time())
f = open(name + ".txt", "w+")
pbounds = {"attn_dropout": (0, 0.5), "relu_dropout": (0, 0.5), "res_dropout": (0, 0.5), 'lr': (0.00001, 0.1)}

optimizer = BayesianOptimization(
    f=run,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

logger = JSONLogger(path="./logs.json")
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)


optimizer.maximize(
    init_points=1,
    n_iter=1,
)

for i in optimizer.max:
    f.write("Max configuration {} {}\n".format(i, optimizer.max[i]))
f.write("\n")
for i, res in enumerate(optimizer.res):
    f.write("Iteration {}: \n\t{}".format(i, res))
f.close()
