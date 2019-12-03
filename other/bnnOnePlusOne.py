import nevergrad as ng

def sumX(x):
    return sum(x)



optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(instrumentation=2, budget=100000000)

recommendation = optimizer.minimize(sumX)
print(recommendation)