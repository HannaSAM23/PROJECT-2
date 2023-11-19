# PROJECT 2 - PART A.5

# gradient descent optimization with rmsprop for a two-dimensional test function
from math import sqrt
from numpy import asarray
from numpy.random import rand
from numpy.random import seed

# Objective function for two-dimensional input
def objective(x):
    return 3.0 + 4.0 * x[0] + 2.0 * x[1]**2

# Derivative of objective function for two-dimensional input
def derivative(x):
    return asarray([4.0 + 4.0 * x[0], 4.0 * x[1]])

# Gradient descent algorithm with RMSprop
def rmsprop(objective, derivative, bounds, n_iter, step_size, rho):
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    sq_grad_avg = [0.0 for _ in range(bounds.shape[0])]

    for it in range(n_iter):
        gradient = derivative(solution)
        for i in range(gradient.shape[0]):
            sg = gradient[i]**2.0
            sq_grad_avg[i] = (sq_grad_avg[i] * rho) + (sg * (1.0 - rho))

        new_solution = list()
        for i in range(solution.shape[0]):
            alpha = step_size / (1e-8 + sqrt(sq_grad_avg[i]))
            value = solution[i] - alpha * gradient[i]
            new_solution.append(value)

        solution = asarray(new_solution)
        solution_eval = objective(solution)
        print('>%d f(%s) = %.5f' % (it, solution, solution_eval))

    return [solution, solution_eval]

# Seed the pseudo random number generator
seed(1)
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
n_iter = 50
step_size = 0.01
rho = 0.99
best, score = rmsprop(objective, derivative, bounds, n_iter, step_size, rho)
print('Done!')
print('f(%s) = %f' % (best, score))

# gradient descent optimization with adam for a two-dimensional test function

# Objective function for two-dimensional input
def objective(x,y):
    return 3.0 + 4.0 * x + 2.0 * y**2

# Derivative of objective function for two-dimensional input
def derivative(x,y):
    return asarray([4.0 + 4.0 * x, 4.0 * y])

# gradient descent algorithm with adam
def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
	# generate an initial point
	x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
	score = objective(x[0], x[1])
	# initialize first and second moments
	m = [0.0 for _ in range(bounds.shape[0])]
	v = [0.0 for _ in range(bounds.shape[0])]
	# run the gradient descent updates
	for t in range(n_iter):
		# calculate gradient g(t)
		g = derivative(x[0], x[1])
		# build a solution one variable at a time
		for i in range(x.shape[0]):
			# m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
			m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
			# v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
			v[i] = beta2 * v[i] + (1.0 - beta2) * g[i]**2
			# mhat(t) = m(t) / (1 - beta1(t))
			mhat = m[i] / (1.0 - beta1**(t+1))
			# vhat(t) = v(t) / (1 - beta2(t))
			vhat = v[i] / (1.0 - beta2**(t+1))
			# x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
			x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
		# evaluate candidate point
		score = objective(x[0], x[1])
		# report progress
		print('>%d f(%s) = %.5f' % (t, x, score))
	return [x, score]

# seed the pseudo random number generator
seed(1)
# define range for input
bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
# define the total iterations
n_iter = 60
# steps size
alpha = 0.02
# factor for average gradient
beta1 = 0.8
# factor for average squared gradient
beta2 = 0.999
# perform the gradient descent search with adam
best, score = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
print('Done!')
print('f(%s) = %f' % (best, score))