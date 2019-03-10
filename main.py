#!/usr/bin/env python3
import os
import numpy as np
from numpy import random
import scipy
import matplotlib
import mnist
import pickle
matplotlib.use('agg')
from matplotlib import pyplot

mnist_data_directory = os.path.join(os.path.dirname(__file__), "data")

# TODO add any additional imports and global variables


def load_MNIST_dataset():
    PICKLE_FILE = os.path.join(mnist_data_directory, "MNIST.pickle")
    try:
        dataset = pickle.load(open(PICKLE_FILE, 'rb'))
    except:
        # load the MNIST dataset
        mnist_data = mnist.MNIST(mnist_data_directory, return_type="numpy", gz=True)
        Xs_tr, Lbls_tr = mnist_data.load_training();
        Xs_tr = Xs_tr.transpose() / 255.0
        Ys_tr = np.zeros((10, 60000))
        for i in range(60000):
            Ys_tr[Lbls_tr[i], i] = 1.0  # one-hot encode each label
        # shuffle the training data
        np.random.seed(8675309)
        perm = np.random.permutation(60000)
        Xs_tr = np.ascontiguousarray(Xs_tr[:,perm])
        Ys_tr = np.ascontiguousarray(Ys_tr[:,perm])
        Xs_te, Lbls_te = mnist_data.load_testing();
        Xs_te = Xs_te.transpose() / 255.0
        Ys_te = np.zeros((10, 10000))
        for i in range(10000):
            Ys_te[Lbls_te[i], i] = 1.0  # one-hot encode each label
        Xs_te = np.ascontiguousarray(Xs_te)
        Ys_te = np.ascontiguousarray(Ys_te)
        dataset = (Xs_tr, Ys_tr, Xs_te, Ys_te)
        pickle.dump(dataset, open(PICKLE_FILE, 'wb'))
    return dataset


# compute the gradient of the multinomial logistic regression objective, with regularization (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs        training examples (d * n)
# Ys        training labels   (c * n)
# ii        the list/vector of indexes of the training example to compute the gradient with respect to
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the average gradient of the regularized loss of the examples in vector ii with respect to the model parameters
def multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W):
    # TODO students should use their implementation from programming assignment 2
    Xs, Ys = Xs[:,ii], Ys[:,ii]
    ewx = np.exp(np.matmul(W,Xs))
    p = np.sum(ewx, axis=0)
    return 1 / Xs.shape[1] * np.matmul(-Ys+ 1/p * ewx, Xs.T) + gamma * W


# compute the error of the classifier (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# W         parameters        (c * d)
#
# returns   the model error as a percentage of incorrect labels
def multinomial_logreg_error(Xs, Ys, W):
    # TODO students should use their implementation from programming assignment 1
    return np.mean(np.argmax(np.matmul(W, Xs), axis=0) != np.argmax(Ys, axis=0))


# compute the cross-entropy loss of the classifier
#
# Xs        examples          (d * n)
# Ys        labels            (c * n)
# gamma     L2 regularization constant
# W         parameters        (c * d)
#
# returns   the model cross-entropy loss
def multinomial_logreg_loss(Xs, Ys, gamma, W):
    # TODO students should implement this
    ewx = np.exp(np.matmul(W, Xs))
    logSoft = np.log( ewx / np.sum(ewx, axis=0) )
    return - np.sum( Ys * logSoft) / Xs.shape[1] + gamma / 2 * np.sum(W**2)


# gradient descent (SAME AS PROGRAMMING ASSIGNMENT 1)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gradient_descent(Xs, Ys, gamma, W0, alpha, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 1
    models = []
    for i in range(1,num_epochs+1):
        W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, np.arange(Xs.shape[1]), gamma, W0)
        if i % monitor_period == 0:
            models.append(W0)
            print("epoch", i)
    return models


# gradient descent with nesterov momentum
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# num_epochs      number of epochs (passes through the training set, or equivalently iterations of gradient descent) to run
# monitor_period  how frequently, in terms of epochs/iterations to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" epochs
def gd_nesterov(Xs, Ys, gamma, W0, alpha, beta, num_epochs, monitor_period):
    # TODO students should implement this
    V0, models = W0, []
    for i in range(1, num_epochs+1):
        tmp = V0
        V0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, np.arange(Xs.shape[1]), gamma, W0)
        W0 = V0 + beta * (V0 - tmp)
        if i % monitor_period == 0:
            models.append(W0)
            print("epoch", i)
    return models


# SGD: run stochastic gradient descent with minibatching and sequential sampling order (SAME AS PROGRAMMING ASSIGNMENT 2)
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_minibatch_sequential_scan(Xs, Ys, gamma, W0, alpha, B, num_epochs, monitor_period):
    # TODO students should use their implementation from programming assignment 2
    n = Xs.shape[1]
    models = []
    for i in range(num_epochs):
        cur = i*(n//B)
        for j in range(n // B):
            ii = np.arange(j*B,(j+1)*B)
            W0 = W0 - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0) - alpha * gamma * W0
            if (j+cur+1) % monitor_period == 0:
                models.append(W0)
        print("epoch", i)
    return models


# SGD + Momentum: add momentum to the previous algorithm
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# beta            momentum hyperparameter
# B               minibatch size
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def sgd_mss_with_momentum(Xs, Ys, gamma, W0, alpha, beta, B, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    models = []
    V0 = 0 * np.zeros(W0.shape)
    for i in range(num_epochs):
        cur = i*(n//B)
        for j in range(n // B):
            ii = np.arange(j*B,(j+1)*B)
            V0 = beta * V0 - alpha * multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0)
            W0 = W0 + V0
            if (j+cur+1) % monitor_period == 0:
                models.append(W0)
        print("epoch", i)
    return models


# Adam Optimizer
#
# Xs              training examples (d * n)
# Ys              training labels   (c * n)
# gamma           L2 regularization constant
# W0              the initial value of the parameters (c * d)
# alpha           step size/learning rate
# rho1            first moment decay rate ρ1
# rho2            second moment decay rate ρ2
# B               minibatch size
# eps             small factor used to prevent division by zero in update step
# num_epochs      number of epochs (passes through the training set) to run
# monitor_period  how frequently, in terms of batches (not epochs) to output the parameter vector
#
# returns         a list of model parameters, one every "monitor_period" batches
def adam(Xs, Ys, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period):
    # TODO students should implement this
    n = Xs.shape[1]
    r, s = np.zeros(W0.shape), np.zeros(W0.shape)
    t = 0
    models = []
    for i in range(num_epochs):
        cur = i * (n // B)
        for j in range(n//B):
            t += 1
            ii = np.arange(j * B, (j + 1) * B)
            g = multinomial_logreg_grad_i(Xs, Ys, ii, gamma, W0)
            s = rho1 * s + (1-rho1) * g
            r = rho2 * r + (1-rho2) * g**2
            sprime, rprime = s / (1-rho1**t), r / (1-rho2**t)
            W0 = W0 - alpha * sprime / np.sqrt(rprime + eps)
            if (j+cur+1) % monitor_period == 0:
                models.append(W0)
        print("epoch", i)
    return models




if __name__ == "__main__":
    (Xs_tr, Ys_tr, Xs_te, Ys_te) = load_MNIST_dataset()
    # TODO add code to produce figures
    import timeit
    from argparse import ArgumentParser

    # Define the parser to run script on cmd
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--part1", action='store_true',
                        help="To run part 1 of the assignment")
    parser.add_argument("--part2", action='store_true',
                        help="To run part 2 of the assignment")
    parser.add_argument("--part3", action='store_true',
                        help="To run part 3 of the assignment")
    parser.add_argument("--accu", action='store_true',
                        help="To run accuracy section of the corresponding part of the assignment")
    parser.add_argument("--time", action='store_true',
                        help="To time corresponding part of the assignment")
    parser.add_argument("--explore", action='store_true',
                        help="To run the exploration part of the assignment")

    args = parser.parse_args()

    # Global constants
    c,_ = Ys_tr.shape
    d,_ = Xs_tr.shape
    W0 = np.zeros((c,d))
    beta1, beta2 = 0.9, 0.99
    gamma = 0.0001

    def get_error(Xs, Ys, models):
        return [multinomial_logreg_error(Xs, Ys, W) for W in models]

    def get_loss(Xs, Ys, models):
        return [multinomial_logreg_loss(Xs, Ys, gamma, W) for W in models]

    def get_model_errors(models, loss=False):
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        if loss:
            model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
            print("Loss for training set done")
            return [model_error_tr, model_error_te, model_training_loss]
        return [model_error_tr, model_error_te]


    # Other functions

    ''' Plot the model error for the models, whose respective names are given by [names].
            Save the image by the name [title].png and indlude [title] in the figure title '''
    def plot_error(t, model_error, names, title, measure='Error'):
        pyplot.figure(np.random.randint(1000))
        pyplot.xlabel('Epochs')
        pyplot.ylabel(measure)
        pyplot.title('MNIST ' + title + ' ' + measure)
        pyplot.grid(True)
        for name, error in zip(names, model_error):
            pyplot.plot(t, error, label=name)
        pyplot.gca().legend()
        pyplot.savefig(("plots/" + title + measure + '.png').replace(" ",""), bbox_inches='tight')


    ''' Plot the model runtimes [times] of the models whose respective names are given by [names].
            Save the image by the name [filename] '''
    def plot_runtimes(x_positions, times, names, filename='train_time.png'):
        # plot runtime for training as a bar graph
        pyplot.figure(3)
        pyplot.bar(x_positions, times, align='center', alpha=0.5)
        pyplot.xticks(x_positions, names)
        pyplot.ylabel('Average Runtime per Model (seconds)')
        pyplot.xlabel('Models')
        pyplot.title('Runtime of Model Training ')
        for i, v in enumerate(times):
            pyplot.text(i-.25, v * (1.015), " " + str(round(v,2)) +' s', color='black', va='center', fontweight='bold')
        pyplot.savefig("plots/" + filename, bbox_inches='tight')

    def plot_metrics(t, metrics, names, titles, measures):
        for metric, name, title, measure in zip(metrics, names, titles, measures):
            plot_error(t, metric, name, title, measure=measure)

    ''' Time the algorightms in [algos], whose respective names are given by [names],
        by averaging the runtime of the algorithm over 5 runs.
        PreC :The algorithms must be lambdas that take no inputs '''
    def time_algos(names, algos, runs=5):
        times = []
        for name, algo in zip(names,algos):
            time = 0
            for _ in range(runs):
                time -= timeit.default_timer()
                _ = algo()
                time += timeit.default_timer()
            times.append(time/runs)
            print(name,time/runs,"seconds")
        return times


if args.part1:
    print("Part 1")
    alpha, num_epochs = 1.0, 100
    monitor_period = 1

    gd = lambda _ : gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)

    momentum = lambda beta : gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta, num_epochs, monitor_period)

    gd_alpha = lambda alpha : gradient_descent(Xs_tr, Ys_tr, gamma, W0, alpha, num_epochs, monitor_period)

    momentum_alpha = lambda alpha, beta: gd_nesterov(Xs_tr, Ys_tr, gamma, W0, alpha, beta, num_epochs, monitor_period)

    if args.accu:
        algos = [gd, momentum, momentum]
        params = [0, beta1, beta2]
        names = ["Gradient Descent",
                 "Nesterov's Momentum beta="+str(round(beta1,3)),
                 "Nesterov's Momentum beta="+str(round(beta2,3))
                 ]
        models = []

        for algo, param, name in zip(algos, params, names):
            models.append(algo(param))
            print(name, "done")

        # Get model errors
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
        print("Loss for training set done")
        # Get the range of times
        t = np.arange(1, num_epochs + 1)
        # plot training error as a function of epochs
        plot_error(t, model_error_tr, names, "Gradient Descent and Nesterov's Momentum Training")
        # plot test error as a function of epochs
        plot_error(t, model_error_te, names, "Gradient Descent and Nesterov's Momentum Test")
        # plot training loss as a function of epochs
        plot_error(t, model_training_loss, names, "Gradient Descent and Nesterov's Momentum Training", measure='Loss')

    if args.time:
        algos = [lambda : gd(0), lambda : momentum(beta1)]
        names = ["Gradient Descent", "Nesterov's Momentum"]

        # Make plots for the average runtimes
        times = time_algos(names, algos)
        x_positions = np.arange(len(names))

        # plot runtime for training as a bar graph
        plot_runtimes(x_positions, times, names, filename='train_time_part1.png')

    if args.explore:
        alpha1, alpha2 = 0.8, 0.7
        beta1, beta2 = .85, .95

        algos = [lambda : gd_alpha(alpha1),
                 lambda : gd_alpha(alpha2),
                 lambda : momentum_alpha(alpha1, beta1),
                 lambda : momentum_alpha(alpha2, beta2)
                 ]

        names = [
            "GD alpha=" + str(round(alpha1,3)),
            "GD alpha=" + str(round(alpha2, 3)),
            "Nesterov's alpha="+str(round(alpha1,3))+" beta="+str(round(beta1,3)),
            "Nesterov's alpha="+str(round(alpha2,3))+" beta="+str(round(beta2,3)),
            ]
        models = []

        for algo, name in zip(algos, names):
            models.append(algo())
            print(name, "done")


        # Get model errors
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
        print("Loss for training set done")
        # Get the range of times
        t = np.arange(1, num_epochs + 1)
        # plot training error as a function of epochs
        plot_error(t, model_error_tr, names, "GD and Nesterov's Momentum Training Exploration")
        # plot test error as a function of epochs
        plot_error(t, model_error_te, names, "GD and Nesterov's Momentum Test Exploration")
        # plot training loss as a function of epochs
        plot_error(t, model_training_loss, names, "GD and Nesterov's Momentum Training Exploration", measure='Loss')




if args.part2:
    print("Part 2")
    alpha, B = 0.2, 600
    num_epochs, monitor_period = 10, 10

    sgd_mini = lambda _ : sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    sgd_momentum = lambda beta : sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs, monitor_period)
    sgd_mini_alpha = lambda alpha : sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    sgd_momentum_alpha = lambda alpha, beta : sgd_mss_with_momentum(Xs_tr, Ys_tr, gamma, W0, alpha, beta, B, num_epochs, monitor_period)

    if args.accu:
        algos = [sgd_mini, sgd_momentum, sgd_momentum]
        params = [0, beta1, beta2]
        names = ["Sequential Minibatch SGD",
                 "Minibatch SGD + Momentum beta=" + str(round(beta1, 3)),
                 "Minibatch SGD + Momentum beta=" + str(round(beta2, 3))
                 ]
        models = []

        for algo, param, name in zip(algos, params, names):
            models.append(algo(param))
            print(name, "done")

        # Get model errors
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
        print("Loss for training set done")

        # Get the range of times
        t = np.arange(.1, num_epochs + .1, .1)
        # plot training error as a function of epochs
        plot_error(t, model_error_tr, names, "SGD and Momentum Training")
        # plot test error as a function of epochs
        plot_error(t, model_error_te, names, "SGD and Momentum Test")
        # plot training loss as a function of epochs
        plot_error(t, model_training_loss, names, "SGD and Momentum Training", measure='Loss')

    if args.time:
        algos = [lambda : sgd_mini(0), lambda : sgd_momentum(beta1)]
        names = ["Minibatch SGD", "Minibatch SGD + Momentum"]

        # Make plots for the average runtimes
        times = time_algos(names, algos)
        x_positions = np.arange(len(names))

        # plot runtime for training as a bar graph
        plot_runtimes(x_positions, times, names, filename='train_time_part2.png')

    if args.explore:
        alpha1, alpha2, B = 0.25, 0.3, 600
        beta1, beta2 = 0.85, 0.95
        num_epochs, monitor_period = 10, 10


        algos = [lambda : sgd_mini_alpha(alpha1),
                 lambda : sgd_mini_alpha(alpha2),
                 lambda : sgd_momentum_alpha(alpha1, beta1),
                 lambda : sgd_momentum_alpha(alpha2, beta2)
                 ]
        names = ["Sequential Minibatch SGD alpha=" + str(round(alpha1, 3)),
                 "Sequential Minibatch SGD alpha=" + str(round(alpha2, 3)),
                 "Minibatch SGD + Momentum alpha=" + str(round(alpha1, 3)) + " beta=" + str(round(beta1, 3)),
                 "Minibatch SGD + Momentum alpha=" + str(round(alpha2, 3)) + " beta=" + str(round(beta2, 3)),
                 ]
        models = []

        for algo, name in zip(algos, names):
            models.append(algo())
            print(name, "done")

        # Get model errors
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
        print("Loss for training set done")

        # Get the range of times
        t = np.arange(.1, num_epochs + .1, .1)
        # plot training error as a function of epochs
        plot_error(t, model_error_tr, names, "SGD and Momentum Training Exploration")
        # plot test error as a function of epochs
        plot_error(t, model_error_te, names, "SGD and Momentum Test Exploration")
        # plot training loss as a function of epochs
        plot_error(t, model_training_loss, names, "SGD and Momentum Training Exploration", measure='Loss')



if args.part3:
    print("Part 3")
    alpha1, alpha2 = 0.2, 0.01
    B, monitor_period = 600, 10
    rho1, rho2 = 0.9, 0.999
    eps, num_epochs = 0.00001, 10

    sgd_mini = lambda : sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha1, B, num_epochs, monitor_period)
    sgd_mini_alpha = lambda alpha : sgd_minibatch_sequential_scan(Xs_tr, Ys_tr, gamma, W0, alpha, B, num_epochs, monitor_period)
    adam_l = lambda : adam(Xs_tr, Ys_tr, gamma, W0, alpha2, rho1, rho2, B, eps, num_epochs, monitor_period)
    adam_l_alpha = lambda alpha, rho1, rho2 : adam(Xs_tr, Ys_tr, gamma, W0, alpha, rho1, rho2, B, eps, num_epochs, monitor_period)

    if args.accu:
        algos = [sgd_mini, adam_l]
        names = ["Sequential Minibatch SGD",
                 "Adam rho1=" + str(round(rho1,3)) + ' rho2=' + str(round(rho2,3))]
        models = []

        for algo, name in zip(algos, names):
            models.append(algo())
            print(name, "done")

        # Get model errors
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
        print("Loss for training set done")

        # Get the range of times
        t = np.arange(.1, num_epochs+.1, .1)
        # plot training error as a function of epochs
        plot_error(t, model_error_tr, names, "SGD and ADAM Training")
        # plot test error as a function of epochs
        plot_error(t, model_error_te, names, "SGD and ADAM Test")
        # plot training loss as a function of epochs
        plot_error(t, model_training_loss, names, "SGD and ADAM Training", measure='Loss')

    if args.time:
        algos = [sgd_mini, adam_l]
        names = ["Minibatch SGD", "ADAM"]

        # Make plots for the average runtimes
        times = time_algos(names, algos, runs=10)
        x_positions = np.arange(len(names))

        # plot runtime for training as a bar graph
        plot_runtimes(x_positions, times, names, filename='train_time_part3.png')

    if args.explore:
        num_epochs = 10
        alpha1, alpha2 = 0.01, 0.02
        rho1, rho2 = 0.8, 0.88
        rho1p, rho2p = 0.85, 0.93
        alpha_sgd1, alpha_sgd2 = 0.25, 0.3

        algos = [lambda : sgd_mini_alpha(alpha_sgd1),
                 lambda : sgd_mini_alpha(alpha_sgd2),
                 lambda : adam_l_alpha(alpha1, rho1, rho2),
                 lambda: adam_l_alpha(alpha1, rho1p, rho2p),
                 lambda: adam_l_alpha(alpha2, rho1, rho2),
                 lambda: adam_l_alpha(alpha2, rho1p, rho2p)
                 ]
        names = ["Sequential Minibatch SGD alpha=" + str(round(alpha_sgd1,3)),
                 "Sequential Minibatch SGD alpha=" + str(round(alpha_sgd2, 3)),
                 "Adam alpha=" +str(round(alpha1, 3)) + " rho1=" + str(round(rho1, 3)) + ' rho2=' + str(round(rho2, 3)),
                 "Adam alpha=" + str(round(alpha1, 3)) + " rho1=" + str(round(rho1p, 3)) + ' rho2=' + str(round(rho2p, 3)),
                 "Adam alpha=" + str(round(alpha2, 3)) + " rho1=" + str(round(rho1, 3)) + ' rho2=' + str(round(rho2, 3)),
                 "Adam alpha=" + str(round(alpha2, 3)) + " rho1=" + str(round(rho1p, 3)) + ' rho2=' + str(round(rho2p, 3))
                 ]
        models = []

        for algo, name in zip(algos, names):
            models.append(algo())
            print(name, "done")

        # Get model errors
        model_error_tr = [get_error(Xs_tr, Ys_tr, model) for model in models]
        print("Errors for training set done")
        model_error_te = [get_error(Xs_te, Ys_te, model) for model in models]
        print("Errors for test set done")
        model_training_loss = [get_loss(Xs_te, Ys_te, model) for model in models]
        print("Loss for training set done")

        # Get the range of times
        t = np.arange(.1, num_epochs + .1, .1)
        # plot training error as a function of epochs
        plot_error(t, model_error_tr, names, "SGD and ADAM Training Exploration")
        # plot test error as a function of epochs
        plot_error(t, model_error_te, names, "SGD and ADAM Test Exploration")
        # plot training loss as a function of epochs
        plot_error(t, model_training_loss, names, "SGD and ADAM Training Exploration", measure='Loss')