
import LearningAutomata as LA
import numpy as np

def genData(period_size, variance, sequence_repeat_number):
    # define the atom of the repeating sequence
    S = np.arange(period_size)
    np.random.shuffle(S)
    # S = [0, 1, 6, 4, 3, 2, 5]
    print(S)
    # the output sequence
    T = []
    # fulfill the sequence
    for i in range(sequence_repeat_number):
        noise = np.random.uniform(-variance, variance, period_size)
        S_noise = np.add(S, noise)
        T.extend(S_noise)
    print(T)
    return S, T


def newLA(action, steep, defStep, maxStep, maxVariance, learningRate):
    return LA.LearningAutomata(action, steep, defStep, maxStep, maxVariance, learningRate)


def resetAllLA(groupLA):
    for LA in groupLA:
        LA.reset()


def checkFIttness(groupLA):
    # print('we are checking fittness')

    for LA in groupLA:

        if not LA.isStationary():
            # print('la is not stationary')
            return False
    return True


def LAReceives(groupLA, T):
    for i in range(len(T)):
        groupLA[i % len(groupLA)].learn(T[i])


def main():

    # LA config
    # action = -1
    steep = 0.5
    defStep = 1
    maxStep = 500
    maxVariance = 0.3
    learningRate = 0.01

    groupLA = []
    # environment config
    period_size = 7
    variance = 0.1
    sequence_repeat_number = 10

    # input sequence
    S, T = genData(period_size, variance, sequence_repeat_number)

    # sim started
    fit = False

    # print(len(groupLA))
    while not fit:
        # print('we are here')
        resetAllLA(groupLA)
        groupLA.append(newLA(T[len(groupLA)], steep, defStep, maxStep, maxVariance, learningRate))
        LAReceives(groupLA, T)
        fit = checkFIttness(groupLA)

    print('we installed', len(groupLA), 'to fit the data')

if __name__ == "__main__":
    main()
