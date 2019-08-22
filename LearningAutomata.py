import numpy


class LearningAutomata:
    ## action of LM
    action = 0
    variance = 0
    stationary = True
    ## confidence to choose the action
    # min: 1 (no confidence)
    # max: 3
    # friction to stay in one action
    #    confidence = 1
    #    maxConf = 2

    ## increment step

    ## decrement step
    # min: 1 max: 500

    maxStep = 200

    defStep = 10

    steep = 0.5

    maxVariance = 0.5

    previous = 0  # 0 : reward, 1: userPan, 2: comPan

    # def logistic function
    def step(self):
        a = (self.maxStep - self.defStep) / self.defStep
        return self.maxStep / (1 + a * numpy.exp(-self.steep * self.ctnPan))

    ## consecutive increament
    # ctnPan = 0

    ## Alternative
    alterLast = 0  # 0: no 1: yes
    ctnAlter = 0
    learningRate = 0.01
    ############################## Constructor ###############################
    def __init__(self, action, steep, defStep, maxStep, maxVariance, learningRate):
        self.action = action
        self.defStep = defStep
        self.maxStep = maxStep
        self.steep = steep
        self.maxVariance = maxVariance
        self.learningRate = learningRate
        self.stationary = True

    ##############################public methods##############################
    def learn(self, enviorResp):
        if not self.stationary:
            return
        exceedlower = self.action - self.variance >= enviorResp
        exceedupper = self.action + self.variance <= enviorResp
        if abs(enviorResp - self.action) > self.maxVariance:
            self.stationary = False
            return
        else:
            # learn from penalty
            # self.ctnPan += 1
            self.action = (self.action + enviorResp)/2
            self.variance = self.variance * (1+self.learningRate)
            self.learningRate *= 0.9

    def reset(self):
        self.stationary = True

    def getAction(self):
        return self.action

    def isStationary(self):
        return self.stationary

