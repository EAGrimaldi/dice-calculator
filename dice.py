import numpy as np
import matplotlib.pyplot as pp
import re


class RunInPlace:
    """Helper class for making list comprehensions of running sums in O(n) time"""
    def __init__(self, value: float = 0):
        self.value = value
    def runningSum(self, newValue: float) -> float:
        self.value += newValue
        return self.value

class Die:
    type = ""
    expression = ""
    values = np.array([])
    probabilities = np.array([])
    probabilitiesAtMost = np.array([])
    probabilitiesAtLeast = np.array([])
    def __init__(self, vals: np.ndarray, probs: np.ndarray, ex: str) -> None:
        self.expression = ex
        self.values = vals
        self.probabilities = probs
        self.__init_other_probs__()
    def __init_other_probs__(self):
        """Calculates probabilitiesAtMost as an integral of probabilities, and probabilitiesAtLeast as a compliment of probabilitiesAtMost."""
        tempSum = RunInPlace()
        self.probabilitiesAtMost = np.array([tempSum.runningSum(probability) for probability in self.probabilities])
        self.probabilitiesAtLeast = np.ones(self.probabilitiesAtMost.shape) - self.probabilitiesAtMost
    def applyModifier(self, mod: int, op: str='+') -> None:
        """ Applies a flat modifier by addition or subtraction to a die expression.
            Valid operators include '+' and '-'.

            Currently does not simplify expression.
            """
        self.expression = self.expression + op + f"{mod}"
        if op == '+':
            self.values = self.values + np.ones(self.values.shape)*mod
        elif op == '-':
            self.values = self.values - np.ones(self.values.shape)*mod
        else:
            raise Exception("Invalid operation. Valid operations include '+' and '-'.")
    def graph(self, mode: str = "normal") -> None:
        regex = re.compile("[^a-zA-Z]")
        mode = regex.sub("", mode).lower()
        mode2data = {
            "normal": self.probabilities,
            "atmost": self.probabilitiesAtLeast,
            "atleast": self.probabilitiesAtMost
        }
        if mode not in mode2data:
            raise KeyError("Invalid mode. Valid modes include 'normal', 'at most', 'at least'.")
        data = mode2data[mode]
        pp.bar(self.values, data)
        pp.xlabel("Possible Values")
        pp.ylabel("Probability")
        pp.title(f"{self.expression}, mode='{mode}'")
        pp.show()

class SingleDie(Die):
    def __init__(self, s: int = 6) -> None:
        if s < 2:
            raise ValueError("The size of the die must be greater than 1.")
        self.size = s
        self.expression = f"1d{self.size}"
        self.values = np.array([i for i in range(1,self.size+1)])
        self.probabilities = np.ones(self.size)/self.size
        self.__init_other_probs__()

class DicePool(Die):
    def __init__(self, s: int, n: int) -> None:
        if s < 2:
            raise ValueError("The size of the dice must be greater than 1.")
        if n < 2:
            raise ValueError("The number of dice in the dice pool must be greater than 1.")
        self.size = s
        self.number = n
        self.expression = f"{self.number}d{self.size}"
        self.values = np.array([i for i in range(self.number, self.number*self.size+1)])
        self.probabilities = self.__pool_convolution__()
        self.__init_other_probs__()
    def __pool_convolution__(self):
        """ Helper function for building the probability mass function of a dice pool (NdS).
        
            # TODO Try to speed this up further with a general form from the calculus of convolutions?
        """
        convDict = {1: np.ones(self.size)/self.size}
        convHighway = [1] if self.number%2 else []
        bit = 2
        while bit <= self.number:
            convDict[bit] = np.convolve(convDict[bit/2], convDict[bit/2])
            if self.number & bit:
                convHighway.append(bit)
            bit *= 2
        if len(convHighway) == 1:
            return convDict[convHighway[0]]
        result = np.convolve(convDict[convHighway[0]], convDict[convHighway[1]])
        if len(convHighway) > 2:
            for i in convHighway[2:]:
                result = np.convolve(result, convDict[i])
        return result

def composeDice(dieA: Die, dieB: Die, op: str='+') -> Die:
    """ Composes two arbitrary die expressions by addition or subtraction into one meta die.
        Valid operators include '+' and '-'.

        Currently does not simplify expression or intelligently choose output type.
    """
    probs = np.convolve(dieA.probabilities, dieB.probabilities)
    if op == '+':
        vals = np.array([i for i in range(dieA.values[0] + dieB.values[0], dieA.values[-1] + dieB.values[-1] + 1)])
    elif op == '-':
        vals = np.array([i for i in range(dieA.values[0] - dieB.values[-1], dieA.values[-1] - dieB.values[0] + 1)])
    else:
        raise Exception("Invalid operation. Valid operations include '+' and '-'.")
    ex = dieA.expression + op + dieB.expression
    return Die(vals, probs, ex)


if __name__ == "__main__":
    test = DicePool(8, 100)
    test.graph()
