import numpy as np
import matplotlib.pyplot as pp
import re
import warnings


class RunInPlace:
    """Helper class for making list comprehensions of running sums in O(n) time"""
    def __init__(self, value: float = 0):
        self.value = value
    def runningSum(self, newValue: float) -> float:
        self.value += newValue
        return self.value

class AnyDie:
    expression = ""
    values = np.array([])
    probabilities = np.array([])
    probabilitiesAtMost = np.array([])
    probabilitiesAtLeast = np.array([])
    def __init__(self, values: np.ndarray, probabilities: np.ndarray, expression: str) -> None:
        self.expression = expression
        self.values = values
        self.probabilities = probabilities
        self.__init_other_probs__()
    def __init_other_probs__(self) -> None:
        """Calculates probabilitiesAtMost as an integral of probabilities, and probabilitiesAtLeast as a compliment of probabilitiesAtMost."""
        tempSum = RunInPlace()
        self.probabilitiesAtMost = np.array([tempSum.runningSum(probability) for probability in self.probabilities])
        self.probabilitiesAtLeast = np.ones(self.probabilitiesAtMost.shape) - self.probabilitiesAtMost + self.probabilitiesAtMost[0]
    def applyModifier(self, modifier: int, operator: str='+') -> None:
        """ Applies a flat modifier by addition or subtraction to a die expression.
            Valid operators include '+' and '-'.

            Currently does not simplify expression.
            """
        self.expression = self.expression + operator + f"{modifier}"
        if operator == '+':
            self.values = self.values + np.ones(self.values.shape)*modifier
        elif operator == '-':
            self.values = self.values - np.ones(self.values.shape)*modifier
        else:
            raise Exception("Invalid operation. Valid operations include '+' and '-'.")
    def graph(self, mode: str = "normal") -> None:
        regex = re.compile("[^a-zA-Z]")
        mode = regex.sub("", mode).lower()
        mode2data = {
            "normal": self.probabilities,
            "atmost": self.probabilitiesAtMost,
            "atleast": self.probabilitiesAtLeast
        }
        if mode not in mode2data:
            raise KeyError("Invalid mode. Valid modes include 'normal', 'at most', 'at least'.")
        pp.bar(self.values, mode2data[mode])
        pp.xlabel("Possible Values")
        pp.ylabel("Probability")
        pp.title(f"{self.expression}, mode='{mode}'")
        pp.show()

class SingleDie(AnyDie):
    def __init__(self, size: int = 6) -> None:
        if size < 2:
            raise ValueError("The size of the die must be greater than 1.")
        self.size = size
        self.expression = f"1d{self.size}"
        self.values = np.array([i for i in range(1,self.size+1)])
        self.probabilities = np.ones(self.size)/self.size
        self.__init_other_probs__()

class DicePool(AnyDie):
    def __init__(self, size: int, number: int) -> None:
        if size < 2:
            raise ValueError("The size of the dice must be greater than 1.")
        if number < 2:
            raise ValueError("The number of dice in the pool must be greater than 1.")
        self.size = size
        self.number = number
        self.expression = f"{self.number}d{self.size}"
        self.values = np.array([i for i in range(self.number, self.number*self.size+1)])
        self.probabilities = self.__pool_convolution__()
        self.__init_other_probs__()
    def __pool_convolution__(self) -> np.ndarray:
        """ Helper function for building the probability mass function of a dice pool (NdS).

            Leverages the symmetry of dice pools and the associativity of convolutions to speed up the calculation.

            ---

            In order to calculate the PMF of NdS, we start with the simple case of 1dS.
            We know that the convolution of the PMF of 1dS with itself gives the PMF of 2dS.
            Further, the convolution of 2dS with itself gives 4ds, and so on for all other powers of 2.
            In the case that N is not a power of 2, you can construct them out of power-of-2-cases encoded by the binary representation of N.
            For instance PMF_10d6 = conv(PMF_8d6, PMF_2d6).
            
            We use this relationship to minimize the number of convolutions used in our calculation.
            Successive convolutions can quickly blow up the number of matrix operations in your calculation.
        """
        convDict = {1: np.ones(self.size)/self.size}
        convHighway = [1] if self.number%2 else []
        bit = 2
        while bit <= self.number:
            convDict[bit] = np.convolve(convDict[bit/2], convDict[bit/2])
            if self.number & bit:
                convHighway.append(bit)
            bit *= 2
        result = convDict[convHighway[0]]
        if len(convHighway) > 1:
            for i in convHighway[1:]:
                result = np.convolve(result, convDict[i])
        return result

class AdvantagePool(AnyDie):
    def __init__(self, size: int, totalNumber: int=2, keepNumber: int=1, mode: str='h') -> None:
        if size < 2:
            raise ValueError("The size of the dice must be greater than 1.")
        if totalNumber < 2:
            raise ValueError("The total number of dice in the pool must be greater than 1.")
        if keepNumber > totalNumber:
            raise ValueError("The number of dice kept must be less than the total number of dice in the pool.")
        if keepNumber == totalNumber:
            raise ValueError("If the number of dice kept is the same as the total number of dice in the pool, then you should use the simple DicePool object.")
        regex = re.compile("[^a-zA-Z]")
        mode = regex.sub("", mode).lower()
        mode = mode[4] if mode[0:4] == 'keep' else mode[1] if mode[0] == 'k' else mode[0]
        if mode not in {'h', 'l'}:
            raise KeyError("Invalid mode. Valid modes include 'h' (keep highest) and 'l' (keep lowest).")
        self.size = size
        self.totalNumber = totalNumber
        self.keepNumber = keepNumber
        self.mode = mode
        self.expression = f"{self.totalNumber}d{self.size}k{self.mode}{self.keepNumber}"
        self.values = np.array([i for i in range(self.keepNumber, self.keepNumber*self.size+1)])
        self.probabilities = self.__advantage_formula__() if self.keepNumber == 1 else self.__advantage_statistics__()
        self.__init_other_probs__()
    def __advantage_formula__(self) -> np.ndarray:
        """ Helper function for building the probability mass function of an advantage or disadvantage dice pool (NdSkh1 or NdSkl1).

            As it turns out there is an algebraic solution:

            For a roll of N dice of size S where we keep only the highest result of the pool, the probability of rolling some value X is:

                P(X) = ( X^N - (X-1)^N ) / ( S^N )

            Neat!
        """
        if self.mode == 'h':
            result = np.array([ (i**self.totalNumber - (i-1)**self.totalNumber) / (self.size**self.totalNumber) for i in range(1,self.size+1) ])
        else:
            result = np.array([ (i**self.totalNumber - (i-1)**self.totalNumber) / (self.size**self.totalNumber) for i in range(self.size,0,-1) ])
        return result
    def __advantage_statistics__(self) -> np.ndarray:
        """ Helper function for building the probability mass function of rolling N dice of size S and keeping and summing the highest or lowest K results.

            We use simple statistics. We have not yet found the formula for this process. If you work it out, let us know!
        """
        warnings.warn(f"These results were obtained using statistical methods and thus may include small error.")
        result = np.zeros(len(self.values))
        for _ in range(10**3):
            print("woops!") # TODO please implement this bit lol
        return result

def isRealDie(size: int) -> bool:
    realDiceSet = {2, 4, 6, 8, 10, 12, 20, 100}
    if size in realDiceSet:
        return True
    else:
        warnings.warn(f"{size} is an impractical size for real dice. \nIf you are designing for tabletop play, you should probably choose from {realDiceSet}")
        return False

def composeDice(dieA: AnyDie, dieB: AnyDie, operator: str='+') -> AnyDie:
    """ Composes two arbitrary dice expressions by addition or subtraction into one meta die.
        Valid operators include '+' and '-'.

        Currently does not simplify expression or intelligently choose output type.
    """
    probabilities = np.convolve(dieA.probabilities, dieB.probabilities)
    if operator == '+':
        values = np.array([i for i in range(dieA.values[0] + dieB.values[0], dieA.values[-1] + dieB.values[-1] + 1)])
    elif operator == '-':
        values = np.array([i for i in range(dieA.values[0] - dieB.values[-1], dieA.values[-1] - dieB.values[0] + 1)])
    else:
        raise Exception("Invalid operation. Valid operations include '+' and '-'.")
    expression = dieA.expression + operator + dieB.expression
    return AnyDie(values, probabilities, expression)


if __name__ == "__main__":
    test0 = AdvantagePool(size=20, totalNumber=2, mode='h')
    test0.graph(mode="normal")
    test1 = AdvantagePool(size=20, totalNumber=2, mode='l')
    test1.graph(mode="normal")
    test2 = AdvantagePool(size=6, totalNumber=4, mode='h')
    test2.graph(mode="normal")
    test3 = AdvantagePool(size=6, totalNumber=4, mode='l')
    test3.graph(mode="normal")
