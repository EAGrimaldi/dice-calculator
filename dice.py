import numpy as np
import matplotlib.pyplot as pp
import re
import warnings


class RunInPlace:
    """Helper class for making list comprehensions of running sums in O(n) time"""
    def __init__(self, value: float = 0):
        self.value = value
    def running_sum(self, new_value: float) -> float:
        self.value += new_value
        return self.value

class AnyDie:
    expression = ""
    values = np.array([])
    probabilities = np.array([])
    probabilities_at_most = np.array([])
    probabilities_at_least = np.array([])
    def __init__(self, values: np.ndarray, probabilities: np.ndarray, expression: str) -> None:
        self.expression = expression
        self.values = values
        self.probabilities = probabilities
        self.__init_other_probs__()
    def __init_other_probs__(self) -> None:
        """Calculates probabilities_at_most as an integral of probabilities, and probabilities_at_least as a compliment of probabilities_at_most."""
        tempSum = RunInPlace()
        self.probabilities_at_most = np.array([tempSum.running_sum(probability) for probability in self.probabilities])
        self.probabilities_at_least = np.ones(self.probabilities_at_most.shape) - self.probabilities_at_most + self.probabilities_at_most[0]
    def apply_modifier(self, modifier: int, operator: str='+') -> None:
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
            "atmost": self.probabilities_at_most,
            "atleast": self.probabilities_at_least
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
        conv_dict = {1: np.ones(self.size)/self.size}
        conv_highway = [1] if self.number%2 else []
        bit = 2
        while bit <= self.number:
            conv_dict[bit] = np.convolve(conv_dict[bit/2], conv_dict[bit/2])
            if self.number & bit:
                conv_highway.append(bit)
            bit *= 2
        result = conv_dict[conv_highway[0]]
        if len(conv_highway) > 1:
            for i in conv_highway[1:]:
                result = np.convolve(result, conv_dict[i])
        return result

class AdvantagePool(AnyDie):
    def __init__(self, size: int, total_number: int=2, keep_number: int=1, mode: str='h') -> None:
        if size < 2:
            raise ValueError("The size of the dice must be greater than 1.")
        if total_number < 2:
            raise ValueError("The total number of dice in the pool must be greater than 1.")
        if keep_number > total_number:
            raise ValueError("The number of dice kept must be less than the total number of dice in the pool.")
        if keep_number == total_number:
            raise ValueError("If the number of dice kept is the same as the total number of dice in the pool, then you should use the simple DicePool object.")
        regex = re.compile("[^a-zA-Z]")
        mode = regex.sub("", mode).lower()
        mode = mode[4] if mode[0:4] == 'keep' else mode[1] if mode[0] == 'k' else mode[0]
        if mode not in {'h', 'l'}:
            raise KeyError("Invalid mode. Valid modes include 'h' (keep highest) and 'l' (keep lowest).")
        self.size = size
        self.total_number = total_number
        self.keep_number = keep_number
        self.mode = mode
        self.expression = f"{self.total_number}d{self.size}k{self.mode}{self.keep_number}"
        self.values = np.array([i for i in range(self.keep_number, self.keep_number*self.size+1)])
        self.probabilities = self.__advantage_formula__() if self.keep_number == 1 else self.__advantage_statistics__()
        self.__init_other_probs__()
    def __advantage_formula__(self) -> np.ndarray:
        """ Helper function for building the probability mass function of an advantage or disadvantage dice pool (NdSkh1 or NdSkl1).

            As it turns out there is an algebraic solution:

            For a roll of N dice of size S where we keep only the highest result of the pool, the probability of rolling some value X is:

                P(X) = ( X^N - (X-1)^N ) / ( S^N )

            Neat!
        """
        if self.mode == 'h':
            result = np.array([ (i**self.total_number - (i-1)**self.total_number) / (self.size**self.total_number) for i in range(1,self.size+1) ])
        else:
            result = np.array([ (i**self.total_number - (i-1)**self.total_number) / (self.size**self.total_number) for i in range(self.size,0,-1) ])
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

def is_real_die(size: int) -> bool:
    real_dice_set = {2, 4, 6, 8, 10, 12, 20, 100}
    if size in real_dice_set:
        return True
    else:
        warnings.warn(f"{size} is an impractical size for real dice. \nIf you are designing for tabletop play, you should probably choose from {real_dice_set}")
        return False

def compose_dice(die0: AnyDie, die1: AnyDie, operator: str='+') -> AnyDie:
    """ Composes two arbitrary dice expressions by addition or subtraction into one meta die.
        Valid operators include '+' and '-'.

        Currently does not simplify expression or intelligently choose output type.
    """
    probabilities = np.convolve(die0.probabilities, die1.probabilities)
    if operator == '+':
        values = np.array([i for i in range(die0.values[0] + die1.values[0], die0.values[-1] + die1.values[-1] + 1)])
    elif operator == '-':
        values = np.array([i for i in range(die0.values[0] - die1.values[-1], die0.values[-1] - die1.values[0] + 1)])
    else:
        raise Exception("Invalid operation. Valid operations include '+' and '-'.")
    expression = die0.expression + operator + die1.expression
    return AnyDie(values, probabilities, expression)


if __name__ == "__main__":
    test0 = AdvantagePool(size=20, total_number=2, mode='h')
    test0.graph(mode="normal")
    test1 = AdvantagePool(size=20, total_number=2, mode='l')
    test1.graph(mode="normal")
    test2 = AdvantagePool(size=6, total_number=4, mode='h')
    test2.graph(mode="normal")
    test3 = AdvantagePool(size=6, total_number=4, mode='l')
    test3.graph(mode="normal")
