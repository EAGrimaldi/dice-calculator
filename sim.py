import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as pp

def rolld(num_sides: int) -> int:
    return np.random.randint(1 , num_sides+1)

def sim_nova_turn(target_armor: int = 18, reckless: bool = True) -> int:
    """ simulate total damage in a single turn for GWM dream team in Baldur's Gate 3

        (GWM + Action Surge + Reckless Attack + Haste + Improved Bless) x 2
    """
    damage_running_total = 0
    for fighter_number in range(2):
        GWM_crit_flag = False
        for attack_number in range(7):
            hit_roll = max(rolld(20), rolld(20)) if reckless else rolld(20)
            bless_roll = rolld(4) + rolld(4)
            hit_total = hit_roll + bless_roll + 5
            damage_roll = rolld(6) + rolld(6)
            damage_total = ( ( 2*damage_roll + 15 if hit_roll==20 else damage_roll + 15 ) if hit_total > target_armor else 0 ) if attack_number < 6 or GWM_crit_flag else 0
            #print(f"fighter {fighter_number}, attack {attack_number}: {damage_total}")
            damage_running_total += damage_total
            if hit_roll == 20:
                GWM_crit_flag = True
    return damage_running_total

def sim_nova_turn_average(target_armor: int = 18, reckless: bool = True, num_experiments: int = 100000, log: bool = False) -> float:
    data = np.array([ sim_nova_turn(target_armor, reckless) for i in tqdm(range(num_experiments)) ], dtype=int)
    average, sigma = np.average(data), np.std(data)
    if log:
        print(f"average {average}, sigma {sigma}")
    return average

def sim_nova_turn_range(target_armor_lower: int = 15, target_armor_upper: int = 25, num_experiments: int = 100000, graph: bool = True) -> np.ndarray:
    data = np.array([ sim_nova_turn_average(target_armor, num_experiments) for target_armor in tqdm(range(target_armor_lower, target_armor_upper+1)) ], dtype=float)
    if graph:
        pp.plot(range(target_armor_lower, target_armor_upper+1), data)
        pp.show()

if __name__ == "__main__":
    sim_nova_turn_range()