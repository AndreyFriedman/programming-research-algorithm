import copy

import doctest
import logging
import unittest
import pandas as pd
import random
import numpy as np
from itertools import compress
import math
from flask_example import app

InitialTemp = 10000
LastTemp = 0.9
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)


class TestStringMethods(unittest.TestCase):

    def test_alot_jobs(self):
        natun = [
            ['', 'M1', 'M2', 'M3'],
            ['J1', 12, 15, 12],
            ['J2', 15, 54, 32],
            ['J3', 15, 1, 1],
            ['J4', 15, 15, 1],
            ['J5', 15, 1, 1],
            ['J6', 15, 13, 12],
            ['J7', 15, 12, 11],
            ['J8', 7, 154, 1],
            ['J9', 7, 9, 1],
            ['J10', 8, 4, 78],
            ['J11', 3, 87, 45],
            ['J12', 45, 78, 2],
            ['J13', 4, 1, 11],
            ['J14', 54, 78, 1],
            ['J15', 15, 45, 2],
            ['J16', 87, 1, 121],
            ['J17', 54, 1, 32],
            ['J18', 43, 1, 41],
            ['J19', 15, 3, 871],
            ['J20', 8, 31, 8],
            ['J21', 7, 45, 34],
            ['J22', 12, 1, 4],
            ['J23', 54, 45, 3],
            ['J24', 453, 41, 4],
            ['J25', 15, 81, 35],
            ['J26', 4, 91, 45],
            ['J27', 78, 9, 45],
            ['J28', 185, 9, 8],
            ['J29', 15, 3, 78],
            ['J30', 15, 1, 1]]
        answer = unrelated_parallel_machine_scheduling(natun,1)

        self.assertEqual(answer,
                         {'J1': 'M1', 'J10': 'M3', 'J11': 'M1', 'J12': 'M3', 'J13': 'M1', 'J14': 'M3', 'J15': 'M3',
                          'J16': 'M1', 'J17': 'M2', 'J18': 'M3', 'J19': 'M2', 'J2': 'M1', 'J20': 'M3', 'J21': 'M3',
                          'J22': 'M2', 'J23': 'M3', 'J24': 'M2', 'J25': 'M3', 'J26': 'M2', 'J27': 'M1', 'J28': 'M1',
                          'J29': 'M2', 'J3': 'M1', 'J30': 'M3', 'J4': 'M2', 'J5': 'M1', 'J6': 'M3', 'J7': 'M3',
                          'J8': 'M2', 'J9': 'M2'})

    def test_alot_machines(self):
        natun = [
            ['', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11', 'M12', 'M13', 'M14', 'M15', 'M16',
             'M17', 'M18', 'M19', 'M20', 'M21', 'M22', 'M23', 'M24', 'M25', 'M26', 'M27', 'M28', 'M29', 'M30'],
            ['J1', 12, 15, 12, 54, 12, 5, 1, 13, 21, 54, 12, 45, 15, 22, 48, 6, 5, 12, 15, 12, 51, 4, 1, 4, 45, 21, 321,
             56, 32, 65],
            ['J2', 15, 54, 32, 65, 15, 1, 2, 21, 1, 4, 13, 32, 1, 12, 5, 12, 15, 26, 32, 44, 15, 1, 231, 21, 5, 21, 21,
             12, 1, 1],
            ['J3', 15, 1, 1, 5, 52, 56, 13, 51, 12, 5, 5, 52, 45, 21, 22, 54, 54, 12, 1, 456, 45, 54, 21, 12, 45, 15, 1,
             1, 12, 24]]
        answer = unrelated_parallel_machine_scheduling(natun,1)
        self.assertEqual(answer, {'J1': 'M7', 'J2': 'M30', 'J3': 'M27'})

def initialization(matrix):
    '''
    given matrix it returns a matrix with random jobs given to random machines
    if a job is given to machine there will be + sign in the returned matrix, else there will be - sign

    mat = [["", "M1", "M2", "M3", "M4"], ["J1", 14, 6, 8, 20], ["J2", 7, 4, 11, 20], ["J3", 3, 21, 5, 20], ["J4", 1, 1, 5, 20]]
    >>> initialization([["", "M1", "M2", "M3", "M4"], ["J1", 14, 6, 8, 20], ["J2", 7, 4, 11, 20], ["J3", 3, 21, 5, 20], ["J4", 1, 1, 5, 20]])
    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '+', '-', '-', '-'], ['J2', '+', '-', '-', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '+', '-']]

    mat = [["", "M1"], ["J1", 14]]
    >>> initialization([["", "M1"], ["J1", 14]])
    [['', 'M1'], ['J1', '+']]


    mat = [["", "M1", "M2", "M3"], ["J1", 14, 6, 8]]
    >>> initialization([["", "M1", "M2", "M3"], ["J1", 14, 6, 8]])
    [['', 'M1', 'M2', 'M3'], ['J1', '+', '-', '-']]


    mat = [["", "M1"], ["J1", 14], ["J2", 7], ["J3", 3]]
    >>> initialization([["", "M1"], ["J1", 14], ["J2", 7], ["J3", 3]])
    [['', 'M1'], ['J1', '+'], ['J2', '+'], ['J3', '+']]

    mat = [["", "M1", "M2", "M3"],
          ["J1", -1, -1, -1]]
    >>> initialization([["", "M1", "M2", "M3"], ["J1", -1, -1, -1]])
    Traceback (most recent call last):
        ...
    ValueError: bad matrix with number that is smaller than 0

    mat = [[""]]
    >>> initialization([[""]])
    Traceback (most recent call last):
        ...
    ValueError: bad matrix


    :param matrix:
    :return:
    '''

    random.seed(2)
    if len(matrix) <= 1 or len(matrix[0]) <= 1:
        logging.error("bad matrix")
        raise ValueError("bad matrix")
    solution2 = copy.deepcopy(matrix)

    for i in range(len(solution2) - 1):
        for j in range(len(solution2[0]) - 1):
            if solution2[i + 1][j + 1] <= 0:
                logging.error("bad matrix with number that is smaller than 0")
                raise ValueError("bad matrix with number that is smaller than 0")
            solution2[i + 1][j + 1] = "-"
    i = 0
    while i < len(solution2) - 1:
        x = random.randint(1, len(matrix[0]) - 1)

        solution2[i + 1][x] = "+"
        i = i + 1

    logging.info("Initialized matrix:  %s", solution2)

    return solution2


def check(matrix, sol1, sol2, j, k):
    '''
    checks which of 2 matrices is better, after changing the neighbors or before
    it checks the changed columns, j and k is the columns
    returns true if its better to change, the old sum and the new sum

    [['', 'M1', 'M2', 'M3'], ['J1', 14, 6, 8]]
    [['', 'M1', 'M2', 'M3'], ['J1', '-', '-', '+']]
    [['', 'M1', 'M2', 'M3'], ['J1', '-', '+', '-']]
    2
    3
    >>> check([['', 'M1', 'M2', 'M3'], ['J1', 14, 6, 8]], [['', 'M1', 'M2', 'M3'], ['J1', '-', '-', '+']], [['', 'M1', 'M2', 'M3'], ['J1', '-', '+', '-']], 2, 3)
    (True, 8, 6)


    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', 14, 6, 8, 20], ['J2', 7, 4, 11, 20], ['J3', 3, 21, 5, 20], ['J4', 1, 1, 5, 20]]
    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '-', '+'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '-', '+']]
    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '-', '+'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '+', '-']]
    3
    4
    >>> check([['', 'M1', 'M2', 'M3', 'M4'], ['J1', 14, 6, 8, 20], ['J2', 7, 4, 11, 20], ['J3', 3, 21, 5, 20], ['J4', 1, 1, 5, 20]], [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '-', '+'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '-', '+']], [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '-', '+'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '+', '-']], 3, 4)
    (True, 40, 16)

    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', 14, 6, 8, 20], ['J2', 7, 4, 11, 20], ['J3', 3, 21, 5, 20], ['J4', 1, 1, 5, 20]]
    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '+', '-'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '+', '-', '-']]
    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '+', '-'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '+', '-']]
    3
    2
    >>> check([['', 'M1', 'M2', 'M3', 'M4'], ['J1', 14, 6, 8, 20], ['J2', 7, 4, 11, 20], ['J3', 3, 21, 5, 20], ['J4', 1, 1, 5, 20]], [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '+', '-'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '+', '-', '-']], [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '+', '-'], ['J2', '-', '-', '+', '-'], ['J3', '+', '-', '-', '-'], ['J4', '-', '-', '+', '-']], 3, 2)
    (False, 1, 24)

    :param sol2:
    :param sol1:
    :param matrix:
    :param mat1:
    :param mat2:
    :return:
    '''
    sum1 = 0
    sum2 = 0
    for i in range(len(matrix) - 1):
        if sol1[i + 1][k] == "+":
            sum1 = sum1 + int(matrix[i + 1][k])

    for i in range(len(matrix) - 1):
        if sol2[i + 1][j] == "+":
            sum2 = sum2 + int(matrix[i + 1][j])

    if sum2 < sum1:
        return True, sum1, sum2

    if sum2 == sum1:
        subSum1 = subSum2 = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if sol1[i][j] == "+":
                    subSum1 = subSum1 + matrix[i][j]
                if sol2[i][j] == "+":
                    subSum2 = subSum2 + matrix[i][j]
        if subSum1 >= subSum2:
            return True, sum1, sum2

    return False, sum1, sum2


def result(sol):
    res = {"J1": -1}
    for idx in range(len(sol) - 1):
        bap = "J" + str(idx + 1)
        for idj in range(len(sol[0]) - 1):
            # print(idx+1,idj+1)
            if sol[idx + 1][idj + 1] == "+":
                # print(idx+1,idj+1)
                bap2 = "M" + str(idj + 1)
                res.update({bap: bap2})
    logging.info(res)
    return res

def reult_to_show(sol, mat):
    res_to_show = {"M1":-1}
    for idx in range(len(sol[0]) - 1):
        bap = "M" + str(idx + 1)
        bap2 = ""
        time = 0
        for idj in range(len(sol) - 1):
            # print(idx+1,idj+1)
            if sol[idj + 1][idx + 1] == "+":
                time = time + mat[idj + 1][idx + 1]
                # print(idx+1,idj+1)
                bap2 = bap2 + "J" + str(idj + 1) + ", "
        bap2 = bap2 + str(time)
        res_to_show.update({bap: bap2})
    logging.info(res_to_show)
    return res_to_show


def unrelated_parallel_machine_scheduling(matrix, mod, kToEnd=1000):
    '''
    main function with temperatures
    mat = [["", "M1", "M2", "M3", "M4"],
          ["J1", 14, 6, 8, 20],
          ["J2", 7, 4, 11, 20],
          ["J3", 3, 21, 5, 20],
          ["J4", 1, 1, 5, 20]]
    [['', 'M1', 'M2', 'M3', 'M4'], ['J1', '-', '-', '+', '-'], ['J2', '-', '+', '-', '-'], ['J3', '+', '-', '-', '-'], ['J4', '+', '-', '-', '-']]

    >>> unrelated_parallel_machine_scheduling([["", "M1", "M2", "M3", "M4"], ["J1", 14, 6, 8, 20], ["J2", 7, 4, 11, 20], ["J3", 3, 21, 5, 20], ["J4", 1, 1, 5, 20]],1)
    {'J1': 'M3', 'J2': 'M2', 'J3': 'M1', 'J4': 'M1'}

    mat = [["", "M1", "M2"],
          ["J1", 14, 6],
          ["J2", 7, 4]]
    [['', 'M1', 'M2'], ['J1', '-', '+'], ['J2', '+', '-']]

    >>> unrelated_parallel_machine_scheduling([["", "M1", "M2"], ["J1", 14, 6], ["J2", 7, 4]],1)
    {'J1': 'M2', 'J2': 'M1'}

    mat = [["", "M1", "M2", "M3"],
          ["J1", 14, 60, 80],
          ["J2", 7, 40, 110],
          ["J3", 3, 210, 50]]

    [['', 'M1', 'M2', 'M3'], ['J1', '+', '-', '-'], ['J2', '+', '-', '-'], ['J3', '+', '-', '-']]
    >>> unrelated_parallel_machine_scheduling([["", "M1", "M2", "M3"], ["J1", 14, 60, 80], ["J2", 7, 40, 110], ["J3", 3, 210, 50]],1)
    {'J1': 'M1', 'J2': 'M1', 'J3': 'M1'}

    mat = [["", "M1"],
          ["J1", 14],
          ["J2", 7],
          ["J3", 3]]
    [['', 'M1'], ['J1', '+'], ['J2', '+'], ['J3', '+']]
    >>> unrelated_parallel_machine_scheduling([["", "M1"], ["J1", 14], ["J2", 7], ["J3", 3]],1)
    {'J1': 'M1', 'J2': 'M1', 'J3': 'M1'}


    mat = [["", "M1", "M2", "M3"],
          ["J1", 14, 6, 8]]
    [['', 'M1', 'M2', 'M3'], ['J1', '-', '+', '-']]
    >>> unrelated_parallel_machine_scheduling([["", "M1", "M2", "M3"], ["J1", 14, 6, 8]],1)
    {'J1': 'M2'}

    mat = [["", "M1", "M2", "M3"],
          ["J1", -1, -1, -1]]
    >>> unrelated_parallel_machine_scheduling([["", "M1", "M2", "M3"], ["J1", -1, -1, -1]],1)
    Traceback (most recent call last):
        ...
    ValueError: bad matrix with number that is smaller than 0

    mat = [[""]]
    >>> unrelated_parallel_machine_scheduling([[""]],1)
    Traceback (most recent call last):
        ...
    ValueError: bad matrix

    :param mod:
    :param matrix:
    :return:
    '''
    random.seed(2)
    if len(matrix) <= 1 or len(matrix[0]) <= 1:
        logging.error("bad matrix")
        raise ValueError("bad matrix")
    sol1 = initialization(matrix)

    if sol1 == -1:
        logging.error("bad matrix")
        raise ValueError("bad matrix")

    if len(matrix[0]) <= 2:
        return result(sol1)

    kToEnd = kToEnd
    temp = InitialTemp
    b = (InitialTemp - LastTemp) / ((kToEnd - 1) * InitialTemp * LastTemp)
    logging.info("Beta: %s", b)
    for x in range(kToEnd):
        i = random.randint(1, len(sol1) - 1)  # random job
        k = 0  # the machine that working on i
        for z in range(len(sol1[0]) - 1):
            if sol1[i][z + 1] == "+":
                k = z + 1
                break
        j = random.randint(1, len(sol1[0]) - 1)  # random machine j
        while j == k:  # if j and k the same machine nothing will happen, we don't want this
            j = random.randint(1, len(sol1[0]) - 1)

        sol2 = copy.deepcopy(sol1)
        sol2[i][k] = "-"
        sol2[i][j] = "+"

        toChangeOrNot, old_cost, new_cost = check(matrix, sol1, sol2, j, k)

        temp = temp / (1 + b * temp)
        p = math.e ** ((- new_cost - old_cost) / temp)
        logging.info("probability DELTA %s:  %s", x + 1, p)
        logging.info("temperature%s:  %s", x + 1, temp)
        if toChangeOrNot is True and random.random() < p:
            sol1 = copy.deepcopy(sol2)
    reult_to_show(sol1, matrix)
    if mod == 1:
        return result(sol1)
    else:
        print("bapbap",reult_to_show(sol1, matrix))
        return reult_to_show(sol1, matrix)


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)

    (failures, tests) = doctest.testmod()
    print("{} failures, {} tests".format(failures, tests))
    unittest.main()

    logging.disable(logging.NOTSET)

