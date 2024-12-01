#! /bin/env python

# TODO: Add type hints
# TODO: Make input parsing consistent

#
# IMPORTS (Only stdlib allowed !!) -------------------------------------------------------------------------------------
#

from __future__ import annotations
from typing import Tuple
from itertools import islice, count
from enum import Enum
from functools import reduce
from time import perf_counter

#
# INPUTS ---------------------------------------------------------------------------------------------------------------
#

input_I = "inputs/1.in"
input_II = "inputs/2.in"
input_III = "inputs/3.in"
input_IV = "inputs/4.in"
input_V = "inputs/5.in"
input_VI = "inputs/6.in"
input_VII = "inputs/7.in"
input_VIII = "inputs/8.in"
input_IX = "inputs/9.in"
input_X = "inputs/10.in"
input_XI = "inputs/11.in"

#
# Section I. -----------------------------------------------------------------------------------------------------------
#

TIME_ONE = perf_counter()

with open(input_I, "r") as file:
    depths = [
        int(depth.strip()) for depth in file.readlines() if depth.strip() != ""
    ]

def has_positive_delta(couple):
    return couple[1] > couple[0]

def pair_with_next(sequence):
    return zip(sequence, sequence[1:])

print("Part One: ", len(list(filter(has_positive_delta, pair_with_next(depths))))) # 1502

summed_windows = [sum(window) for window in zip(depths, depths[1:], depths[2:])]

print("Part Two: ", len(list(filter(has_positive_delta, pair_with_next(summed_windows))))) # 1538

print("Time: ", perf_counter() - TIME_ONE)
print("")

#
# Section II. ----------------------------------------------------------------------------------------------------------
#

TIME_TWO = perf_counter()

directions = []
with open(input_II, "r") as file:
    for positions in file:
        if positions.strip() != "":
            directions.append(tuple(positions.strip().split(" ")))


pos = 0
depth = 0

location = lambda : pos * depth


for direction in directions:
    (direction, value) = direction
    value = int(value)

    match direction:
        case "forward":
            pos += value
        case "up":
            depth -= value
        case "down":
            depth += value

print("Part Three: ", end="")  # 1524750
print(location())

pos = 0
depth = 0
bearing = 0

for direction in directions:
    (direction, value) = direction
    value = int(value)

    match direction:
        case "forward":
            pos += value
            depth += bearing * value
        case "up":
            bearing -= value
        case "down":
            bearing += value

print("Part Four: ", end="")  # 1592426537
print(location())

print("Time: ", perf_counter() - TIME_TWO)
print("")

#
# Section III. ---------------------------------------------------------------------------------------------------------
#

# TODO: This section needs a cleaner implementation
# TODO: This section needs performance improvements

TIME_THREE = perf_counter()

numbers = []
with open(input_III, "r") as file:
    for number in file:
        if number.strip() != "":
            numbers.append(number.strip())


def get_modal_bit_(numbers, index, antimodal: bool = False):
    zeroes = 0
    ones = 0
    for num in numbers:
        if num[index] == "0":
            zeroes += 1
        if num[index] == "1":
            ones += 1
    if antimodal:
        return "0" if (zeroes <= ones) else "1"
    else:
        return "1" if (ones >= zeroes) else "0"


gamma = ""
epsilon = ""

for i in range(0, len(numbers[0])):
    gamma += get_modal_bit_(numbers, i)
    epsilon += get_modal_bit_(numbers, i, antimodal=True)

print("Part Five: ", end="")  # 4147524
print(int(gamma, 2) * int(epsilon, 2))


def filter_nums_by_modality(numbers, index, antimodal: bool):
    if len(numbers) == 1:
        return numbers
    toremove = [
        num
        for num in numbers
        if num[index] != get_modal_bit_(numbers, index, antimodal=antimodal)
    ]
    return filter_nums_by_modality(
        [number for number in numbers if not number in toremove],
        index + 1,
        antimodal=antimodal,
    )


oxygen_generator_rating = filter_nums_by_modality(numbers, 0, antimodal=False)
co2_scrubber_rating = filter_nums_by_modality(numbers, 0, antimodal=True)

print("Part Six: ", end="") # 3570354
print(int(oxygen_generator_rating[0], 2) * int(co2_scrubber_rating[0], 2))

print("Time: ", perf_counter() - TIME_THREE)
print("")

#
# Section IV. ----------------------------------------------------------------------------------------------------------
#

TIME_FOUR = perf_counter()

raw_line_input = []
with open(input_IV, "r") as file:
    draws = list(map(lambda str: int(str), file.readline().strip().split(",")))
    for positions in file:
        positions = positions.strip()
        if positions != "":
            raw_line_input.append(
                list(
                    map(
                        lambda str: int(str), filter(None, positions.split(" "))
                    )
                )
            )

boards = []
boards_marked = []

col_count = 5
row_count = 5
board_count = len(raw_line_input) // row_count

transpose_board = lambda board: list(map(list, zip(*board)))

for start_end in islice(
    zip(count(0, row_count), count(row_count, row_count)), board_count
):
    (start, end) = start_end
    boards.append(raw_line_input[start:end])
    boards_marked.append([[None] * col_count for i in range(row_count)])


def mark_board(draw):
    for board_index in range(0, board_count):
        for row_index in range(0, row_count):
            for col_index in range(0, col_count):
                value = boards[board_index][row_index][col_index]
                if value == draw:
                    boards_marked[board_index][row_index][col_index] = value


def check_boards():
    boards_with_matches = []
    for id_board in enumerate(boards_marked):
        (id, board) = id_board
        for row in board:
            for col in transpose_board(board):
                if None not in row or None not in col:
                    boards_with_matches.append(id)
    return boards_with_matches


def calculate_score(draw, board_id):
    sum = 0
    for row_index in range(0, row_count):
        for col_index in range(0, col_count):
            if boards_marked[board_id][row_index][col_index] is None:
                sum += boards[board_id][row_index][col_index]
    return sum * draw


completed = set([])
scores = []
for draw in draws:
    mark_board(draw)
    for board_id in check_boards():
        if board_id not in completed:
            completed.add(board_id)
            score = calculate_score(draw, board_id)
            scores.append(score)

print("Part Seven: ", end="") # 4662
print(scores[0])
print("Part Eight: ", end="") # 12080
print(scores[len(scores) - 1])

print("Time: ", perf_counter() - TIME_FOUR)
print("")

#
# Section V. -----------------------------------------------------------------------------------------------------------
#

# I gave up and copied this one from Reddit user /u/KronoLord at
# https://www.reddit.com/r/adventofcode/comments/r9824c/comment/hp0swd8/?utm_source=share&utm_medium=web2x&context=3

# TODO: Do this myself

TIME_FIVE = perf_counter()

read_data = None
with open(input_V, "r") as f:
    read_data = f.readlines()

max_row, max_col = 0, 0
segments = []

for positions in read_data:
    if positions.strip() != "":
        end_1, end_2 = positions.strip().split(" -> ")
        segments.append(
            (
                tuple(map(int, end_1.split(","))),
                tuple(map(int, end_2.split(","))),
            )
        )
        max_row = max(max_row, segments[-1][0][0], segments[-1][1][0])
        max_col = max(max_col, segments[-1][0][1], segments[-1][1][1])

diagram = [[0] * (max_col + 1) for _ in range(max_row + 1)]

diagonal_segments = []

for segment in segments:
    if segment[0][0] == segment[1][0]:
        min_col, max_col = min(segment[0][1], segment[1][1]), max(
            segment[0][1], segment[1][1]
        )
        for col in range(min_col, max_col + 1):
            diagram[segment[0][0]][col] += 1
    elif segment[0][1] == segment[1][1]:
        min_row, max_row = min(segment[0][0], segment[1][0]), max(
            segment[0][0], segment[1][0]
        )
        for row in range(min_row, max_row + 1):
            diagram[row][segment[0][1]] += 1
    else:
        diagonal_segments.append(segment)

# 5306
print("Part Nine:", len([val for row in diagram for val in row if val >= 2]))

for segment in diagonal_segments:
    X_incr = 1 if segment[0][0] < segment[1][0] else -1
    Y_incr = 1 if segment[0][1] < segment[1][1] else -1
    (X, Y) = segment[0]
    diagram[X][Y] += 1
    while True:
        X += X_incr
        Y += Y_incr
        diagram[X][Y] += 1
        if (X, Y) == segment[1]:
            break
# 17787
print("Part Ten:", len([val for row in diagram for val in row if val >= 2]))

print("Time: ", perf_counter() - TIME_FIVE)
print("")

#
# Section VI. ----------------------------------------------------------------------------------------------------------
#

TIME_SIX = perf_counter()

with open(input_VI, "r") as file:
    input = list(map(lambda x: int(x), file.readline().strip().split(",")))

counts = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for num in input:
    counts[num] += 1


def rotate(seed):
    return [
        seed[1],
        seed[2],
        seed[3],
        seed[4],
        seed[5],
        seed[6],
        seed[7] + seed[0],
        seed[8],
        seed[0],
    ]


for i in range(0, 80):
    counts = rotate(counts)
print("Part Eleven:", sum(counts)) # 393019

for i in range(0, 256 - 80):
    counts = rotate(counts)
print("Part Twelve:", sum(counts)) # 1757714216975

print("Time: ", perf_counter() - TIME_SIX)
print("")

#
# Section VII. ---------------------------------------------------------------------------------------------------------
#

# TODO: This section needs performance improvements

TIME_SEVEN = perf_counter()

with open(input_VII, "r") as file:
    positions = [int(x) for x in file.readline().strip().split(",")]

fuel_cons = {
    str(i): sum(map(lambda x: abs(x - i), positions))
    for i in range(min(positions), max(positions) + 1)
}

fuel_cons_2 = {
    str(i): sum(
        map(lambda x: ((abs(x - i) * (abs(x - i) + 1)) // 2), positions)
    )
    for i in range(min(positions), max(positions) + 1)
}

print("Part Thirteen:", fuel_cons[min(fuel_cons, key=fuel_cons.get)]) # 357353
print("Part Fourteen:", fuel_cons_2[min(fuel_cons_2, key=fuel_cons_2.get)]) # 104822130

print("Time: ", perf_counter() - TIME_SEVEN)
print("")

#
# Section VIII. --------------------------------------------------------------------------------------------------------
#

TIME_EIGHT = perf_counter()

lines = []

with open(input_VIII, "r") as f:
    for entry in f.read().split('\n')[:-1]:
        lines.append(tuple(entry.split(' | ')))

# Maps digits to the segment signal wires that constitute it
segment_map = {
    0:frozenset(),
    1:frozenset(),
    2:frozenset(),
    3:frozenset(),
    4:frozenset(),
    5:frozenset(),
    6:frozenset(),
    7:frozenset(),
    8:frozenset(),
    9:frozenset(),
}

# Returns an inverted version of segment map. To map signals to digits.
SIGNAL_MAP = lambda: {segment_map[i]:i for i in segment_map}

def ResetSegmentMap():
    global segment_map
    segment_map = {
        0:frozenset(),
        1:frozenset(),
        2:frozenset(),
        3:frozenset(),
        4:frozenset(),
        5:frozenset(),
        6:frozenset(),
        7:frozenset(),
        8:frozenset(),
        9:frozenset(),
    }

# We need to use frozensets because we need to hash them in signal_map

# Decode the easy digits first. Digits 1,4,7, and 8 have a unique no. of segments
def DecodeUniqueInputs(input):
    for signal in input.split():
        signal_len = len(signal) # No. of segments in the signal
        wires = frozenset(signal) # Set of segments in the signal
        match signal_len:
            case 2: # If 2 segments are on, digit is 1
                segment_map[1] = wires
            case 3: # If 3 segments are on, digit is 7
                segment_map[7] = wires
            case 4: # If 4 segments are on, digit is 4
                segment_map[4] = wires
            case 7: # If 7 segments are on, digit is 8
                segment_map[8] = wires

# We can check which segments a given signal shares with the known digits (1,4,7,8) to decode it.
def DecodeNonUniqueInputs(input):
    seg_map = segment_map
    for signal in input.split():
        signal_len = len(signal) # No. of segments in the signal
        wires = frozenset(signal) # Set of segments in the signal
        match signal_len:
            case 5: # If 5 segments are on, digit is 3 OR 2 OR 5
                if len(wires.intersection(seg_map[1])) == 2: # If it shares 2 segments with 1, digit is 3
                    seg_map[3] = wires
                else:
                    match len(wires.intersection(seg_map[4])): # If it and 4 share...
                        case 2: # ...2 segments, digit is 2
                            seg_map[2] = wires
                        case 3: # ...3 segments, digit is 5
                            seg_map[5] = wires
            case 6: # If 6 segments are on, digit is 6 OR 3 OR 4
                if len(wires.intersection(seg_map[1])) == 1: # If it shares 1 segment with 1, digit is 6
                    seg_map[6] = wires
                else:
                    match len(wires.intersection(seg_map[4])): # If it and 4 share...
                        case 3: # ...3 segments, digit is 0
                            seg_map[0] = wires
                        case 4: # ...4 segments, digit is 9
                            seg_map[9] = wires

# Uses the decoded segment-digit mappings to decode the output
def DecodeOutput(output):
    # Maps a signal to a digit
    signal_map = SIGNAL_MAP()
    output_string = ""

    # We decode each signal and append it to the string
    for signal in output.split():
        output_string += str(signal_map[frozenset(signal)])

    return int(output_string)


outputs = []

for line in lines:
    (input,output) = line[0],line[1]
    ResetSegmentMap() # Each line has a different segment-digit mapping.
    DecodeUniqueInputs(input)
    DecodeNonUniqueInputs(input)
    outputs.append(DecodeOutput(output))

def digitsCounter(number):
    number_str = str(number)
    return  number_str.count('1') +\
            number_str.count('4') +\
            number_str.count('7') +\
            number_str.count('8')

print("Part Fifteen:", sum(map(digitsCounter, outputs))) # 519
print("Part Sixteen:", sum(outputs)) # 1027483

print("Time: ", perf_counter() - TIME_EIGHT)
print("")

#
# Section IX. ----------------------------------------------------------------
#

TIME_NINE = perf_counter()

with open(input_IX, "r") as f:
    lines = f.read().split('\n')[:-1]

height: int = len(lines)
width: int = len(lines[0])

class Point:
    x: int = None
    y: int = None
    val: int = None

    def __init__(self,x,y,val=None) -> None:
        self.x = x
        self.y = y
        if val is None:
            self.val = int(lines[x][y])
        else:
            self.val = val

    def getNeighbours(self) -> list[Point]:

        neighbours: list[Point] = []

        if not self.y == 0:
            neighbours.append(Point(self.x,self.y-1))
        if not self.y == height - 1:
            neighbours.append(Point(self.x,self.y+1))
        if not self.x == 0:
            neighbours.append(Point(self.x-1,self.y))
        if not self.x == width - 1:
            neighbours.append(Point(self.x+1,self.y))

        return neighbours

    def isLowPoint(self) -> bool:

        neighbours: list[Point] = self.getNeighbours()

        for neighbour in neighbours:
            if self.val >= neighbour.val:
                return False

        return True

    def __hash__(self) -> int:
        return hash((self.x,self.y))

    def __eq__(self, __o: object) -> bool:
        return self.x == __o.x and self.y == __o.y

class Basin:
    # Use set for all points in basin so we dont have to check if we've visited a point before while finding
    points: set[Point] = set()
    low_point: Point = None
    size: int = lambda self: len(self.points)

    def __init__(self, low_point) -> None:
        self.points = set()
        self.low_point = low_point
        self.points.add(low_point)

    # Recursively finds all the points in the basin
    def Find(self, point) -> None:
        neighbours = point.getNeighbours()
        for neighbour in neighbours:
            if neighbour.val > point.val and neighbour.val != 9:
                self.points.add(neighbour)
                self.Find(neighbour)

    def __lt__(self, __o: Basin) -> True:
        return self.size() < __o.size()

low_points: list[Point] = []
for i in range(0,height):
    for k in range(0, width):
        point = Point(k,i)
        if point.isLowPoint():
            low_points.append(point)

basins: list[Basin] = []
for point in low_points:
    basin = Basin(point)
    basin.Find(point)
    basins.append(basin)

basins.sort(reverse=True)

print("Part Seventeen:", sum(map(lambda p: p.val, low_points)) + len(low_points)) # 564
print("Part Eighteen:", reduce(lambda acc,x: acc*x, map(lambda b: b.size(), basins[0:3]))) # 1038240

print("Time: ", perf_counter() - TIME_NINE)
print("")

#
# Section X. -----------------------------------------------------------------------------------------------------------
#

TIME_TEN = perf_counter()

with open(input_X, "r") as f:
    lines = f.read().split('\n')[:-1]

BRACKETS_MAP = {
    '(':')',
    '[':']',
    '{':'}',
    '<':'>',
}

OPENING_BRACKETS = list(BRACKETS_MAP.keys())

SYNTAX_ERROR_SCORES = {
    ')': 3,
    ']': 57,
    '}': 1197,
    '>': 25137,
}
assert(len(SYNTAX_ERROR_SCORES) == len(OPENING_BRACKETS))

COMPLETION_SCORES = {
    ')': 1,
    ']': 2,
    '}': 3,
    '>': 4,
}
assert(len(COMPLETION_SCORES) == len(OPENING_BRACKETS))

stack = []

def Push(bracket) -> None:
    stack.append(bracket)

# Pops from stack and checks if bracket is its closing version. If not, returns False
def Pop(bracket) -> bool:
    if BRACKETS_MAP[stack.pop()] == bracket:
        return True
    else:
        return False

class LineState(Enum):
    CORRUPT = 1
    INCOMPLETE = 2

# Pushes and pops brackets until end of line is reached or corrupt character found. Returns a LineState and
# corresponding error score
def ParseAndScore(line) -> Tuple[LineState,str]:
    for bracket in line:
        if bracket in OPENING_BRACKETS:
            Push(bracket)
        else:
            if not Pop(bracket):
                return (LineState.CORRUPT, SYNTAX_ERROR_SCORES[bracket])

    # Reached end of line without any corrupt chars. This means line is only incomplete -- proceed to calculate
    # autocompletion score for this line

    # Stack now contains all the unclosed brackets in the order they were opened.
    completion_score = 0
    for char in map(lambda c: BRACKETS_MAP[c],stack[::-1]):
        completion_score *= 5
        completion_score += COMPLETION_SCORES[char]

    return (LineState.INCOMPLETE,completion_score)

syntax_score_total = 0
completion_scores = []

for line in lines:
    stack = []
    (state,score) = ParseAndScore(line)
    match state:
        case LineState.CORRUPT:
            syntax_score_total += score
        case LineState.INCOMPLETE:
            completion_scores.append(score)

print("Part Nineteen:", syntax_score_total) # 369105
print("Part Twenty: ", sorted(completion_scores)[len(completion_scores)//2]) # 3999363569

print("Time: ", perf_counter() - TIME_TEN)
print("")

#
# Section XI. ----------------------------------------------------------------------------------------------------------
#

#TIME_ = perf_counter()

matrix = [ list(map(lambda x: int(x), list(line.strip('\n')))) for line in open(input_XI, "r").readlines() ]

print(matrix)

#print("Time: ", perf_counter() - TIME_)
#print("")
