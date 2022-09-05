#! /bin/python

from itertools import *

#
# Section I. -------------------------------------------------------------------
#

with open("depths_input", "r") as file:
    depths = [
        int(depth.strip()) for depth in file.readlines() if depth.strip() != ""
    ]


def has_positive_delta(couple):
    return couple[1] > couple[0]


def pair_with_next(sequence):
    return zip(sequence, sequence[1:])


print("Part One: ", end="") # 1502
print(len(list(filter(has_positive_delta, pair_with_next(depths)))))

summed_windows = [sum(window) for window in zip(depths, depths[1:], depths[2:])]

print("Part Two: ", end="") # 1538
print(len(list(filter(has_positive_delta, pair_with_next(summed_windows)))))

#
# Section II. ------------------------------------------------------------------
#

directions = []
with open("directions_input", "r") as file:
    for positions in file:
        if positions.strip() != "":
            directions.append(tuple(positions.strip().split(" ")))


class Ship:
    pos = 0
    depth = 0
    bearing = 0

    def location(self):
        return self.pos * self.depth


ship = Ship()

for direction in directions:
    (direction, value) = direction
    value = int(value)

    match direction:
        case "forward":
            ship.pos += value
        case "up":
            ship.depth -= value
        case "down":
            ship.depth += value

print("Part Three: ", end="")  # 1524750
print(ship.location())

ship = Ship()

for direction in directions:
    (direction, value) = direction
    value = int(value)

    match direction:
        case "forward":
            ship.pos += value
            ship.depth += ship.bearing * value
        case "up":
            ship.bearing -= value
        case "down":
            ship.bearing += value

print("Part Four: ", end="")  # 1592426537
print(ship.location())

#
# Section III. -----------------------------------------------------------------
#

# TODO: This section needs performance improvements

numbers = []
with open("diagnostics_input", "r") as file:
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

#
# Section IV. ------------------------------------------------------------------
#

raw_line_input = []
with open("bingo_input", "r") as file:
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

#
# Section V. -------------------------------------------------------------------
#

# I gave up and copied this one from Reddit user /u/KronoLord at
# https://www.reddit.com/r/adventofcode/comments/r9824c/comment/hp0swd8/?utm_source=share&utm_medium=web2x&context=3

read_data = None
with open("vents_input") as f:
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

#
# Section VI. ------------------------------------------------------------------
#

with open("lanternfish_input", "r") as file:
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

#
# Section VII. -----------------------------------------------------------------
#

# TODO: This section needs performance improvements

with open("crabs_input", "r") as file:
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

#
# Section VIII. ----------------------------------------------------------------
#

# This one almost broke me and I gave up on solving AoC for several months, but in the
#   end I ended up really happy with the solution I got.

lines = []

with open('signals_input') as f:
    for entry in f.read().split('\n')[:-1]:
        lines.append(tuple(entry.split(' | ')))

class Decoder:

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
    signal_map = lambda self: {self.segment_map[i]:i for i in self.segment_map}

    def ResetSegmentMap(self):
        self.segment_map = {
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

    def DecodeUniqueInputs(self, input):
        for signal in input.split():
            signal_len = len(signal) # No. of segments in the signal
            wires = frozenset(signal) # Set of segments in the signal
            match signal_len:
                case 2: # If 2 segments are on, digit is 1
                    self.segment_map[1] = wires
                case 3: # If 3 segments are on, digit is 7
                    self.segment_map[7] = wires
                case 4: # If 4 segments are on, digit is 4
                    self.segment_map[4] = wires
                case 7: # If 7 segments are on, digit is 8
                    self.segment_map[8] = wires

    def DecodeNonUniqueInputs(self, input):
        seg_map = self.segment_map
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

    def DecodeOutput(self, output):
        # Maps a signal to a digit
        signal_map = self.signal_map()
        output_string = ""

        # We decode each signal and append it to the string
        for signal in output.split():
            output_string += str(signal_map[frozenset(signal)])

        return int(output_string)

    def Decode(self, line):
        (input,output) = line[0],line[1]
        self.ResetSegmentMap()
        self.DecodeUniqueInputs(input)
        self.DecodeNonUniqueInputs(input)
        return self.DecodeOutput(output)

decoder = Decoder()
outputs = [decoder.Decode(line) for line in lines]

def digitsCounter(number):
    number_str = str(number)
    return  number_str.count('1') +\
            number_str.count('4') +\
            number_str.count('7') +\
            number_str.count('8')

print("Part Fifteen:", sum(map(digitsCounter, outputs))) # 519
print("Part Sixteen:", sum(outputs)) # 1027483
