############################################################
# CMPSC 442: Uninformed Search
############################################################

student_name = "Justin Cote"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.

# basic python functions
import math
import random

# https://docs.python.org/3/library/copy.html from online lib
import copy

# from slides
from collections import deque


############################################################
# Section 1: N-Queens
############################################################

def num_placements_all(n):
    if n < 0:
        return 0
    else:
        x = n * n
        return math.comb(x,n)

def num_placements_one_per_row(n):
    if n < 0:
        return 0
    else:
        return n ** n

def n_queens_valid(board):
    # get number of rows
    rows = len(board)

    # current row in reference (1 queen exists)
    for i in range(rows): 
        # row being referenced against (1 queen exists)
        for j in range(i+1,rows): 
            # check if queen position passes column restriction
            if board[i] == board[j]:
                return False
            # check if queen positions passes diagonal restriction
            if abs(board[i]-board[j]) == abs(i-j):
                return False
    return True

def n_queens_solutions(n):
    frontier = [[] + [i] for i in range(n-1,-1,-1)]
    while frontier:
        board = frontier.pop()
        if n_queens_valid(board):
            if len(board) == n:
                yield board
            else:
                frontier.extend([board + [i] for i in range(n-1,-1,-1)])
    
############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        

    def get_board(self):
        return self.board
        

    def perform_move(self, row, col):

        board = self.board

        rows, cols = len(board), len(board[0])

        # define a list of where neighbors exist
        neighbors = [
            (-1,0), #above
            (1,0),  #below
            (0,-1), #left
            (0,1),  #right
        ]

        # flip target cell
        board[row][col] = not board[row][col]

        # loop through neighbors
        for r, c in neighbors:
            new_row, new_col = row+r, col+c

            #check position on table (corner or edge)
            if 0 <= new_row < rows and 0 <= new_col < cols:
                board[new_row][new_col] = not board[new_row][new_col]

        self.board = board

    def scramble(self):
        rows, cols = len(self.board), len(self.board[0])
        for row in range(rows):
            for col in range(cols):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        return  not any(True in row for row in self.board)

    # https://docs.python.org/3/library/copy.html for deepcopy function
    def copy(self):
        new_board = copy.deepcopy(self.board)
        new_puzzle = LightsOutPuzzle(new_board)
        return new_puzzle

    def successors(self):
        rows, cols = len(self.board), len(self.board[0])
        for row in range(rows):
            for col in range(cols):
                self.perform_move(row, col)
                new_puzzle = self.copy()
                yield ((row, col), new_puzzle)
                #need to undo the move
                self.perform_move(row, col)

    def find_solution(self):
        initial_state = [row[:] for row in self.board]
        frontier = deque([(initial_state, [])])
        visited = set(tuple(map(tuple,initial_state)))
        
        while frontier:
            current_state, current_moves = frontier.popleft()
            puzzle = self.copy()
            puzzle.board = current_state

            # check if current node is solved
            if puzzle.is_solved():
                return current_moves
            
            # explore current node
            for move, new_puzzle in puzzle.successors():
                successor_state = new_puzzle.get_board()
                if tuple(map(tuple, successor_state)) not in visited:
                    frontier.append((successor_state, current_moves + [move]))
                    visited.add(tuple(map(tuple,successor_state)))
        return None
    
def create_puzzle(rows, cols):
    board = []
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(False)
        board.append(row)
    return LightsOutPuzzle(board)    

############################################################
# Section 3: Linear Disk Movement
############################################################

# my helper class
class IdenticalDiskGrid():
    def __init__(self, length, n):
        self.length = length
        self.n = n
        self.grid = []
        self.solution = []

    def buildGrid(self):
        self.grid = [1] * self.n + [0] * (self.length - self.n)
        self.solution = [0] * (self.length - self.n) + [1] * self.n

    def is_solved(self, grid):
        return grid == self.solution
    
    def perform_move(self, grid, i, j):
        new_grid = copy.deepcopy(grid)
        new_grid[i], new_grid[j] = new_grid[j] , new_grid[i]
        return new_grid

    # checks possible moves for each disc in given grid
    def generate_moves(self, grid):
        # iterate through whole list
        for i in range(self.length):
            if grid[i] == 1:
                # Move to the right 
                if i < self.length - 1 and grid[i+1] == 0:
                    yield ((i, i+1), self.perform_move(grid,i,i+1))
                # Move to the left
                if i > 0 and grid[i - 1] == 0:
                    yield ((i, i-1), self.perform_move(grid, i, i-1))
                # Jump to the right
                if i < self.length - 2 and grid[i + 2] == 0 and grid[i + 1] == 1:
                    yield ((i, i + 2), self.perform_move(grid, i, i + 2))
                # Jump to the left
                if i > 1 and grid[i - 2] == 0 and grid[i - 1] == 1:
                    yield ((i, i - 2), self.perform_move(grid, i, i - 2))


    def solve_disks(self):
        frontier = deque([(self.grid, [])])
        visited = set((self.grid))

        while frontier:
            current_state, current_moves = frontier.popleft()

            # Check if the solution state is reached
            if self.is_solved(current_state):
                return current_moves

            # Iterate through next possible moves
            for move, new_state in self.generate_moves(current_state):
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    frontier.append((new_state, current_moves + [move]))
        

        return None

# utilizes class above
def solve_identical_disks(length, n):
    grid = IdenticalDiskGrid(length,n)
    grid.buildGrid()
    solution = grid.solve_disks()
    return solution

# pretty much copied from IdenticalDiskGrid with a few modifications
class DistinctDiskGrid():
    def __init__ (self,length,n):
        self.length = length
        self.n = n
        self.grid = []
        self.solution = []

    def buildGrid(self):
        self.grid = [{i: "disk"} if i < self.n else 0 for i in range(self.length)]

    def buildSolution(self):
        solution = copy.deepcopy(self.grid)
        for i in range(self.n -1 , -1 , -1):
            disk = solution.pop(i)
            solution.append(disk)
        self.solution = solution
    
    def is_solved(self, grid):
        return grid == self.solution
    
    def perform_move(self, grid, i, j):
        new_grid = copy.deepcopy(grid)
        new_grid[i], new_grid[j] = new_grid[j] , new_grid[i]
        return new_grid

    def generate_moves(self, grid):
        # iterate through whole grid to find each possible move
        for i in range(self.length):
            if grid[i] != 0:
                # Move to the right 
                if i < self.length - 1 and grid[i+1] == 0:
                    yield ((i, i+1), self.perform_move(grid,i,i+1))
                # Move to the left
                if i > 0 and grid[i - 1] == 0:
                    yield ((i, i-1), self.perform_move(grid, i, i-1))
                # Jump to the right
                if i < self.length - 2 and grid[i + 2] == 0 and grid[i + 1] != 0:
                    yield ((i, i + 2), self.perform_move(grid, i, i + 2))
                # Jump to the left
                if i > 1 and grid[i - 2] == 0 and grid[i - 1] != 0:
                    yield ((i, i - 2), self.perform_move(grid, i, i - 2))

    def solve_disks(self):
        frontier = deque([(self.grid, [])])
        visited = [self.grid]

        while frontier:
            current_state, current_moves = frontier.popleft()

            # Check if the solution state is reached
            if self.is_solved(current_state):
                return current_moves

            # Iterate through next possible moves
            for move, new_state in self.generate_moves(current_state):
                if new_state not in visited:
                    visited.append(new_state)
                    frontier.append((new_state, current_moves + [move]))
        return None
 
# utilizes class from above
def solve_distinct_disks(length, n):
    grid = DistinctDiskGrid(length,n)
    grid.buildGrid()
    grid.buildSolution()
    solution = grid.solve_disks()
    return solution
