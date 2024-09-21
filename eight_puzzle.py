import tkinter as tk
from tkinter import filedialog, messagebox
import random
import time
from collections import deque
import heapq
from PIL import Image, ImageTk, ImageDraw, ImageFont

TARGET_STATE = [1, 2, 3, 4, 5, 6, 7, 8, 0]  # 0 represents the empty space

class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8 Puzzle Game")
        self.root.geometry('700x700')  # Set the default window size to 600x600
        self.root.configure(bg='white')  # Set the background color to white
        self.initial_state = TARGET_STATE[:]
        self.current_state = self.initial_state[:]
        self.steps = 0
        self.buttons = []
        self.tile_images = {}
        self.create_widgets()
        self.update_puzzle()

    def create_widgets(self):
        # Basic button style
        self.btn_style = {
            'font': ('Helvetica', 20),
            'borderwidth': 2,
            'relief': 'raised'
        }

        # Create the puzzle grid
        self.frame = tk.Frame(self.root, bg='black')
        self.frame.pack(pady=20)

        for i in range(9):
            btn = tk.Button(self.frame, text='', **self.btn_style,
                            command=lambda idx=i: self.move_tile(idx))
            btn.grid(row=i // 3, column=i % 3, padx=1, pady=1)
            self.buttons.append(btn)

        # Control buttons
        self.shuffle_button = tk.Button(self.root, text='Shuffle', command=self.shuffle, width=20, height=2)
        self.shuffle_button.pack(pady=5)

        self.solve_bfs_button = tk.Button(self.root, text='Solve with BFS', command=self.solve_bfs, width=20, height=2)
        self.solve_bfs_button.pack(pady=5)

        self.solve_dfs_button = tk.Button(self.root, text='Solve with DFS', command=self.solve_dfs, width=20, height=2)
        self.solve_dfs_button.pack(pady=5)

        self.solve_astar_button = tk.Button(self.root, text='Solve with A*', command=self.solve_astar, width=20, height=2)
        self.solve_astar_button.pack(pady=5)

        self.reset_button = tk.Button(self.root, text='Reset', command=self.reset, width=20, height=2)
        self.reset_button.pack(pady=5)

        self.upload_button = tk.Button(self.root, text='Upload Background', command=self.upload_background, width=20, height=2)
        self.upload_button.pack(pady=5)

        # Label for steps and time
        self.info_label = tk.Label(self.root, text='', font=('Helvetica', 16), bg='white', fg='black')
        self.info_label.pack(pady=5)

    def update_puzzle(self):
        for i in range(9):
            value = self.current_state[i]
            btn = self.buttons[i]
            if self.tile_images:
                btn.config(image=self.tile_images[value], text='', state='normal')
                btn.image = self.tile_images[value]
            else:
                if value == 0:
                    btn.config(text='', state='disabled', bg='gray')
                else:
                    btn.config(text=str(value), state='normal', bg='lightblue')

    def move_tile(self, idx):
        zero_idx = self.current_state.index(0)
        if self.is_adjacent(idx, zero_idx):
            self.current_state[zero_idx], self.current_state[idx] = self.current_state[idx], self.current_state[zero_idx]
            self.update_puzzle()
            if self.current_state == TARGET_STATE:
                messagebox.showinfo("Congratulations", "You solved the puzzle!")

    def is_adjacent(self, idx1, idx2):
        row1, col1 = idx1 // 3, idx1 % 3
        row2, col2 = idx2 // 3, idx2 % 3
        return abs(row1 - row2) + abs(col1 - col2) == 1

    def shuffle(self):
        self.current_state = self.initial_state[:]
        random.shuffle(self.current_state)
        while not self.is_solvable(self.current_state):
            random.shuffle(self.current_state)
        self.update_puzzle()
        self.info_label.config(text='')
        print("Generated initial state:", self.current_state)
        if self.is_solvable(self.current_state):
            print("This initial state is solvable, starting search!")
        else:
            print("This initial state is unsolvable!")

    def reset(self):
        self.current_state = self.initial_state[:]
        self.update_puzzle()
        self.info_label.config(text='')

    def upload_background(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            img = Image.open(filepath)
            img = img.resize((300, 300))
            tiles = {}
            count = 1
            for i in range(3):
                for j in range(3):
                    left = j * 100
                    upper = i * 100
                    right = left + 100
                    lower = upper + 100
                    tile = img.crop((left, upper, right, lower))
                    draw = ImageDraw.Draw(tile)
                    if count <= 8:
                        text = str(count)
                    else:
                        text = ''
                    try:
                        # Update the font path to a common font
                        font = ImageFont.truetype("Arial.ttf", 50)
                    except IOError:
                        print("Could not find Arial.ttf, using default font")
                        font = ImageFont.load_default()
                    bbox = font.getbbox(text)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    draw.text(((100 - w) / 2, (100 - h) / 2), text, fill="white", font=font)
                    tile = ImageTk.PhotoImage(tile)
                    tiles[count % 9] = tile
                    count += 1
            self.tile_images = tiles
            self.update_puzzle()

    def run_all_algorithms(self):
        results = {}

        # BFS
        start_time = time.time()
        path_bfs, steps_bfs, visited_nodes_bfs = self.bfs(self.current_state)
        end_time = time.time()
        elapsed_time_bfs = end_time - start_time
        results['BFS'] = {
            'path': path_bfs,
            'steps': steps_bfs,
            'visited_nodes': visited_nodes_bfs,
            'time': elapsed_time_bfs
        }

        # DFS
        start_time = time.time()
        path_dfs, steps_dfs, visited_nodes_dfs = self.dfs(self.current_state)
        end_time = time.time()
        elapsed_time_dfs = end_time - start_time
        results['DFS'] = {
            'path': path_dfs,
            'steps': steps_dfs,
            'visited_nodes': visited_nodes_dfs,
            'time': elapsed_time_dfs
        }

        # A*
        start_time = time.time()
        path_astar, steps_astar, visited_nodes_astar = self.a_star(self.current_state)
        end_time = time.time()
        elapsed_time_astar = end_time - start_time
        results['A*'] = {
            'path': path_astar,
            'steps': steps_astar,
            'visited_nodes': visited_nodes_astar,
            'time': elapsed_time_astar
        }

        return results

    def display_results(self, results, algorithm_name):
        res = results[algorithm_name]
        if res['path']:
            self.info_label.config(text=f"Algorithm: {algorithm_name}, Steps: {res['steps']}, Time: {res['time']:.2f}s")
            self.animate_solution(res['path'])
        else:
            messagebox.showinfo("Unsolvable", f"No solution found with {algorithm_name}.")

    def solve_bfs(self):
        results = self.run_all_algorithms()
        # Print the outputs
        for alg in ['BFS', 'DFS', 'A*']:
            res = results[alg]
            print(f"\nSolving with {alg}:")
            if res['path']:
                moves = self.get_move_sequence(res['path'])
                print(f"Solution path: {moves}")
                print(f"Solution steps: {res['steps']}")
                print(f"Search space size: {res['visited_nodes']}")
                print(f"Time taken: {res['time']:.2f}s")
            else:
                print("No solution found.")
        # Compare times
        fastest_alg = min(results.keys(), key=lambda x: results[x]['time'])
        print(f"\nThe fastest algorithm is {fastest_alg} with time {results[fastest_alg]['time']:.2f}s")
        # Display and animate BFS solution
        self.display_results(results, 'BFS')

    def solve_dfs(self):
        results = self.run_all_algorithms()
        # Print the outputs
        for alg in ['BFS', 'DFS', 'A*']:
            res = results[alg]
            print(f"\nSolving with {alg}:")
            if res['path']:
                moves = self.get_move_sequence(res['path'])
                print(f"Solution path: {moves}")
                print(f"Solution steps: {res['steps']}")
                print(f"Search space size: {res['visited_nodes']}")
                print(f"Time taken: {res['time']:.2f}s")
            else:
                print("No solution found.")
        # Compare times
        fastest_alg = min(results.keys(), key=lambda x: results[x]['time'])
        print(f"\nThe fastest algorithm is {fastest_alg} with time {results[fastest_alg]['time']:.2f}s")
        # Display and animate DFS solution
        self.display_results(results, 'DFS')

    def solve_astar(self):
        results = self.run_all_algorithms()
        # Print the outputs
        for alg in ['BFS', 'DFS', 'A*']:
            res = results[alg]
            print(f"\nSolving with {alg}:")
            if res['path']:
                moves = self.get_move_sequence(res['path'])
                print(f"Solution path: {moves}")
                print(f"Solution steps: {res['steps']}")
                print(f"Search space size: {res['visited_nodes']}")
                print(f"Time taken: {res['time']:.2f}s")
            else:
                print("No solution found.")
        # Compare times
        fastest_alg = min(results.keys(), key=lambda x: results[x]['time'])
        print(f"\nThe fastest algorithm is {fastest_alg} with time {results[fastest_alg]['time']:.2f}s")
        # Display and animate A* solution
        self.display_results(results, 'A*')

    def animate_solution(self, path):
        if path:
            self.current_state = path[0]
            self.update_puzzle()
            self.root.after(500, lambda: self.animate_solution(path[1:]))
        else:
            if self.current_state == TARGET_STATE:
                messagebox.showinfo("Solved", "The puzzle is solved!")

    def get_move_sequence(self, path):
        moves = []
        for i in range(1, len(path)):
            prev_state = path[i - 1]
            curr_state = path[i]
            zero_prev = prev_state.index(0)
            zero_curr = curr_state.index(0)
            diff = zero_curr - zero_prev
            if diff == -3:
                moves.append('Up')
            elif diff == 3:
                moves.append('Down')
            elif diff == -1:
                moves.append('Left')
            elif diff == 1:
                moves.append('Right')
        return moves

    def get_valid_moves(self, zero_idx):
        moves = []
        row, col = zero_idx // 3, zero_idx % 3
        if row > 0:
            moves.append(-3)  # Up
        if row < 2:
            moves.append(3)   # Down
        if col > 0:
            moves.append(-1)  # Left
        if col < 2:
            moves.append(1)   # Right
        return moves

    def bfs(self, start_state):
        queue = deque()
        queue.append((start_state, []))
        visited = set()
        visited.add(tuple(start_state))
        while queue:
            current_state, path = queue.popleft()
            if current_state == TARGET_STATE:
                return path + [current_state], len(path), len(visited)
            zero_idx = current_state.index(0)
            for move in self.get_valid_moves(zero_idx):
                new_state = current_state[:]
                new_idx = zero_idx + move
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    queue.append((new_state, path + [new_state]))
        return None, -1, len(visited)

    def dfs(self, start_state):
        stack = []
        stack.append((start_state, []))
        visited = set()
        visited.add(tuple(start_state))
        max_depth = 50  # Set a reasonable limit to prevent infinite loops
        while stack:
            current_state, path = stack.pop()
            if current_state == TARGET_STATE:
                return path + [current_state], len(path), len(visited)
            if len(path) >= max_depth:
                continue
            zero_idx = current_state.index(0)
            for move in reversed(self.get_valid_moves(zero_idx)):
                new_state = current_state[:]
                new_idx = zero_idx + move
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    stack.append((new_state, path + [new_state]))
        return None, -1, len(visited)

    def manhattan_distance(self, state):
        distance = 0
        for i, value in enumerate(state):
            if value != 0:
                target_index = TARGET_STATE.index(value)
                distance += abs(target_index % 3 - i % 3) + abs(target_index // 3 - i // 3)
        return distance

    def a_star(self, start_state):
        heap = []
        heapq.heappush(heap, (self.manhattan_distance(start_state), start_state, []))
        visited = set()
        visited.add(tuple(start_state))
        while heap:
            _, current_state, path = heapq.heappop(heap)
            if current_state == TARGET_STATE:
                return path + [current_state], len(path), len(visited)
            zero_idx = current_state.index(0)
            for move in self.get_valid_moves(zero_idx):
                new_state = current_state[:]
                new_idx = zero_idx + move
                new_state[zero_idx], new_state[new_idx] = new_state[new_idx], new_state[zero_idx]
                if tuple(new_state) not in visited:
                    visited.add(tuple(new_state))
                    new_path = path + [new_state]
                    f = len(new_path) + self.manhattan_distance(new_state)
                    heapq.heappush(heap, (f, new_state, new_path))
        return None, -1, len(visited)

    def is_solvable(self, state):
        inversions = self.count_inversions(state)
        return inversions % 2 == 0

    def count_inversions(self, state):
        inversions = 0
        state_list = [num for num in state if num != 0]
        for i in range(len(state_list)):
            for j in range(i + 1, len(state_list)):
                if state_list[i] > state_list[j]:
                    inversions += 1
        return inversions


if __name__ == '__main__':
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()
