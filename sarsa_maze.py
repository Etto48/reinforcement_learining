import time
import numpy as np
import random

def generate_maze(n):
    # Dimensions must be odd to ensure walls surround paths
    maze = [[1 for _ in range(n)] for _ in range(n)]

    def carve(x, y):
        maze[y][x] = 0
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]
        random.shuffle(directions)
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < n and 0 < ny < n and maze[ny][nx] == 1:
                maze[ny - dy // 2][nx - dx // 2] = 0  # carve wall between
                carve(nx, ny)

    carve(n//2-1, n//2-1)
    maze[1][1] = 0
    maze[n-2][n-2] = 0
    maze = np.array(maze)
    return maze

class BackwardSARSA:
    def __init__(self, maze: np.ndarray, n, g, l, eps, start=(1,1), goal=None, max_steps=None):
        self.maze = maze
        self.n = n
        self.g = g
        self.l = l
        self.eps = eps
        self.last_action = None
        self.last_state = None
        self.start = start
        self.sleep = 0.0
        self.restarts = 0
        self.best_distance = np.inf
        self.wins = 0
        self.best_steps = np.inf
        self.needs_restart = False
        self.reset()
        if goal is None:
            self.goal = (len(maze) - 2, len(maze[0]) - 2)
        else:
            self.goal = goal
        if max_steps is None:
            self.max_steps = self.maze.size * 10
        else:
            self.max_steps = max_steps
        self.q = np.zeros((len(maze), len(maze[0]), 4))
        self.e = np.zeros_like(self.q)

    def print_maze(self):
        print("\033[H\033[J", end="")
        if self.wins > 0:
            print(f"Wins: {self.wins}, Best steps: {self.best_steps}, Restarts: {self.restarts}")
        else:
            print(f"Restarts: {self.restarts}, Best distance: {self.best_distance}")
        print(f"Steps: {self.steps}")
        x, y = self.state
        print("Q(s): ", self.q[x, y])
        print("E(s): ", self.e[x, y])
        for i, row in enumerate(self.maze):
            for j,cell in enumerate(row):
                if i==x and j==y:
                    if i==self.goal[0] and j==self.goal[1]:
                        print("##", end="")
                    else:
                        print("◖◗", end="")
                elif i==self.goal[0] and j==self.goal[1]:
                    print(f"XX", end="")
                elif cell != 1:
                    print("  ", end="")
                else:
                    print("██", end="")
            print()
        print()

    def available_actions(self, state) -> np.ndarray:
        actions = [True, True, True, True]  # up, right, down, left
        x, y = state
        if x == 0 or self.maze[x - 1][y] == 1:
            actions[0] = False
        if y == len(self.maze[0]) - 1 or self.maze[x][y + 1] == 1:
            actions[1] = False
        if x == len(self.maze) - 1 or self.maze[x + 1][y] == 1:
            actions[2] = False
        if y == 0 or self.maze[x][y - 1] == 1:
            actions[3] = False
        return np.array(actions)
    
    def manual_policy(self, state) -> int:
        actions = self.available_actions(state)
        if not any(actions):
            raise ValueError("No available actions")
        # get what key was pressed
        while True:
            key = input("Enter action (w/a/s/d): ")
            if key == "w":
                action = 0
            elif key == "d":
                action = 1
            elif key == "s":
                action = 2
            elif key == "a":
                action = 3
            elif key == "":
                action = self.policy(state)
            else:
                continue
            if actions[action]:
                break
        return action

    def policy(self, state) -> int:
        actions = self.available_actions(state)
        if not any(actions):
            raise ValueError("No available actions")
        q_s = self.q[state[0], state[1]].copy()
        q_s[~actions] = -np.inf
        m = np.sum(actions)
        prob = np.zeros_like(q_s)
        eps = self.eps * (0.9 ** self.wins)
        prob[actions] = eps / m
        max_q = np.max(q_s)
        max_actions = np.where(q_s == max_q)[0]
        if max_actions.size == 0:
            raise ValueError("No best action")
        prob[max_actions] += (1-eps) / len(max_actions)
        assert np.isclose(np.sum(prob), 1)
        action = np.random.choice(np.arange(len(prob)), p=prob, size=None)
        return action

    def distance(self, state1, state2):
        return np.abs(state1[0] - state2[0]) + np.abs(state1[1] - state2[1])

    def inverse_action(self, action):
        return (action + 2) % 4

    def step(self, action) -> float:
        self.steps += 1
        x, y = self.state
        if action == 0:
            x -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x += 1
        elif action == 3:
            y -= 1
        if self.maze[x][y] == 1:
            raise ValueError("Invalid move")
        if x < 0 or x >= len(self.maze) or y < 0 or y >= len(self.maze[0]):
            raise ValueError("Out of bounds")
        self.next_state = (x, y)
        distance = self.distance(self.next_state, self.goal)
        self.best_distance = min(self.best_distance, distance)
        reward = -1
        if self.next_state == self.goal:
            reward = 100000
            self.best_steps = min(self.best_steps, self.steps)
            self.restarts += 1
            self.wins += 1
            self.max_steps = min(self.steps * 2, self.max_steps)
            self.needs_restart = True
        elif self.steps > self.max_steps:
            reward = -1000
            self.restarts += 1
            self.needs_restart = True
        return reward
    
    def reset(self):
        self.steps = 0
        self.state = self.start
        self.next_state = None
        self.last_action = None
        self.last_state = None
        self.last_reward = None
        self.needs_restart = False

    def update(self, action, reward):
        x, y = self.state
        if self.last_action is not None and self.last_state is not None and self.last_reward is not None:
            lx, ly = self.last_state

            delta = reward + self.g * self.q[x, y, action] - self.q[lx, ly, self.last_action]
            self.q += self.n * self.e * delta

        self.e *= self.g*self.l
        self.e[x, y, action] += 1

        self.last_reward = reward
        self.last_action = action
        self.last_state = (x, y)
        self.state = self.next_state
        self.next_state = None
        if self.needs_restart:
            self.reset()

    def train(self, verbose=False, max_restarts=1000):
        self.reset()
        while self.restarts < max_restarts:
                if verbose:
                    self.print_maze()
                #action = self.manual_policy(self.state)
                action = self.policy(self.state)
                reward = self.step(action)
                self.update(action, reward)
                if verbose:
                    time.sleep(self.sleep)
        if verbose:
            self.print_maze()

def main():
    maze = generate_maze(21)
    verbose = True
    agent = BackwardSARSA(maze, n=0.2, g=0.7, l=0.7, eps=0.5)
    agent.print_maze()
    input("Press Enter to start training...")
    while True:
        agent.train(verbose=verbose)
        do_continue = input("Continue (y/n/s)? ")
        if do_continue.lower() == "n":
            break
        elif do_continue.lower() == "s":
            agent.sleep = 0.1
            verbose = True
        agent.restarts = 0


if __name__ == "__main__":
    main()
