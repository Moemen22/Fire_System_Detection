import random
import tkinter as tk
import time
from tkinter import messagebox
import winsound
from main import *
def play_ringtone():
    winsound.PlaySound(r"sway_by_d_halpin.wav", winsound.SND_ASYNC)
# Maze dimensions
maze_width = 25
maze_height = 17

# Function to generate a random maze
def generate_maze(width, height, obstacle_density):
    maze = []
    for _ in range(height):
        row = []
        for _ in range(width):
            # Randomly select whether the cell is a wall or empty space
            if random.random() < obstacle_density:
                row.append("#")  # Wall
            else:
                row.append(" ")  # Empty space
        maze.append(row)
    return maze

# Function to update the maze display
def update_display():
    for i in range(maze_height):
        for j in range(maze_width):
            cell = maze[i][j]
            fill_color = 'white' if cell == ' ' else 'red' if cell == '#' else 'black'
            canvas.itemconfigure(maze_cells[i][j], fill=fill_color)
    canvas.itemconfigure(maze_cells[player_row][player_col], fill='green')
    canvas.itemconfigure(maze_cells[goal_row][goal_col], fill='blue')

# Function to handle keypress events
def on_key_press(event):
    global player_row, player_col, start_time
    key = event.keysym.lower()

    # Move the player based on the input
    if key == "up" and player_row > 0 and maze[player_row-1][player_col] != "#":
        player_row -= 1  # Move up
    elif key == "down" and player_row < maze_height-1 and maze[player_row+1][player_col] != "#":
        player_row += 1  # Move down
    elif key == "left" and player_col > 0 and maze[player_row][player_col-1] != "#":
        player_col -= 1  # Move left
    elif key == "right" and player_col < maze_width-1 and maze[player_row][player_col+1] != "#":
        player_col += 1  # Move right

    # Check if the player reached the goal
    if player_row == goal_row and player_col == goal_col:
        canvas.itemconfigure(maze_cells[goal_row][goal_col], fill='green')
        end_time = time.time()
        elapsed_time = end_time - start_time
        status_label.config(text="Congratulations! You reached the goal in {:.2f} seconds.".format(elapsed_time))
        root.unbind("<KeyPress>")
    else:
        cell = maze[player_row][player_col]
        if cell == "#" or cell == "red":
            status_label.config(text="Game Over! You lost.")
            root.unbind("<KeyPress>")
            messagebox.showinfo("Game Over", "You lost the game.")
            root.destroy()

    update_display()

# Create the game window
root = tk.Tk()
root.title("Maze Game")

# Create the canvas to draw the maze
cell_size = 40  # Adjust the size of each cell as desired
canvas_width = maze_width * cell_size
canvas_height = maze_height * cell_size
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
canvas.pack()

# Create the maze
initial_obstacle_density = 0.2  # Initial obstacle density
maze = generate_maze(maze_width, maze_height, initial_obstacle_density)
maze_cells = []
player_row = random.randint(0, maze_height-1)
player_col = random.randint(0, maze_width-1)
for i in range(maze_height):
    cell_row = []
    for j in range(maze_width):
        cell = maze[i][j]
        x1 = j * cell_size
        y1 = i * cell_size
        x2 = x1 + cell_size
        y2 = y1 + cell_size
        cell_id = canvas.create_rectangle(x1, y1, x2, y2, fill='white' if cell == ' ' else 'red')
        cell_row.append(cell_id)
    maze_cells.append(cell_row)

# Set the player's starting position and goal position
goal_row = random.randint(0, maze_height-1)
goal_col = random.randint(0, maze_width-1)
maze[player_row][player_col] = 'P'  # Player
maze[goal_row][goal_col] = 'G'  # Goal

# Bind the keypress event
root.bind("<KeyPress>", on_key_press)
root.focus_set()

# Create a status label
status_label = tk.Label(root, text="Use arrow keys to navigate. Reach the goal (blue cell).")
status_label.pack()

# Start the timer
start_time = time.time()

# Increase the size and complexity of the obstacles randomly over time
obstacle_density = initial_obstacle_density
def increase_obstacle_density():
    global obstacle_density, maze
    for i in range(maze_height):
        for j in range(maze_width):
            if maze[i][j] == "#":
                if random.random() < obstacle_density:
                    maze[i][j] = "red"  # Change the obstacle color to red
    obstacle_density -= 0.02  # Increase the obstacle density reduction for faster growth
    root.after(5000, increase_obstacle_density)

# Start increasing obstacle density after 5 seconds
root.after(5000, increase_obstacle_density)

# Play the ringtone
play_ringtone()

# Update the display
update_display()

# Start the GUI event loop
root.mainloop()