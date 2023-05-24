# import cv2
# import os
# import tkinter as tk
# from tkinter import messagebox, ttk
#
# import delete
# import detect
# import game
#
# # Load the pre-trained cascade classifier for face detection
# cascade_path = "haarcascade_frontalface_default.xml"
# face_cascade = cv2.CascadeClassifier(cascade_path)
#
# # Create a directory to save the face images
# output_dir = "faces"
#
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
#
# # Function to save the face image with the person's name
# def save_face_image(image, name):
#     filename = os.path.join(output_dir, name + ".jpg")
#     cv2.imwrite(filename, image)
#     print(f"Face image saved: {filename}")
#
#
#
# # Function to handle the name entry pop-up
# def enter_name_popup(frame):
#     def submit_name():
#         name = name_entry.get()
#         if name:
#             root.destroy()
#             save_face_image(frame, name)
#         else:
#             messagebox.showwarning("Invalid Name", "Please enter a valid name.")
#
#     # Create a pop-up window
#     root = tk.Tk()
#     root.title("Enter Name")
#
#     # Create a label and an entry widget
#     name_label = tk.Label(root, text="Enter the person's name:")
#     name_label.pack()
#     name_entry = tk.Entry(root)
#     name_entry.pack()
#
#     # Create a submit button
#     submit_button = tk.Button(root, text="Submit", command=submit_name)
#     submit_button.pack()
#
#     # Run the Tkinter event loop
#     root.mainloop()
#
#
# # Function to handle the Create button click event
# def create_button_click():
#     # Capture image from camera
#     camera = cv2.VideoCapture(0)
#
#     while True:
#         # Read frame from the camera
#         ret, frame = camera.read()
#
#         # Convert the frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#         # Detect faces in the grayscale frame
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#
#         if len(faces) == 1:
#             # Extract the face region
#             (x, y, w, h) = faces[0]
#             face_image = frame[y:y + h, x:x + w]
#
#             # Display the frame
#             cv2.imshow("Camera Feed", frame)
#
#             # Prompt the user to enter the person's name
#             enter_name_popup(face_image)
#
#             break
#
#         # Display the frame
#         cv2.imshow("Camera Feed", frame)
#
#         # Check for 'q' key press to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # Release the camera and close all windows
#     camera.release()
#     cv2.destroyAllWindows()
#
#
# def create_screen():
#     # Create a new window for the Create screen
#     create_window = tk.Toplevel(root)
#     create_window.title("Create")
#
#     # Add content to the Create screen
#     create_label = tk.Label(create_window, text="Click Create button to start capturing face image.")
#     create_label.pack()
#
#     create_button = ttk.Button(create_window, text="Create", command=create_button_click)
#     create_button.pack()
#
#
# def Read_screen():
#     fr = detect.FaceRecognition()
#     fr.run_recognition()
#
# def Delete_screen():
#     fr = delete.FaceRecognition()
#     fr.run_recognition()
#
# def admin_screen():
#     # Create a new window for the Create screen
#     admin = tk.Toplevel(root)
#     admin.title("Admin Panel")
#
#     def close_window():
#         admin.destroy()
#
#
#     fr = detect.FaceRecognition()
#     if fr.admin():
#
#         create_button = ttk.Button(admin, text="Create", command=create_screen)
#         create_button.pack()
#
#         read_button = ttk.Button(admin, text="Read", command=Read_screen)
#         read_button.pack()
#
#         update_button = ttk.Button(admin, text="Update")
#         update_button.pack()
#
#         delete_button = ttk.Button(admin, text="Delete", command=Delete_screen)
#         delete_button.pack()
#
#     else:
#         No_Admin = tk.Label(admin, text="You don't have access in this screen")
#         No_Admin.pack()
#
#         close_button = ttk.Button(admin,text="OK" ,command= close_window)
#         close_button.pack()
#
# def play_Screen():
#     play = tk.Toplevel(root)
#     play.title("Game Screen")
#
#     # fr = detect.FaceRecognition()
#     fr = game.Game1()
#     fr.run()
#
#
#     # # name = fr.play()
#     # def close_window():
#     #     play.destroy()
#     # if name:
#     #     No_Admin = tk.Label(play, text="You Must go to admin to create your profile")
#     #     No_Admin.pack()
#     #
#     #     close_button = ttk.Button(play, text="OK", command=close_window)
#     #     close_button.pack()
#
#
# def how_to_play():
#     play = tk.Toplevel(root)
#     play.title("How To Play")
#     No_Admin = tk.Label(play, text="LEASER WITH RIGHT AND DRAW CIRCLE WITH RIGHT MEAN HELP AND COUNT IT ")
#     No_Admin.pack()
#
# # Main application window
# root = tk.Tk()
# root.title("Fire System")
#
# play_button = ttk.Button(root,text="Play A Game",command=play_Screen)
# play_button.pack()
#
# admin_button = ttk.Button(root,text="Admin",command=admin_screen)
# admin_button.pack()
#
# how_to_play_button = ttk.Button(root,text="How to Play",command=how_to_play)
# how_to_play_button.pack()
#
# # Run the Tkinter event loop
# root.mainloop()
import threading

import cv2
import os
import tkinter as tk
from tkinter import messagebox, ttk
from ttkthemes import ThemedTk
import delete
import detect
import game

import random
import tkinter as tk
import time
from tkinter import messagebox
import winsound


# Load the pre-trained cascade classifier for face detection
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Create a directory to save the face images
output_dir = "faces"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to save the face image with the person's name
def save_face_image(image, name):
    filename = os.path.join(output_dir, name + ".jpg")
    cv2.imwrite(filename, image)
    print(f"Face image saved: {filename}")


# Function to handle the name entry pop-up
def enter_name_popup(frame):
    def submit_name():
        name = name_entry.get()
        if name:
            root.destroy()
            save_face_image(frame, name)
        else:
            messagebox.showwarning("Invalid Name", "Please enter a valid name.")

    # Create a pop-up window
    popup = tk.Toplevel(root)
    popup.title("Enter Name")
    popup.resizable(False, False)

    # Create a label and an entry widget
    name_label = ttk.Label(popup, text="Enter the person's name:")
    name_label.pack()
    name_entry = ttk.Entry(popup)
    name_entry.pack()

    # Create a submit button
    submit_button = ttk.Button(popup, text="Submit", command=submit_name)
    submit_button.pack()

    popup.mainloop()


# Function to handle the Create button click event
def create_button_click():
    # Capture image from camera
    camera = cv2.VideoCapture(0)

    while True:
        # Read frame from the camera
        ret, frame = camera.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 1:
            # Extract the face region
            (x, y, w, h) = faces[0]
            face_image = frame[y:y + h, x:x + w]

            # Display the frame
            cv2.imshow("Camera Feed", frame)

            # Prompt the user to enter the person's name
            enter_name_popup(face_image)

            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    camera.release()
    cv2.destroyAllWindows()


def create_screen():
    # Create a new window for the Create screen
    create_window = tk.Toplevel(root)
    create_window.title("Create")

    # Add content to the Create screen
    create_label = ttk.Label(create_window, text="Click Create button to start capturing face image.")
    create_label.pack()

    create_button = ttk.Button(create_window, text="Create", command=create_button_click)
    create_button.pack()


def read_screen():
    fr = detect.FaceRecognition()
    fr.run_recognition()


def delete_screen():
    fr = delete.FaceRecognition()
    fr.run_recognition()


def admin_screen():
    # Create a new window for the Admin screen
    admin = tk.Toplevel(root)
    admin.title("Admin Panel")
    admin.resizable(False, False)

    def close_window():
        admin.destroy()

    fr = detect.FaceRecognition()
    if fr.admin():
        create_button = ttk.Button(admin, text="Create", command=create_screen)
        create_button.pack()

        read_button = ttk.Button(admin, text="Read", command=read_screen)
        read_button.pack()

        update_button = ttk.Button(admin, text="Update")
        update_button.pack()

        delete_button = ttk.Button(admin, text="Delete", command=delete_screen)
        delete_button.pack()
    else:
        no_admin_label = ttk.Label(admin, text="You don't have access to this screen.")
        no_admin_label.pack()

        close_button = ttk.Button(admin, text="OK", command=close_window)
        close_button.pack()

def generate_report():
    result = str(name) + " " + str(experison) + " " + str(count) + " " + str(yolo)
    with open(f'{str(name)}.txt', 'w') as file:
        file.write(result)

def play_screen():
    global player_row, player_col, start_time,finaltime
    global obstacle_density, maze
    global name, count, yolo, experison, time,stop_event , stop_thread1
    stop_thread1 = False,


    #thread1
    def game_thread():
        global player_row, player_col, start_time
        global obstacle_density, maze,stop_event , stop_thread1




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
            global player_row, player_col, start_time ,finaltime,stop_event,stop_thread1

            key = event.keysym.lower()

            # Move the player based on the input
            if key == "up" and player_row > 0 and maze[player_row - 1][player_col] != "#":
                player_row -= 1  # Move up
            elif key == "down" and player_row < maze_height - 1 and maze[player_row + 1][player_col] != "#":
                player_row += 1  # Move down
            elif key == "left" and player_col > 0 and maze[player_row][player_col - 1] != "#":
                player_col -= 1  # Move left
            elif key == "right" and player_col < maze_width - 1 and maze[player_row][player_col + 1] != "#":
                player_col += 1  # Move right

            # Check if the player reached the goal
            if player_row == goal_row and player_col == goal_col:
                canvas.itemconfigure(maze_cells[goal_row][goal_col], fill='green')
                end_time = time.time()
                elapsed_time = end_time - start_time
                status_label.config(
                    text="Congratulations! You reached the goal in {:.2f} seconds.".format(elapsed_time))
                play.unbind("<KeyPress>")
                window = tk.Tk()
                stop_thread1=True
                # Create a button for generating the report
                submit_button = tk.Button(window, text="Generate Report", command=generate_report)
                submit_button.pack()
            else:
                cell = maze[player_row][player_col]
                if cell == "#" or cell == "red":
                    status_label.config(text="Game Over! You lost.")
                    play.unbind("<KeyPress>")
                    messagebox.showinfo("Game Over", "You lost the game.")
                    play.destroy()

            update_display()

        def increase_obstacle_density():
            global obstacle_density, maze
            for i in range(maze_height):
                for j in range(maze_width):
                    if maze[i][j] == "#":
                        if random.random() < obstacle_density:
                            maze[i][j] = "red"  # Change the obstacle color to red
            obstacle_density -= 0.02  # Increase the obstacle density reduction for faster growth
            play.after(5000, increase_obstacle_density)

        # Create the game window
        play = tk.Toplevel(root)
        play.title("Game Screen")
        play.resizable(False, False)

        # Create the canvas to draw the maze
        cell_size = 40  # Adjust the size of each cell as desired
        canvas_width = maze_width * cell_size
        canvas_height = maze_height * cell_size
        canvas = tk.Canvas(play, width=canvas_width, height=canvas_height)
        canvas.pack()

        # Create the maze
        initial_obstacle_density = 0.2  # Initial obstacle density
        maze = generate_maze(maze_width, maze_height, initial_obstacle_density)
        maze_cells = []
        player_row = random.randint(0, maze_height - 1)
        player_col = random.randint(0, maze_width - 1)
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
        goal_row = random.randint(0, maze_height - 1)
        goal_col = random.randint(0, maze_width - 1)
        maze[player_row][player_col] = 'P'  # Player
        maze[goal_row][goal_col] = 'G'  # Goal

        # Bind the keypress event
        play.bind("<KeyPress>", on_key_press)
        play.focus_set()

        # Create a status label
        status_label = tk.Label(play, text="Use arrow keys to navigate. Reach the goal (blue cell).")
        status_label.pack()

        # Start the timer
        start_time = time.time()

        # Increase the size and complexity of the obstacles randomly over time
        obstacle_density = initial_obstacle_density

        # Start increasing obstacle density after 5 seconds
        play.after(5000, increase_obstacle_density)

        # Play the ringtone
        play_ringtone()

        # Update the display
        update_display()

    def game1_thread():
        global name , count , yolo , experison
        global stop_thread1
        fr = game.Game1()
        fr.run()

        name = fr.face_name
        count = fr.count
        yolo = fr.yolo_now
        experison = fr.face_expersion_now


    game_thread = threading.Thread(target=game_thread)

    game_thread.start()
    # Start the game1 thread
    game1_thread = threading.Thread(target=game1_thread)
    game1_thread.start()

    # game_thread.join()
    # game1_thread.join()

    # result = str(name) + " " + str(experison) + " " + str(count) + " " + str(yolo)




def how_to_play():
    play = tk.Toplevel(root)
    play.title("How To Play")
    play.resizable(False, False)

    no_admin_label = ttk.Label(play, text="LEASER WITH RIGHT AND DRAW CIRCLE WITH RIGHT MEAN HELP AND COUNT IT")
    no_admin_label.pack()


# Main application window
root = ThemedTk(theme="arc")  # Apply a theme (e.g., "arc", "radiance", "equilux")

root.title("Fire System")
root.resizable(False, False)

play_button = ttk.Button(root, text="Play A Game", command=play_screen)
play_button.pack(pady=10)

admin_button = ttk.Button(root, text="Admin", command=admin_screen)
admin_button.pack(pady=10)

how_to_play_button = ttk.Button(root, text="How to Play", command=how_to_play)
how_to_play_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()


