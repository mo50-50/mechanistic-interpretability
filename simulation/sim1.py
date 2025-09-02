import time

import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import numpy as np
from ml_models import *
import torch
import matplotlib.colors as mcolors
from tkinter import messagebox

# initial visualization part
# Create Tkinter window
root = tk.Tk()
root.title("Interactive ML Visualization")

# create left and right panels for buttons/controls and plot respectively
left_panel = tk.Frame(root)
left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
right_panel = tk.Frame(root)
right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

# create the outer window in matplotlib
fig = plt.Figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('Interactive Classification')
# Embed Matplotlib Plot in Tkinter Window
canvas = FigureCanvasTkAgg(fig, master=right_panel)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)




###############################
# creating some global variables
num_classes = 1
max_class_num = 5
selected_class = 1
class_colors = ['red', 'blue', 'green', 'orange', 'purple']
ood_class_color = 'black'
ood_class_checked = False
lightred = '#FF9999'
lightblue = '#ADD8E6'
lightgreen = '#90EE90'
points_plts = [None for _ in range(max_class_num)]
ood_points_plt = None
ood_points = []
max_width = 20
max_depth = 10
epsilon = 1e-8
show_simulation = True
# create image onto the plot
image = ax.imshow(np.zeros((100, 100, 3)), extent=(0, 1, 0, 1), origin='lower', alpha=0.5)
# do not show the image
image.set_visible(False)
# create a bunch of stars to represent max points
stars = [ax.scatter(0.5, 0.5, marker='*', s=200, c='gold', edgecolors=class_colors[i]) for i in range(max_class_num)]
# set them invisible
for star in stars:
    star.set_visible(False)

train_object = None

# create lines to plot gradient paths later
lines = []
for i in range(max_class_num):
    line = Line2D([], [], color=class_colors[i], linestyle='dashed', linewidth=2)
    ax.add_line(line)
    lines.append(line)
    line.set_visible(False)

# Add drop down menu for selecting the number of classes
# create a num_classes frame to add teh label and dropdown menu
num_classes_frame = tk.Frame(left_panel)
num_classes_frame.pack(pady=1)

# Add a label for the dropdown menu
num_classes_label = tk.Label(num_classes_frame, text="Number of Classes:", bg='lightblue')
num_classes_label.pack(side=tk.LEFT, padx=1)

def num_classes_changed(value):
    print("num classes changed to:", value)
    global num_classes
    num_classes = int(value)
    update_selection_dropdown() # update the dropdown menu for selecting the class
    clear_selected_regions() # clear the selected regions
# Tkinter variable to store the selection
num_class_str_var = tk.StringVar(root)
num_class_str_var.set("1")  # default value

# Create the dropdown
num_classes_dropdown = tk.OptionMenu(num_classes_frame, num_class_str_var, *range(1, max_class_num+1), command=num_classes_changed)
# change color of dropdown menu
num_classes_dropdown.config(bg='lightblue')
num_classes_dropdown.pack(pady=3)

#### same thing for the frame that coontains selected class
selected_class_frame = tk.Frame(left_panel)
selected_class_frame.pack(pady=1)

# Add a label for the dropdown menu
selected_class_label = tk.Label(selected_class_frame, text="Selected Class:", bg=lightred)
selected_class_label.pack(side=tk.LEFT, padx=1)
def selected_class_changed(value):
    print("selected class changed to:", value)
    global selected_class
    selected_class = int(value)
def update_selection_dropdown():
    global num_classes, selected_class
    selected_class_str_var.set("1")  # reset selection
    selected_class = 1 # reset selection
    menu = selected_class_dropdown["menu"]
    menu.delete(0, "end")  # remove old options
    for i in range(1, num_classes + 1):
        menu.add_command(label=str(i), command=tk._setit(selected_class_str_var, str(i), callback=selected_class_changed))
# Tkinter variable to store the selection
selected_class_str_var = tk.StringVar(root)
selected_class_str_var.set("1")  # default value
# Create the dropdown
selected_class_dropdown = tk.OptionMenu(selected_class_frame, selected_class_str_var, *range(1, num_classes+1), command=selected_class_changed)
# change color of dropdown menu
selected_class_dropdown.config(bg=lightred)
selected_class_dropdown.pack(pady=1)

#######################################################
#######################################################
# add the drawing tool for the classes to the plot

shape_blobs = [[] for _ in range(max_class_num)] # all the shapes drawn by user for each class

# define a function to handle lasso selection
def on_lasso_select(verts):
    global selected_class
    # get the indices of the selected points
    print("Lasso drawn with", len(verts), "vertices")
    path = Path(verts)
    shape_blobs[selected_class-1].append(path)  # store path

    # draw the selected region
    patch = plt.Polygon(verts, closed=True, fill=True, color=class_colors[selected_class-1], alpha=0.3) # color of the class
    ax.add_patch(patch)
    fig.canvas.draw_idle()


# initialize the lasso selector
lasso = LassoSelector(ax, on_lasso_select)

# add clear button to clear all selected regions
def clear_selected_regions():
    global shape_blobs, points_plts, ood_points_plt, ood_points, show_simulation
    # stop simulation
    show_simulation = False

    shape_blobs = [[] for _ in range(max_class_num)]
    # delete all drawings on plt
    for patch in ax.patches:
        patch.remove()

    # remove all scatter plots
    for i in range(len(points_plts)):
        if points_plts[i] is not None:
            points_plts[i].remove()
            points_plts[i] = None
    # remove ood scatter plot
    if ood_points_plt is not None:
        ood_points_plt.remove()
        ood_points_plt = None

    # set image to imvisible
    image.set_visible(False)

    # ax.clear()
    # remove sample points
    for i in range(max_class_num):
        sample_points[i] = []

    # remove ood pints
    ood_points = []

    # set stars and lines invisible
    for star in stars:
        star.set_visible(False)
    for line in lines:
        line.set_visible(False)

    fig.canvas.draw_idle()
clear_button = tk.Button(left_panel, text="Clear", command=clear_selected_regions, bg='red')
clear_button.pack(pady=1)

########################################
########################################
# add sample points to the plot
sample_points = [[] for _ in range(max_class_num)]
# create a dropdown menu for the sampling density
# create frame for sample menu
sample_frame = tk.Frame(left_panel)
sample_frame.pack(pady=10)
sampling_density_label = tk.Label(sample_frame, text="Sampling Density:")
sampling_density_label.pack(side=tk.LEFT)
sampling_density_var = tk.StringVar()
sampling_density_var.set("1000")  # default value
sampling_density_dropdown = tk.OptionMenu(sample_frame, sampling_density_var, "100", "200", "500", "1000", "2000")
sampling_density_dropdown.config(width=4)
sampling_density_dropdown.pack(side=tk.RIGHT)

# create a button to sample points
def sample_points_func():
    global sample_points, ood_points_plt, ood_points

    # delete all points beforehand
    for i in range(max_class_num):
        sample_points[i] = []
    # delete teh points from the plt
    for points_plt in points_plts:
        if points_plt is not None:
            print('removed popints')
            points_plt.remove()
    canvas.draw()

    num_samples = int(sampling_density_var.get())
    points = np.random.rand(num_samples, 2)
    # check which class the points belong to
    all_classes_mask = np.zeros(num_samples, dtype=bool) # to see which points will not be added at all to include in ood class
    for clas in range(len(shape_blobs)):
        mask = np.zeros(num_samples, dtype=bool)
        for blob in shape_blobs[clas]:
            mask = mask | blob.contains_points(points)
        all_classes_mask = all_classes_mask | mask
        sample_points[clas] = points[mask]
        # plot the points
        points_plts[clas] = ax.scatter(sample_points[clas][:, 0], sample_points[clas][:, 1], color=class_colors[clas], s=1)
    canvas.draw()

    # add points to ood_points
    ood_points = points[~all_classes_mask]
    if ood_class_checked:
        ood_points_plt = ax.scatter(ood_points[:, 0], ood_points[:, 1], color=ood_class_color, s=1)
        canvas.draw()

sample_button = tk.Button(left_panel, text="Sample Points", command=sample_points_func)
sample_button.pack(pady=1)


##########################################
##########################################
# now adding ood class
def add_ood_class():
    global ood_class_checked, ood_points_plt, ood_class_color
    ood = bool(ood_var.get())
    ood_class_checked = ood

    if ood:
        ax.set_facecolor((0, 0, 0, 0.2))
        # plot the points
        print('len', len(ood_points))
        if len(ood_points) > 0:
            print('ood points not empty', len(ood_points))
            ood_points_plt = ax.scatter(ood_points[:, 0], ood_points[:, 1], color=ood_class_color, s=1)
        canvas.draw()
        print('ood class added')
    else:
        ax.set_facecolor("white")  # reset background
        # erase plot if found
        if ood_points_plt:
            ood_points_plt.remove()
            ood_points_plt = None
        canvas.draw()

# adding a checkbox for ood class
ood_var = tk.IntVar()
ood_check = tk.Checkbutton(left_panel, text="OOD Class", variable=ood_var, command=add_ood_class)
ood_check.pack(pady=1)
# changing color to blue
ood_check.config(fg="blue")


############################################################################
############################################################################
############################################################################
# NOW FOR THE MACHINE LEARNING PART

# make a drop down menu for the optimizer
# add frame first
optimizer_frame = tk.Frame(left_panel)
optimizer_frame.pack(pady=1)
# add label
optimizer_label = tk.Label(optimizer_frame, text="Optimizer:")
optimizer_label.pack(side=tk.LEFT, padx=5)
# add drop down menu
optimizers = ["Adam", "SGD", "RMSprop", "Adagrad", "Adadelta"]
optimizer_var = tk.StringVar()
optimizer_var.set("Adam")  # default value
optimizer_menu = tk.OptionMenu(optimizer_frame, optimizer_var, *optimizers)
optimizer_menu.pack(pady=1)

# add a slider for the width, it should accept any number >= 1 and <= max_width
# add frame first
width_frame = tk.Frame(left_panel)
width_frame.pack(pady=1)
# add label
width_label = tk.Label(width_frame, text="Model Width:")
width_label.pack(side=tk.LEFT, padx=5)
# add slider
width_var = tk.IntVar(value=10) # intial value
width_slider = tk.Scale(width_frame, from_=1, to=max_width, orient=tk.HORIZONTAL, variable=width_var)
width_slider.pack(pady=1)
# color
width_slider.config(fg="blue")

# add a slider for the depth, it should accept any number >= 1 and <= max_depth
# add frame first
depth_frame = tk.Frame(left_panel)
depth_frame.pack(pady=1)
# add label
depth_label = tk.Label(depth_frame, text="Model Depth:")
depth_label.pack(side=tk.LEFT, padx=5)
depth_var = tk.IntVar(value=5) # initial value
depth_slider = tk.Scale(depth_frame, from_=1, to=max_depth, orient=tk.HORIZONTAL, variable=depth_var)
depth_slider.pack(pady=1)
# color
depth_slider.config(fg="blue")

# add a text box for the learning rate, it should accept any number >= 0.0001 and <= 1.
# there should be a value next to the text box indicating the current value, if the user enters
# a valid value the value should update, if the user enters an invalid value the value should not update
def validate_learning_rate(learning_rate_var, learning_rate_entry):

    val = learning_rate_entry.get()
    if len(val) > 6:
        learning_rate_entry.delete(0, tk.END)
        # show an error message to user
        messagebox.showerror("Error", "Learning rate must be be at most 6 digits long.")
        return
    if not val.startswith("0."):
        learning_rate_entry.delete(0, tk.END)
        # show an error message to user
        messagebox.showerror("Error", "Learning rate must be less than 1 and start with '0.'")
        return
    if not val[2:].isdigit():
        learning_rate_entry.delete(0, tk.END)
        # show an error message to user
        messagebox.showerror("Error", "Learning rate characters after the decimal point must all be digits.")
        return
    learning_rate_var.set(float(val))
    learning_rate_value_label.config(text=str(learning_rate_var.get()))

# add frame first
learning_rate_frame = tk.Frame(left_panel)
learning_rate_frame.pack(pady=1)
# add label
learning_rate_label = tk.Label(learning_rate_frame, text="Learning Rate:")
learning_rate_label.pack(side=tk.LEFT, padx=0)
learning_rate_var = tk.DoubleVar()
learning_rate_entry = tk.Entry(learning_rate_frame, textvariable=learning_rate_var)
learning_rate_entry.pack(side=tk.LEFT, padx=0)
learning_rate_var.set(0.001)
learning_rate_entry.bind("<Return>", lambda event: validate_learning_rate(learning_rate_var, learning_rate_entry))
# add a label to display the current value
learning_rate_value_label = tk.Label(learning_rate_frame, text=str(learning_rate_var.get()))
learning_rate_value_label.pack(side=tk.LEFT, padx=0)
# limit the width of the entry widget
learning_rate_entry.config(width=10)
# fix the width of the value label widget
learning_rate_value_label.config(width=6)

# add activation function  selection
activations = ["relu", "sigmoid", "tanh"]
activation_function_frame = tk.Frame(left_panel)
activation_function_frame.pack(pady=1)
activation_function_label = tk.Label(activation_function_frame, text="Activation Function:")
activation_function_label.pack(side=tk.LEFT, padx=0)
activation_function_var = tk.StringVar()
activation_function_var.set("relu")
activation_function_menu = tk.OptionMenu(activation_function_frame, activation_function_var, *activations)
activation_function_menu.pack(side=tk.LEFT, padx=0)

# add loss function selection
loss_functions = ["CrossEntropy", "MSELoss"]
loss_function_frame = tk.Frame(left_panel)
loss_function_frame.pack(pady=1)
loss_function_label = tk.Label(loss_function_frame, text="Loss Function:")
loss_function_label.pack(side=tk.LEFT, padx=0)
loss_function_var = tk.StringVar()
loss_function_var.set("CrossEntropy")
loss_function_menu = tk.OptionMenu(loss_function_frame, loss_function_var, *loss_functions)
loss_function_menu.pack(side=tk.LEFT, padx=0)

# add button to stop the simulation
def stop_simulation():
    global show_simulation
    show_simulation = False
stop_button = tk.Button(left_panel, text="Stop", command=stop_simulation)
stop_button.pack(pady=1)
# make it red
stop_button.config(bg="red")

##############################################################################
def update_plot(train_object, colors, im):
    global show_simulation
    if not show_simulation:
        return
    # train the model
    # for i in range(10):
    train_object.train_one_epoch()

    # predict the probabilities for each point in the grid
    probabilities = train_object.predict(100)
    # convert probabilities to numpy array
    probabilities = np.array(probabilities)
    # pass the probabilities through a softmax function
    probabilities = np.exp(probabilities) / np.sum(np.exp(probabilities), axis=2, keepdims=True)
    # transpose it because pytorch and matplotlib have different conventions
    probabilities = np.transpose(probabilities, (1, 0, 2))
    # print(probabilities[:10, :10, :])
    # multiply the probabilities by the colors, colors:(c x 3), probabilities:(n x n x c)
    # result should be: (n x n x 3)
    weighted_colors = np.tensordot(probabilities, colors, axes=([2], [0]))
    # # pass the weighted colors through a softmax function
    # weighted_colors = np.exp(weighted_colors) / np.sum(np.exp(weighted_colors), axis=2, keepdims=True)
    # plot it onto the plot
    im.set_data(weighted_colors)



    # update the plot
    canvas.draw()
    # sleep and recursive call to update again
    root.after(10, lambda: update_plot(train_object, colors, im))

def run_model():
    global show_simulation, train_object

    # check the number of sample points
    num_samples = 0
    for i in range(num_classes):
        num_samples += len(sample_points[i])
    if ood_class_checked:
        num_samples += len(ood_points)
    # return error message if no sample points
    if num_samples == 0:
        messagebox.showerror("Error", "Please add some sample points first.")
        return


    show_simulation = True

    # hide the stars and lines by invisible
    for star in stars:
        star.set_visible(False)
    for line in lines:
        line.set_visible(False)

    # initializing the model
    out_size = num_classes + 1 if ood_class_checked else num_classes
    model = NeuralNetwork(num_classes=num_classes, num_features=2,
                          num_hidden_layers=depth_var.get(), num_neurons=width_var.get(),
                          activation_function=activation_function_var.get(), ood=ood_class_checked)
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    # define dataset
    dataset = Data(data_points=sample_points, num_classes=num_classes, ood_array=ood_points, ood=ood_class_checked)

    # the training loop should begin here, each iteration the train_epoch function should be called
    # and the predict function should be called, it will output a 2d array, each element containing
    # a vector of probabilities for each class, the last element is the ood class if ood is checked
    # for each point in the grid, a color, (r, g, b) tuple should be calculated as a weighted average
    # of the colors of the classes, the weights being the probabilities of each class
    # then the plot should be updated to draw with alpha=0.5, the color of the point being the weighted average
    train_object = Training(model=model, dataset=dataset, batch_size=16,
                            learning_rate=learning_rate_var.get(), loss_function=loss_function_var.get(),
                            optimizer=optimizer_var.get())

    # train the model
    # calculate the colors for each point in the grid
    # first get the rgb values for all the color classes
    colors = []
    for i in range(num_classes):
        colors.append(mcolors.to_rgb(class_colors[i]))
    if ood_class_checked:
        colors.append(mcolors.to_rgb(ood_class_color))

    # convert the colors to a numpy array
    colors = np.array(colors)
    # normalize the colors by row
    colors = colors / (np.sum(colors, axis=1, keepdims=True) + epsilon)

    # set the image visible
    image.set_visible(True)

    # call the update function to update the plot
    update_plot(train_object, colors, image)


# add button to run the model
run_button = tk.Button(left_panel, text="Run Model", command=run_model)
run_button.pack(pady=10)
# make it green
run_button.config(bg="green", fg="white")

############################################################################
# make the gadient ascent visualization part
############################################################################
# add a button for gradient ascent
def gradient_ascent_visualization():
    global train_object, show_simulation
    if train_object is None:
        # show error to user
        tk.messagebox.showerror("Error", "Please train the model first")
        return
    # stop the simulation to show these plots
    show_simulation = False

    paths = train_object.gradient_ascent(3000, 0.001, num_classes)
    # permute the first and second dimensions of the paths
    paths = np.transpose(paths, (1, 0, 2))

    # plot the paths as dashed lines with the color of each class
    # print('paths', paths.shape)
    for i in range(num_classes):
        # print(f"Class {i} path: {paths[i]}")
        path = paths[i]
        # print('path[-1]', path[-1].shape)
        x = path[:, 0]
        y = path[:, 1]
        lines[i].set_data(x, y)
        lines[i].set_visible(True)
        # show the star of each class in its last position in the path
        stars[i].set_offsets(path[-1].reshape(1, -1))
        stars[i].set_visible(True)

    # draw the changes
    canvas.draw()


gradient_ascent_button = tk.Button(left_panel, text="Gradient Ascent", command=gradient_ascent_visualization)
gradient_ascent_button.pack(pady=10)
gradient_ascent_button.config(bg="blue", fg="white")


# Run the Tkinter Main Loop
root.mainloop()
