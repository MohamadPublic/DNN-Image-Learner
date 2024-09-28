import pygame
import torch
import src.models as models
from src.imageDataset import ImageDataset
from torch.utils.data import DataLoader
from torch import optim, nn
import matplotlib.pyplot as plt
from src.videomaker import renderModel
from tqdm import tqdm
import os
import imageio

# Drawing settings
canvas_width, canvas_height = 1000, 800
bg_color = (0, 0, 0)  # Black background
line_color = (255, 255, 255)  # White drawing color
line_width = 10

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((canvas_width, canvas_height))
pygame.display.set_caption("Draw Your Image")
screen.fill(bg_color)  # Fill the background with black
drawing = False
last_pos = None

# Function to save the drawn image
def save_drawing():
    pygame.image.save(screen, "DatasetImages/drawn_image.png")

# Main loop for drawing
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            save_drawing()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
            last_pos = event.pos
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
        elif event.type == pygame.MOUSEMOTION:
            if drawing and last_pos is not None:
                pygame.draw.line(screen, line_color, last_pos, event.pos, line_width)
                last_pos = event.pos
    
    # Update the display
    pygame.display.flip()

# Quit pygame
pygame.quit()

# Define variables
image_path = 'DatasetImages/drawn_image.png'
hidden_size = 300
num_hidden_layers = 30
batch_size = 8000
lr = 0.001
num_epochs = 40
proj_name = 'helloworld_skipconn'
save_every_n_iterations = 9
scheduler_step = 3

# Create the dataset and data loader
dataset = ImageDataset(image_path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
resx, resy = dataset.width, dataset.height
linspace = torch.stack(torch.meshgrid(torch.linspace(-1, 1, resx), torch.linspace(1, -1, resy)), dim=-1).cuda()
linspace = torch.rot90(linspace, 1, (0, 1))

# Create the model
model = models.SkipConn(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers).cuda()

# Create the loss function and optimizer
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.5)

# Train the model
iteration, frame = 0, 0
for epoch in range(num_epochs):
    epoch_loss = 0
    for x, y in tqdm(loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        x, y = x.cuda(), y.cuda()

        # Forward pass
        y_pred = model(x).squeeze()

        # Compute loss
        loss = loss_func(y_pred, y)
        epoch_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()

        # Save an image of the model every n iterations
        if iteration % save_every_n_iterations == 0:
            os.makedirs(f'./frames/{proj_name}', exist_ok=True)
            plt.imsave(f'./frames/{proj_name}/{frame:05d}.png', renderModel(model, resx=resx, resy=resy, linspace=linspace), cmap='magma', origin='lower')
            frame += 1
        iteration += 1
        
    scheduler.step()

    # Log the average loss per epoch
    print(f'Epoch {epoch+1}, Average Loss: {epoch_loss / len(loader)}')

def save_video_with_imageio(proj_name='helloworld_skipconn', output_file='Video.mp4'):
    output_directory = os.getcwd()  # Get current working directory
    output_path = os.path.join(output_directory, output_file)

    # Define the path to the frames
    frames_directory = f"./frames/{proj_name}/"
    
    # Check if the frame directory exists
    if not os.path.exists(frames_directory):
        print(f"Directory {frames_directory} does not exist.")
        return

    # List existing frames
    existing_frames = os.listdir(frames_directory)
    print("Existing frames:", existing_frames)

    # Use imageio to save the video
    with imageio.get_writer(output_path, fps=30) as writer:
        for i in range(10000):  # Adjust based on your number of frames
            frame_path = f"{frames_directory}/{i:05d}.png"
            if os.path.exists(frame_path):
                image = imageio.imread(frame_path)
                writer.append_data(image)
            else:
                print(f"Frame {frame_path} does not exist. Stopping.")
                break  # Stop if the frame does not exist

    print(f"Video saved successfully at {output_path}")

# Example usage
save_video_with_imageio('helloworld_skipconn')
