import os
from PIL import Image
import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="Image Generator",
    layout="wide",
    initial_sidebar_state="auto",
)

def list_folders_with_entry(entry_folder):
    """Lists all folders including subfolders starting from the entry_folder."""
    folders = [entry_folder]
    for root, dirs, _ in os.walk(entry_folder):
        for dir in dirs:
            folders.append(os.path.join(root, dir))
    return folders

# Retrieve folders for the sidebar
folders = list_folders_with_entry('images')
with st.sidebar:
    gallery_folder = st.selectbox("Image folder:", folders, index=0)
    target_folder = st.selectbox("Target folder:", folders, index=1)

# Filter the image files based on the supported extensions
supported_extensions = ['.png', '.jpg', '.jpeg']
image_files = [
    f for f in os.listdir(gallery_folder)
    if os.path.isfile(os.path.join(gallery_folder, f)) and any(f.lower().endswith(ext) for ext in supported_extensions)
]

def delete_image(image_path):
    """Deletes the image file at the specified path."""
    os.remove(image_path)

def move_image(source_folder, target_folder, file_name):
    """Moves an image from the source folder to the target folder."""
    # Construct the full file paths
    source_path = os.path.join(source_folder, file_name)
    target_path = os.path.join(target_folder, file_name)

    # Move the file
    os.rename(source_path, target_path)

num_columns = 4

# Display images in a grid with buttons to move or delete
for i in range(0, len(image_files), num_columns):
    columns = st.columns(num_columns)
    for col, image_file in zip(columns, image_files[i:i+num_columns]):
        image_path = os.path.join(gallery_folder, image_file)
        image = Image.open(image_path)

        # Resize image for thumbnail
        image.thumbnail((300, 300))

        # Display image with move and delete buttons
        with col:
            st.image(image)
            move_col, del_col = st.columns(2)
            move_col.button(
                label="Move to target",
                key=f"move_{image_file}",
                on_click=move_image,
                args=(gallery_folder, target_folder, image_file)
            )
            del_col.button(
                label="Delete image",
                key=f"delete_{image_path}",
                on_click=delete_image,
                args=(image_path,)
            )
