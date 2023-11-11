from PIL import Image
import os

def convert_to_grayscale(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    # Iterate through each file in the input folder
    for file in files:
        # Construct the full path of the input file
        input_path = os.path.join(input_folder, file)

        # Check if the file is a valid image file
        if os.path.isfile(input_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Open the image
            img = Image.open(input_path)

            # Convert the image to grayscale
            grayscale_img = img.convert('L')

            # Construct the full path of the output file
            output_path = os.path.join(output_folder, file)

            # Save the grayscale image
            grayscale_img.save(output_path)

            print(f"Converted: {input_path} -> {output_path}")

if __name__ == "__main__":
    # Set the input and output folders
    input_folder = "input_rgb_folder"
    output_folder = "output_gray_folder"

    # Call the function to convert images
    convert_to_grayscale(input_folder, output_folder)
