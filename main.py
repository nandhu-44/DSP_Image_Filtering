import cv2
from noise import add_gaussian_noise
from filters import apply_average_filter, apply_gaussian_filter
from psnr import calculate_psnr

# Read the image
image_path = "./assets/dog.webp"
original_image = cv2.imread(image_path)
noisy_image = add_gaussian_noise(original_image)
cv2.imwrite("./assets/noisy_image.jpg", noisy_image)

# Apply average filters of different sizes and calculate PSNR
filter_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] 
psnr_file = open("./psnr.txt", "w")
psnr_file.write("PSNR values for different filter sizes\n\n")

for size in filter_sizes:
    avg_filtered_image = apply_average_filter(noisy_image, size)
    gaussian_filtered_image = apply_gaussian_filter(noisy_image, size)

    # Save the filtered images
    cv2.imwrite(f"./avg_results/average_filtered_image_{size}.jpg", avg_filtered_image)
    cv2.imwrite(
        f"./gaussian_results/gaussian_filtered_image_{size}.jpg",
        gaussian_filtered_image,
    )

    # Calculate and print PSNR
    psnr_value = calculate_psnr(original_image, avg_filtered_image)
    psnr_value_gaussian = calculate_psnr(original_image, gaussian_filtered_image)
    print(
        f"PSNR for average filter (size {size}): {psnr_value} dB and for gaussian filter (size {size}): {psnr_value_gaussian} dB"
    )

    psnr_file.write(
        f"Size: {size}:\n \t- Avg PSNR: {psnr_value} dB \t- Gaussian PSNR: {psnr_value_gaussian} dB\n\n"
    )

psnr_file.close()
