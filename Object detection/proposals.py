import json
import numpy as np
import matplotlib.pyplot as plt

def required_proposals(threshold=0.7):
    # Load data 
    data = []

    # Loop over the range of files and load each JSON
    for i in range(1, 666):  # Goes from 1 to 665 inclusive
        file_path = f'Potholes/annotated-images/img-{i}_ss.json'
        try:
            with open(file_path, 'r') as file:
                image_data = json.load(file)
                data.append(image_data)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            continue
        except json.JSONDecodeError:
            print(f"File {file_path} is not a valid JSON.")
            continue
    

    # Proposal limits
    proposal_limits = np.arange(10, 3001, 50)
    
    # Initialize list to store the results at each threshold
    positive_proposals_list = np.zeros((665,len(proposal_limits)))

    for img_index, image_data in enumerate(data):
        # Initialize counters for this image
        extracted_proposals = 0
        positive_proposals = 0
        next_limit_index = 0  # Start with the first limit in proposal_limits

        # Process proposals in the current image
        for data_dict in image_data:
            extracted_proposals += 1
            if data_dict['iou'] > threshold:
                positive_proposals += 1

            # Check if we have reached the current proposal limit
            if next_limit_index < len(proposal_limits) and extracted_proposals == proposal_limits[next_limit_index]:
                # Store the counts at this limit for the current image
                positive_proposals_list[img_index, next_limit_index] = positive_proposals
                
                # Move to the next limit
                next_limit_index += 1

        # Fill in the remaining limits with the last recorded counts if image_data has fewer proposals
        while next_limit_index < len(proposal_limits):
            positive_proposals_list[img_index, next_limit_index] = positive_proposals
            next_limit_index += 1

    # Calculate the mean positive proposals at each limit across all images
    mean_positive_proposals = np.mean(positive_proposals_list, axis=0)

    # Print the results
    print("Mean positive proposals at each limit:", mean_positive_proposals)

     # Plot the mean positive proposals
    plt.figure(figsize=(10, 6))
    plt.plot(proposal_limits, mean_positive_proposals, marker='o', linestyle='-', color='b')
    plt.xlabel("Number of Proposals")
    plt.ylabel("Mean Positive Proposals")
    plt.title("Mean Positive Proposals at Different Proposal Limits")
    plt.grid(True)
    
    # Save the plot to the 'graphics' folder
    plt.savefig("graphics/mean_positive_proposals_plot.png")
    plt.close()

# Run the function
required_proposals()





