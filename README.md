## ToDo Object Detection
- [x] Visualize some images with GT boxes, SS boxes and EB boxes, to understand the poor recall/MABO.
- [x] Play around with input parameters for edge boxes and SS, since our recall/MABO is not very impressive right now. Most easily done like in the section 'edge boxes' in test_oe.ipynb (should probably be moved to HPC for speed though). Should probably reduce to a subset of images so that different parameters can easily be tested
- [x] Make validation set and change code to use it
- [x] Create a test loop and implement some score functions

- [x] Start training seriously, and visualize results to see if we can improve anything. Stuff to play around with
    - Weighting between Classification Loss and Regression loss
    - Consider smooth L1 Hubert loss for BBOX regression (suggested by ChatGPT)
    - Thresholds for background/foreground IOU at train and test time


## TASKS
- Implement AP (Ø)
- Clean up dataloader for RAM (Frederik) ---DONE ---
- Try to understand what goes wrong in Loss/train/main (Mads)
- Make "different steps"-plot that we can reference. Ideas for steps (probably don't include all): Initial image --> Image with GT -->  Image with proposals --> Proposals classified and adjusted --> Final detection after NMS (Alex)
- Make architecture flow chart graphic (Image -> proposal generator -> stack of proposals -> efficient net w two heads -> classification + t-values
- Make plot of recall/mabo (Ø)
- Make plot of train/validation loss history (Alex)
- Make final test plots given a model (Ø)

- Train some selected models for a long time and save detailed loss history, including accuracy. Maybe IOU before and after bbox adjustment?


## Stuff to include on the poster
- EfficientNet architecture
- A NICE graph of Recall/MABO vs proposals-included. There should probably be one for SS and one for EB. Could include a few different thresholds in each plot for the recall (MABO is threshold-independent)
- A viualization of an image (train) with boxes on top (GT boxes, and a subset of other options (foreground/background, according to IOU/model, SS/EB, adjusted with bbox regression or not))
- A visualization of an image (test) with boxes on top?
- Loss history for classification and regression
- Consider making a flow chart of everything that goes on at test time (images goes into SS -> bboxes... Each bbox goes through Network... boxes are adjusted according to bbox regression... NMS... etc)


Schematic:
- Introduktion
	- VERY short
	- Dataset (maybe image with the different steps)

- Arkitektur (figur)
	- Proposal Generation
	- EfficientNet m. to heads, t_vals

- Proposal generation
	- plot of recall/mabo (think about reordering proposals)
	- Show some examples of proposals

- Training approach
	- Class balance dataloader
	- Loss choice (lambda)
	+ Infobox on training: Loss, optimizer, epochs (in plot), stratification…

- Results
	- Train/validation loss history. Maybe have for regression, classification, but definitely have sum. Include also classification accuracy
	- AP history or final AP table
	- All positive detections in an example image (maybe before and after NMS)
    - Afterwards, show same results on test data (maybe not even needed for validation)

Should we have?
	- Data Augmentation (part of ‘story’, if necessary)
	- Example of generated/selected proposals?
	- Loss history, AP history…
	- Confusion matrix on validation set (pure classification)
	- Confusion matrix on validation set or test set (including proposal generation, NMS). Show AP

## Questions
- What recall/MABO is acceptable?
- Is overall stratified sampling ok, or should it be per-batch?
- How should we sub-select proposals from SS and EB? Apply the method also when plotting Recall/MABO. I wrote about an idea in test_oe
- Should we create our own CNN
- Can we 'flip' the presentation, i.e. start with results
    - Brief answer: depends on the lecturer, but always start with task definition ('we want to solve this task'. Maybe one full sentence)
