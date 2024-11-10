## ToDo Object Detection
- [] Visualize some images with GT boxes, SS boxes and EB boxes, to understand the poor recall/MABO.
- [] Play around with input parameters for edge boxes and SS, since our recall/MABO is not very impressive right now. Most easily done like in the section 'edge boxes' in test_oe.ipynb (should probably be moved to HPC for speed though). Should probably reduce to a subset of images so that different parameters can easily be tested
- [] Make validation set and change code to use it
- [] Create a test loop and implement some score functions

- [] Start training seriously, and visualize results to see if we can improve anything. Stuff to play around with
    - Weighting between Classification Loss and Regression loss
    - Consider smooth L1 Hubert loss for BBOX regression (suggested by ChatGPT)
    - Thresholds for background/foreground IOU at train and test time


## Stuff to include on the poster
- EfficientNet architecture
- A NICE graph of Recall/MABO vs proposals-included. There should probably be one for SS and one for EB. Could include a few different thresholds in each plot for the recall (MABO is threshold-independent)
- A viualization of an image (train) with boxes on top (GT boxes, and a subset of other options (foreground/background, according to IOU/model, SS/EB, adjusted with bbox regression or not))
- A visualization of an image (test) with boxes on top?
- Loss history for classification and regression
- Consider making a flow chart of everything that goes on at test time (images goes into SS -> bboxes... Each bbox goes through Network... boxes are adjusted according to bbox regression... NMS... etc)

## Questions
- What recall/MABO is acceptable?
- Is overall stratified sampling ok, or should it be per-batch?
- How should we sub-select proposals from SS and EB? Apply the method also when plotting Recall/MABO. I wrote about an idea in test_oe